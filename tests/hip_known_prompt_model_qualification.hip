// Explicit G4 known-prompt foundation qualification; never CTest or public CLI.
// Full heads, correctness only. No speed or quality admission.
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/q8_projection_manifest.hpp"
#include "strix/serial_session.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/serial_request.hpp"
#include "strix/serial_prefix_cache.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include <optional>
#include <utility>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::uint64_t context_tokens = 8192;
constexpr std::size_t transfer_chunk = 8U * 1024U * 1024U;

void require(bool condition, const char* reason) {
    if (!condition) throw std::invalid_argument(reason);
}
void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
std::string hex(const strix::ModelContentIdentity& value) {
    return strix::model_content_identity_hex(value);
}
std::string hash_file(const std::filesystem::path& path) {
    return hex(strix::sha256_file_content(path));
}
bool digest_text(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
std::string quote(std::string_view value) {
    std::string result = "\"";
    constexpr char digits[] = "0123456789abcdef";
    for (const char raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 32U) { result += "\\u00"; result += digits[c >> 4U]; result += digits[c & 15U]; }
        else result += static_cast<char>(c);
    }
    return result + '"';
}
struct Options {
    std::filesystem::path model, teacher, output;
    std::string model_sha, teacher_sha, profile = "q8-wave8";
};
Options options(int argc, char** argv) {
    Options result;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; i += 2) {
        require(i + 1 < argc, "every option requires a value");
        const std::string key = argv[i];
        require(std::find(seen.begin(), seen.end(), key) == seen.end(), "duplicate option");
        seen.push_back(key);
        const std::string value = argv[i + 1];
        if (key == "--model") result.model = value;
        else if (key == "--model-sha256") result.model_sha = value;
        else if (key == "--prompt-u32le") result.teacher = value;
        else if (key == "--prompt-sha256") result.teacher_sha = value;
        else if (key == "--output") result.output = value;
        else if (key == "--profile") result.profile = value;
        else throw std::invalid_argument("unknown qualification option: " + key);
    }
    require(!result.model.empty() && !result.teacher.empty() && !result.output.empty(),
        "required: --model --model-sha256 --prompt-u32le --prompt-sha256 --output");
    require(digest_text(result.model_sha) && digest_text(result.teacher_sha), "expected lowercase SHA256 pins required");
    require(result.profile == "q8-wave8" || result.profile == "q8-wave8-a8-q51", "unsupported closed profile");
    require(!std::filesystem::exists(result.output), "output directory must be fresh");
    return result;
}
std::vector<std::uint32_t> teacher_tokens(const Options& args, std::uint64_t vocabulary) {
    require(hash_file(args.teacher) == args.teacher_sha, "teacher SHA differs before execution");
    const auto bytes = std::filesystem::file_size(args.teacher);
    require(bytes == 16U || bytes == 20U,
        "G4 prompt must contain exactly four or five explicit u32le tokens");
    std::ifstream input(args.teacher, std::ios::binary);
    input.exceptions(std::ios::failbit | std::ios::badbit);
    std::vector<std::uint32_t> result;
    std::vector<std::byte> actual_bytes;
    actual_bytes.reserve(static_cast<std::size_t>(bytes));
    for (std::uint64_t offset = 0; offset < bytes; offset += 4U) {
        std::array<unsigned char, 4> word{};
        input.read(reinterpret_cast<char*>(word.data()), 4);
        for (const auto byte : word) actual_bytes.push_back(static_cast<std::byte>(byte));
        const auto token = static_cast<std::uint32_t>(word[0]) |
            (static_cast<std::uint32_t>(word[1]) << 8U) | (static_cast<std::uint32_t>(word[2]) << 16U) |
            (static_cast<std::uint32_t>(word[3]) << 24U);
        require(token < vocabulary, "teacher token outside model vocabulary");
        result.push_back(token);
    }
    require(hex(strix::sha256_bytes(actual_bytes)) == args.teacher_sha,
        "actual loaded teacher bytes differ from the expected pin");
    require(result.size() != 5 || result[2] == strix::ModelContract::qwen38_flash_next().ple_eos_token_id,
        "five-token G4 prompt requires the literal interior EOS input at index two");
    return result;
}

// Strongly retained by the native window. Destruction proves original-device
// quiescence before releasing stream, scratch, states, residency or the mapping.
struct Owners {
    std::shared_ptr<strix::HipSetupQuiescenceGuard> guard = std::make_shared<strix::HipSetupQuiescenceGuard>();
    hipStream_t stream = nullptr;
    strix::AllocationGenerationSource generations;
    strix::ModelContract model = strix::ModelContract::qwen38_flash_next();
    strix::ModelManifest manifest;
    strix::ExecutionTensorAccessPlan access;
    strix::CompiledSerialExecution serial;
    strix::CompiledKnownPrefixWindow window;
    strix::SerialSessionStateLayout layout;
    std::shared_ptr<strix::MappedModelFile> mapped;
    std::shared_ptr<strix::HipModelResidency> resident;
    std::shared_ptr<strix::HipRuntimeAllocator> allocator;
    std::shared_ptr<strix::RuntimeExecutionMemoryOwner> memory;
    strix::ExecutionMemoryBinding window_memory;
    strix::FileMappedPleLookupBackend ple;
    strix::PleHashContract ple_hash;
    strix::PleLookupPrefetchState prefetch;
    strix::ExecutionRuntimeResources resources;
    ~Owners() {
        guard->recover_or_fail_stop();
        if (stream && hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
        if (allocator && window_memory.arena_base)
            allocator->free_device(window_memory.arena_base, window_memory.arena_bytes);
    }
    strix::HipLaunchContext scope() const {
        strix::HipLaunchContext value;
        // Eager launchers use the retained session layout; every graph binding,
        // including graph context/vocabulary fields, remains explicitly empty.
        value.stream = stream;
        return value;
    }
};
strix::HipLauncherCatalogConfig configure(const Options& args, strix::ExecutionBindingRequest& binding,
                                          strix::ExecutionBufferRequest& buffers) {
    strix::HipLauncherCatalogConfig config;
    binding.minimum_qualification = strix::kernels::QualificationStatus::experimental;
    config.parallel_hyper_mix = true; config.staged_hyper_projections = true;
    config.fused_staged_hyper_mix = true; config.typed_hyper_projection = true;
    config.prefetch_typed_hyper_rows = true; config.prefetch_q8_hyper_output = true;
    config.virtual_q8_gdn_input = true; config.prefetch_q8_gdn_input = true;
    config.virtual_q8_gdn_output = true;
    config.wave8_q8_gdn_input = binding.wave8_q8_gdn_input = true;
    config.wave8_q8_gdn_output = binding.wave8_q8_gdn_output = true;
    config.virtual_q8_lm_head = true;
    const bool a8 = args.profile == "q8-wave8-a8-q51";
    config.q4k_a8g32_gate_up = binding.q4k_a8g32_gate_up = buffers.q4k_a8g32_gate_up = a8;
    config.grouped_exact_q51_down = binding.grouped_exact_q51_down = buffers.grouped_exact_q51_down = a8;
    config.q51_slot_parallel_down = binding.q51_slot_parallel_down = buffers.q51_slot_parallel_down = a8;
    config.batched_moe_activation = binding.batched_moe_activation = a8;
    buffers.activation_element_bytes = 2; buffers.mlp_intermediate_element_bytes = 4;
    return config;
}
std::shared_ptr<Owners> load(const Options& args, const strix::HipLauncherCatalogConfig& config,
                             const strix::ExecutionBindingRequest& binding,
                             const strix::ExecutionBufferRequest& buffers) {
    require(hash_file(args.model) == args.model_sha, "model raw-file SHA differs before mapping");
    auto owner = std::make_shared<Owners>();
    auto architecture = owner->guard->architecture();
    architecture.resize(architecture.find(':') == std::string::npos ? architecture.size() : architecture.find(':'));
    require(architecture == "gfx1151", "qualification requires the observed gfx1151 device");
    owner->manifest = strix::build_model_manifest(strix::gguf::read_file(args.model));
    if (config.q4k_a8g32_gate_up) {
        const auto classification = strix::moe_q4k_a8::classify_runtime_manifest(owner->model, owner->manifest);
        strix::moe_q4k_a8::validate_slot_parallel_manifest(owner->model, classification);
    }
    owner->mapped = std::make_shared<strix::MappedModelFile>(
        strix::MappedModelFile::open_read_only(args.model, "model", owner->generations));
    strix::kernels::KernelRegistry kernels;
    strix::register_complete_hip_kernel_catalog(owner->model, binding, config, kernels);
    owner->serial = strix::compile_serial_execution(owner->model, kernels, binding, buffers, 256);
    const auto index = strix::build_model_tensor_index(owner->manifest);
    owner->access = strix::compile_execution_tensor_access_plan(owner->serial.execution, owner->manifest,
        index, strix::build_serial_execution_tensor_requirements(owner->serial.execution));
    owner->window = strix::compile_known_prefix_window(owner->model, owner->serial,
        owner->manifest, owner->access, {true, true});
    require(owner->window.r2_steps == 84, "real model does not admit all 48 shared and 36 GDN R2 steps");
    const auto projections = strix::q8_projection_organization::build_manifest_plan(
        owner->model, owner->manifest, false, true, true, false);
    strix::SerialSessionStateLayoutRequest state_request;
    state_request.context_tokens = context_tokens;
    owner->layout = strix::build_serial_session_state_layout(owner->model, state_request);
    require(owner->layout.allocations.size() == 7, "qualification requires all seven canonical state components");
    owner->resident = std::make_shared<strix::HipModelResidency>(strix::HipModelResidency::create(
        *owner->mapped, owner->manifest, owner->generations));
    strix::q8_projection_organization::validate_resident_weights(projections, owner->resident->binding());
    owner->allocator = std::make_shared<strix::HipRuntimeAllocator>(strix::HipHostAllocationMode::pageable_aligned);
    owner->memory = std::make_shared<strix::RuntimeExecutionMemoryOwner>(strix::RuntimeExecutionMemoryOwner::allocate(
        owner->serial, owner->layout, *owner->allocator, owner->generations));
    strix::q8_projection_organization::validate_output_steps(projections, owner->model, owner->serial,
        owner->access, owner->memory->scratch(), owner->resident->binding());
    owner->window_memory = {owner->window.window_execution_identity, 0,
        owner->window.arena.physical.total_bytes, owner->generations.next()};
    owner->window_memory.arena_base = owner->allocator->allocate_device(owner->window_memory.arena_bytes, 256);
    owner->guard->run([&] {
        hip_check(hipStreamCreateWithFlags(&owner->stream, hipStreamNonBlocking), "create qualification stream");
        strix::initialize_hip_fresh_runtime_memory(owner->memory->scratch(), owner->memory->state());
        hip_check(hipMemsetAsync(reinterpret_cast<void*>(owner->window_memory.arena_base), 0,
            static_cast<std::size_t>(owner->window_memory.arena_bytes), owner->stream), "initialize window scratch");
    });
    owner->ple_hash = strix::build_ple_hash_contract(owner->model);
    owner->resources = {&owner->memory->scratch(), &owner->manifest, &owner->resident->binding(),
        &owner->access, &owner->layout, &owner->memory->state(), &owner->ple, &owner->ple_hash, &owner->prefetch};
    return owner;
}

std::ofstream output_file(const std::filesystem::path& path) {
    require(!std::filesystem::exists(path), "qualification output already exists");
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    return output;
}
bool equal_files(const std::filesystem::path& left, const std::filesystem::path& right) {
    const auto bytes = std::filesystem::file_size(left);
    if (bytes != std::filesystem::file_size(right)) return false;
    std::ifstream a(left, std::ios::binary), b(right, std::ios::binary);
    a.exceptions(std::ios::failbit | std::ios::badbit);
    b.exceptions(std::ios::failbit | std::ios::badbit);
    std::vector<char> x(transfer_chunk), y(transfer_chunk);
    for (std::uint64_t offset = 0; offset < bytes;) {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(x.size(), bytes - offset));
        a.read(x.data(), static_cast<std::streamsize>(count));
        b.read(y.data(), static_cast<std::streamsize>(count));
        if (std::memcmp(x.data(), y.data(), count) != 0) return false;
        offset += count;
    }
    return true;
}
struct StateFile {
    strix::SessionStateComponent component;
    strix::SessionStateStorage storage;
    std::uint64_t bytes;
    std::filesystem::path path;
    std::string sha;
};
std::vector<StateFile> dump_state(const std::shared_ptr<Owners>& owner,
                                 const std::filesystem::path& directory, std::string_view stem) {
    require(!owner->prefetch.active, "state capture requires an inactive PLE lookup");
    std::vector<StateFile> result;
    std::vector<std::byte> buffer(transfer_chunk);
    for (const auto& allocation : owner->layout.allocations) {
        const auto view = strix::resolve_session_state(owner->layout, owner->memory->state(), allocation.component);
        const auto path = directory / (std::string(stem) + "-state-" +
            std::to_string(static_cast<unsigned>(allocation.component)) + ".bin");
        auto output = output_file(path);
        for (std::uint64_t offset = 0; offset < view.bytes;) {
            const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), view.bytes - offset));
            owner->guard->run([&] {
                if (view.storage == strix::SessionStateStorage::device)
                    hip_check(hipMemcpyAsync(buffer.data(), reinterpret_cast<const void*>(view.address + offset),
                        count, hipMemcpyDeviceToHost, owner->stream), "capture state bytes");
                else std::memcpy(buffer.data(), reinterpret_cast<const void*>(view.address + offset), count);
            });
            output.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(count));
            offset += count;
        }
        output.close();
        result.push_back({allocation.component, view.storage, view.bytes, path, hash_file(path)});
    }
    return result;
}
std::vector<float> logits(const std::shared_ptr<Owners>& owner, strix::ExecutionBufferAddress view,
                          const std::filesystem::path& path) {
    std::vector<float> values(static_cast<std::size_t>(owner->model.vocab_size));
    require(view.bytes >= values.size() * sizeof(float), "logit row extent is too small");
    owner->guard->run([&] { hip_check(hipMemcpyAsync(values.data(), reinterpret_cast<const void*>(view.address),
        values.size() * sizeof(float), hipMemcpyDeviceToHost, owner->stream), "capture full FP32 logits"); });
    require(std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }),
        "model logits contain a nonfinite value");
    auto output = output_file(path);
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return values;
}
void write_session(std::ostream& output, const strix::SerialSessionContext& session) {
    output << "{\"execution_identity\":" << session.execution_identity
        << ",\"runtime_compatibility_identity\":" << session.runtime_compatibility_identity
        << ",\"state_layout_identity\":" << session.state_layout_identity
        << ",\"next_position\":" << session.next_position << ",\"committed_tokens\":" << session.committed_tokens
        << ",\"ready\":" << (session.health == strix::SerialSessionHealth::ready ? "true" : "false")
        << ",\"valid\":" << static_cast<std::uint32_t>(session.state.valid) << ",\"generations\":[";
    for (std::size_t i = 0; i < session.state.generations.size(); ++i)
        output << (i ? "," : "") << session.state.generations[i];
    output << "],\"lineage\":{\"history_identity\":" << session.arithmetic_lineage.history_identity
        << ",\"tainted\":" << (session.arithmetic_lineage.tainted ? "true" : "false") << ",\"intervals\":[";
    for (std::size_t i = 0; i < session.arithmetic_lineage.intervals.size(); ++i) {
        const auto& interval = session.arithmetic_lineage.intervals[i];
        output << (i ? "," : "") << "{\"begin\":" << interval.begin << ",\"end\":" << interval.end
            << ",\"producer\":" << static_cast<unsigned>(interval.producer)
            << ",\"numerical\":" << interval.numerical_plan_identity
            << ",\"reduction\":" << interval.reduction_plan_identity << ",\"state_update\":" << interval.state_update_identity << '}';
    }
    output << "]}}";
}
void write_states(std::ostream& output, const std::vector<StateFile>& files) {
    output << '[';
    for (std::size_t i = 0; i < files.size(); ++i) {
        const auto& file = files[i];
        output << (i ? "," : "") << "{\"component\":" << static_cast<unsigned>(file.component)
            << ",\"storage\":" << static_cast<unsigned>(file.storage) << ",\"bytes\":" << file.bytes
            << ",\"file\":" << quote(file.path.filename().string()) << ",\"sha256\":" << quote(file.sha) << '}';
    }
    output << ']';
}

std::string session_text(const strix::SerialSessionContext& session) {
    std::ostringstream output;
    write_session(output, session);
    return output.str();
}

// Check every live byte against the already SHA-pinned initial snapshot. This
// implies the same seven full-component hashes without writing extra snapshots
// for each rejected constructor. Transfers remain bounded to transfer_chunk.
void require_unchanged(const std::shared_ptr<Owners>& owner, const std::vector<StateFile>& initial,
                       const strix::SerialSessionContext& saved, const strix::SerialSessionContext& session) {
    require(session_text(session) == session_text(saved), "owner rejection changed session metadata");
    require(!owner->prefetch.active, "owner rejection activated a PLE lookup");
    require(initial.size() == 7, "owner rejection requires all seven saved components");
    std::vector<std::byte> expected(transfer_chunk), observed(transfer_chunk);
    for (const auto& file : initial) {
        require(hash_file(file.path) == file.sha && std::filesystem::file_size(file.path) == file.bytes,
            "owner rejection initial snapshot changed");
        const auto view = strix::resolve_session_state(owner->layout, owner->memory->state(), file.component);
        require(view.bytes == file.bytes && view.storage == file.storage, "owner rejection changed state binding");
        std::ifstream input(file.path, std::ios::binary);
        input.exceptions(std::ios::failbit | std::ios::badbit);
        for (std::uint64_t offset = 0; offset < view.bytes;) {
            const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(expected.size(), view.bytes - offset));
            input.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(count));
            owner->guard->run([&] {
                if (view.storage == strix::SessionStateStorage::device)
                    hip_check(hipMemcpyAsync(observed.data(), reinterpret_cast<const void*>(view.address + offset),
                        count, hipMemcpyDeviceToHost, owner->stream), "check rejected owner state bytes");
                else std::memcpy(observed.data(), reinterpret_cast<const void*>(view.address + offset), count);
            });
            require(std::memcmp(expected.data(), observed.data(), count) == 0,
                "owner rejection changed full state bytes");
            offset += count;
        }
    }
}

bool same_lineage(const strix::ArithmeticLineage& a, const strix::ArithmeticLineage& b) {
    return a.intervals == b.intervals && a.history_identity == b.history_identity && a.tainted == b.tainted;
}
void require_oracle(const strix::SerialSessionContext& session, const strix::ArithmeticLineage& oracle) {
    require(session.health == strix::SerialSessionHealth::ready && static_cast<std::uint32_t>(session.state.valid) == 63U &&
        session.next_position == session.committed_tokens && same_lineage(session.arithmetic_lineage, oracle),
        "actual session differs from independent arithmetic producer oracle");
}
void require_position_word(const std::shared_ptr<Owners>& owner, const strix::SerialSessionContext& session) {
    const auto position = strix::resolve_session_state(owner->layout, owner->memory->state(),
        strix::SessionStateComponent::position_history);
    require(position.storage == strix::SessionStateStorage::host && position.bytes == sizeof(std::uint64_t),
        "G4 position history is not the complete host word");
    std::uint64_t word = 0;
    std::memcpy(&word, reinterpret_cast<const void*>(position.address), sizeof(word));
    require(word == (session.next_position ? session.next_position - 1U : 0U),
        "G4 position history differs from consumed frontier");
}
strix::SerialSessionContext fresh_session(const std::shared_ptr<Owners>& owner) {
    strix::SerialSessionContext session;
    session.execution_identity = owner->serial.execution_identity;
    session.state_layout_identity = owner->layout.layout_identity;
    session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
        owner->serial.execution_identity, owner->serial.arena, owner->resources).compatibility_identity;
    for (const auto& allocation : owner->layout.allocations)
        session.state.valid = session.state.valid | allocation.state_flag;
    return session;
}
void reset_fresh(const std::shared_ptr<Owners>& owner, const std::unique_ptr<strix::HipKnownPrefixWindow>& window,
                 strix::SerialSessionContext& session) {
    require(!window && !owner->prefetch.active, "reset requires a destroyed window and inactive PLE");
    owner->guard->run([&] {
        strix::initialize_hip_fresh_runtime_memory(owner->memory->scratch(), owner->memory->state());
        hip_check(hipMemsetAsync(reinterpret_cast<void*>(owner->window_memory.arena_base), 0,
            static_cast<std::size_t>(owner->window_memory.arena_bytes), owner->stream), "reset G4 window scratch");
    });
    session = fresh_session(owner);
}
struct RowRecord {
    std::uint64_t position;
    std::uint32_t token;
    std::string source;
    std::filesystem::path file;
    std::string sha;
};
struct WindowRecord {
    std::uint64_t position, r2_dispatches, row_loop_dispatches;
};
struct Branch {
    std::string name;
    strix::ArithmeticLineage oracle;
    std::vector<RowRecord> rows;
    std::vector<std::string> calls;
    std::vector<WindowRecord> windows;
    std::optional<strix::SerialGreedyRequestResult> controller;
    std::uint64_t serial_prompt_steps = 0;
    explicit Branch(std::string value) : name(std::move(value)) {}
};
std::vector<float> record_row(const std::shared_ptr<Owners>& owner, Branch& branch,
    const std::filesystem::path& directory, strix::ExecutionBufferAddress view,
    std::uint64_t position, std::uint32_t token, const char* source) {
    const auto file = directory / (branch.name + "-row-" + std::to_string(position) + ".f32");
    auto values = logits(owner, view, file);
    branch.rows.push_back({position, token, source, file, hash_file(file)});
    return values;
}
class QualSerialBackend final : public strix::SerialGreedyStepBackend {
public:
    QualSerialBackend(std::shared_ptr<Owners> value, strix::SerialSessionContext& current,
        strix::DirectLaunchDispatcher& launches, Branch& records, const std::filesystem::path& directory)
        : owner(std::move(value)), session(current), dispatcher(launches), branch(records), output(directory) {}
    std::vector<float> execute(std::uint32_t token, std::uint64_t position) override {
        require(position == session.next_position, "serial backend position differs from consumed frontier");
        const auto next_oracle = strix::prepare_serial_row_lineage(branch.oracle, position,
            owner->serial.execution_identity, owner->serial.kernel_plan_identity);
        owner->guard->run([&] {
            const auto context = owner->scope();
            const strix::HipLaunchScope scope(context);
            (void)strix::execute_serial_token(owner->serial, session, owner->resources, token, dispatcher,
                [&] { hip_check(hipStreamSynchronize(owner->stream), "complete G4 serial row"); });
        });
        require(!owner->prefetch.active, "G4 serial row left an active PLE lookup");
        branch.oracle = next_oracle;
        require_oracle(session, branch.oracle);
        branch.calls.push_back("serial:" + std::to_string(position) + ":" + std::to_string(token));
        return record_row(owner, branch, output, strix::resolve_execution_buffer(owner->serial.execution_identity,
            owner->serial.arena, owner->memory->scratch(), owner->serial.io.steps.back().writes.front()),
            position, token, "serial");
    }
private:
    std::shared_ptr<Owners> owner;
    strix::SerialSessionContext& session;
    strix::DirectLaunchDispatcher& dispatcher;
    Branch& branch;
    std::filesystem::path output;
};
class QualWindowBackend final : public strix::KnownPromptWindowBackend {
public:
    QualWindowBackend(std::shared_ptr<Owners> value, strix::SerialSessionContext& current,
        std::unique_ptr<strix::HipKnownPrefixWindow>& instance, Branch& records,
        const std::filesystem::path& directory)
        : owner(std::move(value)), session(current), window(instance), branch(records), output(directory) {}
    std::vector<float> execute_window(std::array<std::uint32_t, 2> tokens, std::uint64_t position) override {
        require(window && position == session.next_position, "window backend owner/position mismatch");
        const auto next_oracle = strix::prepare_prefill_chunk_lineage(branch.oracle, position, position + 2U,
            owner->window.window_execution_identity, owner->window.window_kernel_plan_identity,
            owner->window.window_execution_identity);
        const auto result = window->execute(session, tokens);
        require(owner->window.r2_steps == 84 && result.r2_dispatches == 84 && result.row_loop_dispatches == 708,
            "G4 window did not execute the closed 84/708 dispatches");
        require(!owner->prefetch.active, "G4 window left an active PLE lookup");
        branch.windows.push_back({position, result.r2_dispatches, result.row_loop_dispatches});
        branch.calls.push_back("window:" + std::to_string(position) + ":" +
            std::to_string(tokens[0]) + ":" + std::to_string(tokens[1]));
        (void)record_row(owner, branch, output, result.logits[0], position, tokens[0], "window-row0");
        auto terminal = record_row(owner, branch, output, result.logits[1], position + 1U, tokens[1], "window-row1");
        branch.oracle = next_oracle;
        require_oracle(session, branch.oracle);
        return terminal;
    }
private:
    std::shared_ptr<Owners> owner;
    strix::SerialSessionContext& session;
    std::unique_ptr<strix::HipKnownPrefixWindow>& window;
    Branch& branch;
    std::filesystem::path output;
};
class CountingSnapshotBackend final : public strix::SessionSnapshotTransferBackend {
public:
    std::uint64_t captures = 0, restores = 0, throw_restore = 0;
    void clear() { captures = 0; restores = 0; throw_restore = 0; }
    void capture_device(std::uintptr_t a, std::uintptr_t b, std::uint64_t n) override {
        ++captures; actual.capture_device(a,b,n);
    }
    void capture_host(std::uintptr_t a, std::uintptr_t b, std::uint64_t n) override {
        ++captures; actual.capture_host(a,b,n);
    }
    void restore_device(std::uintptr_t a, std::uintptr_t b, std::uint64_t n) override {
        before_restore(); actual.restore_device(a,b,n);
    }
    void restore_host(std::uintptr_t a, std::uintptr_t b, std::uint64_t n) override {
        before_restore(); actual.restore_host(a,b,n);
    }
private:
    strix::HipSessionSnapshotTransferBackend actual;
    void before_restore() {
        ++restores;
        if (throw_restore && restores == throw_restore)
            throw std::runtime_error("G4 deliberate restore failure after one real copy");
    }
};
void require_same_state_files(const std::vector<StateFile>& left, const std::vector<StateFile>& right) {
    require(left.size() == 7 && right.size() == 7, "G4 comparison requires seven complete state components");
    for (std::size_t i = 0; i < left.size(); ++i)
        require(left[i].component == right[i].component && left[i].storage == right[i].storage &&
            left[i].bytes == right[i].bytes && left[i].sha == right[i].sha && equal_files(left[i].path, right[i].path),
            "G4 full final state components differ");
}
void require_same_metadata(const strix::SerialSessionContext& a, const strix::SerialSessionContext& b) {
    // Producer histories intentionally differ; each has its own independent oracle.
    require(a.execution_identity == b.execution_identity && a.runtime_compatibility_identity == b.runtime_compatibility_identity &&
        a.state_layout_identity == b.state_layout_identity && a.next_position == b.next_position &&
        a.committed_tokens == b.committed_tokens && a.health == b.health && a.state.valid == b.state.valid &&
        a.state.generations == b.state.generations, "G4 final session bindings/frontier/generations differ");
}
void require_rows(const Branch& reference, const Branch& actual, std::uint64_t begin) {
    require(reference.controller && actual.controller && actual.rows.size() == reference.rows.size() - begin,
        "G4 actual row coverage differs");
    for (std::size_t i = 0; i < actual.rows.size(); ++i) {
        const auto& a = reference.rows[i + static_cast<std::size_t>(begin)];
        const auto& b = actual.rows[i];
        require(a.position == b.position && b.position == begin + i && a.token == b.token &&
            a.sha == b.sha && equal_files(a.file, b.file), "G4 full FP32 row differs");
    }
}
void require_controller(const strix::SerialGreedyRequestResult& reference,
                        const strix::SerialGreedyRequestResult& actual, std::uint64_t prompt_size) {
    require(reference.generated_tokens == actual.generated_tokens && reference.stop_reason == actual.stop_reason &&
        reference.final_position == actual.final_position && reference.steps_executed >= prompt_size &&
        reference.steps_executed - prompt_size == actual.steps_executed, "G4 continuation controller differs");
}
void write_branch(std::ostream& out, const Branch& branch, const strix::SerialSessionContext& session) {
    out << "{\"name\":" << quote(branch.name) << ",\"calls\":[";
    for (std::size_t i = 0; i < branch.calls.size(); ++i) out << (i ? "," : "") << quote(branch.calls[i]);
    out << "],\"rows\":[";
    for (std::size_t i = 0; i < branch.rows.size(); ++i) {
        const auto& r = branch.rows[i];
        out << (i ? "," : "") << "{\"position\":" << r.position << ",\"input_token\":" << r.token
            << ",\"source\":" << quote(r.source) << ",\"file\":" << quote(r.file.filename().string())
            << ",\"bytes\":" << 248320U * sizeof(float) << ",\"sha256\":" << quote(r.sha) << '}';
    }
    out << "],\"windows\":[";
    for (std::size_t i = 0; i < branch.windows.size(); ++i) {
        const auto& w = branch.windows[i];
        out << (i ? "," : "") << "{\"position\":" << w.position << ",\"r2_dispatches\":" << w.r2_dispatches
            << ",\"row_loop_dispatches\":" << w.row_loop_dispatches << '}';
    }
    require(branch.controller.has_value(), "G4 final branch controller missing");
    const auto& result = *branch.controller;
    out << "],\"serial_prompt_steps\":" << branch.serial_prompt_steps << ",\"controller\":{\"tokens\":[";
    for (std::size_t i = 0; i < result.generated_tokens.size(); ++i) out << (i ? "," : "") << result.generated_tokens[i];
    out << "],\"steps_executed\":" << result.steps_executed << ",\"final_position\":" << result.final_position
        << ",\"stop_reason\":" << static_cast<unsigned>(result.stop_reason) << "},\"session\":";
    write_session(out, session);
    out << '}';
}

struct CacheLifetime {
    std::shared_ptr<Owners> owner;
    std::optional<strix::SerialPrefixCacheEntry> entry;
    explicit CacheLifetime(std::shared_ptr<Owners> value) : owner(std::move(value)) {}
    ~CacheLifetime() {
        try { owner->guard->run([] {}); }
        catch (...) { std::_Exit(3); }
        // entry frees its retained snapshot after this explicit device drain.
    }
};
struct NegativeRecord { std::string name; std::uint64_t transfers; bool unchanged; };

int qualify(const Options& args) {
    const auto prompt = teacher_tokens(args, strix::ModelContract::qwen38_flash_next().vocab_size);
    const auto executable = std::filesystem::canonical("/proc/self/exe");
    const auto executable_sha = hash_file(executable);
    require(std::filesystem::create_directory(args.output), "cannot create fresh G4 output directory");
    strix::ExecutionBindingRequest binding;
    strix::ExecutionBufferRequest buffers;
    const auto config = configure(args, binding, buffers);
    const auto owner = load(args, config, binding, buffers);
    CacheLifetime cached(owner); // destroyed before model/state allocator
    std::unique_ptr<strix::HipKnownPrefixWindow> window; // destroyed before cache on every exit
    auto catalog = std::make_shared<strix::HipLauncherCatalog>(owner->model, config);
    strix::ExecutionLauncherRegistry registry;
    catalog->register_for_compiled_execution(owner->serial, registry, &owner->resources);
    const auto launches = strix::compile_launch_plan(owner->serial, registry);
    strix::DirectLaunchDispatcher dispatcher(launches);
    auto session = fresh_session(owner);
    strix::SerialSessionSnapshotRequest snapshot_request;
    snapshot_request.compiled_execution_identity = session.execution_identity;
    snapshot_request.runtime_compatibility_identity = session.runtime_compatibility_identity;
    snapshot_request.state = session.state;
    const auto capacity_plan = strix::build_serial_session_snapshot_plan(owner->model, owner->layout, snapshot_request);
    const auto free_disk = std::filesystem::space(args.output).available;
    const auto required_disk = 4U * (owner->layout.device_total_bytes + owner->layout.host_total_bytes) +
        3U * 8U * owner->model.vocab_size * sizeof(float) + 64U * 1024U * 1024U;
    std::size_t free_device = 0, total_device = 0;
    owner->guard->run([&] { hip_check(hipMemGetInfo(&free_device, &total_device), "admit G4 snapshot capacity"); });
    require(free_disk >= required_disk && free_device >= capacity_plan.device_bytes + 64U * 1024U * 1024U,
        "G4 snapshot/dump capacity unavailable");

    Branch b1{"B1"}, b2{"B2"}, b3{"B3"};
    reset_fresh(owner, window, session);
    QualSerialBackend serial1(owner, session, dispatcher, b1, args.output);
    strix::SerialGreedyRequest request1;
    request1.prefix_tokens = prompt; request1.vocab_size = owner->model.vocab_size;
    request1.context_capacity = context_tokens; request1.max_new_tokens = 4;
    request1.eos_token_id = static_cast<std::uint32_t>(owner->model.ple_eos_token_id);
    b1.controller = strix::execute_serial_greedy_request(request1, serial1);
    b1.serial_prompt_steps = static_cast<std::uint64_t>(prompt.size());
    require_oracle(session, b1.oracle);
    require_position_word(owner, session);
    const auto b1_session = session;
    const auto b1_state = dump_state(owner, args.output, "B1-final");
    require(b1.controller->steps_executed == b1.rows.size() && session.next_position == b1.rows.size() &&
        b1.controller->final_position == session.next_position && !b1.controller->generated_tokens.empty() &&
        b1.controller->generated_tokens.size() <= 4U &&
        b1.controller->steps_executed == prompt.size() + b1.controller->generated_tokens.size() - 1U,
        "B1 full-row/controller coverage differs");
    const auto expected_stop = b1.controller->generated_tokens.back() == request1.eos_token_id
        ? strix::SerialRequestStopReason::eos : strix::SerialRequestStopReason::max_new_tokens;
    require(b1.controller->stop_reason == expected_stop &&
        (expected_stop == strix::SerialRequestStopReason::eos || b1.controller->generated_tokens.size() == 4U),
        "G4 controller lost EOS/budget priority or consumed its last emitted token");

    reset_fresh(owner, window, session);
    window = std::make_unique<strix::HipKnownPrefixWindow>(owner->window, owner->serial, owner->model, config,
        owner->resources, owner->window_memory, owner->scope(), std::static_pointer_cast<const void>(owner));
    QualSerialBackend serial2(owner, session, dispatcher, b2, args.output);
    QualWindowBackend windows2(owner, session, window, b2, args.output);
    strix::KnownPromptGreedyRequest request2;
    request2.prompt_tokens = prompt; request2.vocab_size = owner->model.vocab_size;
    request2.context_capacity = context_tokens; request2.max_new_tokens = 4;
    request2.eos_token_id = request1.eos_token_id;
    std::uint64_t predicate_calls = 0;
    request2.stop_between_windows = [&](std::uint64_t next) { ++predicate_calls; return next == 2U; };
    const auto stopped = strix::execute_known_prompt_greedy_request(request2, windows2, serial2);
    require(stopped.windows_executed == 1 && stopped.consumed_prompt_tokens == 2 && !stopped.generation &&
        stopped.serial_prompt_steps == 0 && predicate_calls == 1 && b2.calls.size() == 1 &&
        !owner->prefetch.active && session.next_position == 2 && session.committed_tokens == 2,
        "G4 did not stop exactly between two completed windows");
    const auto cut_oracle = strix::prepare_prefill_chunk_lineage({}, 0, 2,
        owner->window.window_execution_identity, owner->window.window_kernel_plan_identity,
        owner->window.window_execution_identity);
    require_oracle(session, cut_oracle);
    require_position_word(owner, session);
    const auto cut_session = session;
    const auto cut_state = dump_state(owner, args.output, "cut2");
    CountingSnapshotBackend transfer;
    const auto first_two = std::span<const std::uint32_t>(prompt).first(2);
    require(!owner->prefetch.active, "capture requires inactive PLE");
    const auto before_capture = session_text(session);
    owner->guard->run([&] {
        cached.entry.emplace(strix::SerialPrefixCacheEntry::capture(owner->model, first_two, session,
            owner->layout, owner->memory->state(), *owner->allocator, owner->generations, transfer));
    });
    require(cached.entry->valid() && transfer.captures == cached.entry->prefix().snapshot.ranges.size() &&
        session_text(session) == before_capture && same_lineage(cached.entry->prefix().arithmetic_lineage, cut_oracle),
        "G4 real cache capture changed session or missed ranges/history");
    const auto capture_calls = transfer.captures;
    const auto snapshot = cached.entry->prefix().snapshot;
    request2.prompt_tokens = std::span<const std::uint32_t>(prompt).subspan(2);
    request2.start_position = 2; request2.stop_between_windows = {};
    const auto resumed2 = strix::execute_known_prompt_greedy_request(request2, windows2, serial2);
    require(resumed2.generation && resumed2.consumed_prompt_tokens == prompt.size() - 2U &&
        resumed2.windows_executed == 1 && resumed2.serial_prompt_steps == prompt.size() % 2U,
        "G4 same-owner suffix resume coverage differs");
    b2.controller = *resumed2.generation; b2.serial_prompt_steps = resumed2.serial_prompt_steps;
    require(b2.windows.size() == 2, "G4 cancellation/resume lost or repeated a window");
    require_oracle(session, b2.oracle);
    require_position_word(owner, session);
    window.reset();
    const auto b2_session = session;
    const auto b2_state = dump_state(owner, args.output, "B2-final");
    require_controller(*b1.controller, *b2.controller, prompt.size());
    require_rows(b1, b2, 0); require_same_state_files(b1_state, b2_state); require_same_metadata(b1_session, b2_session);
    require(!same_lineage(b1_session.arithmetic_lineage, b2_session.arithmetic_lineage),
        "G4 oracle witness cannot distinguish serial from chunk producers");

    // Missing G3 recovery leg only: no constructor-mutation matrix is repeated.
    window = std::make_unique<strix::HipKnownPrefixWindow>(owner->window, owner->serial, owner->model, config,
        owner->resources, owner->window_memory, owner->scope(), std::static_pointer_cast<const void>(owner));
    bool rejected = false;
    try { (void)window->execute(session, {static_cast<std::uint32_t>(owner->model.vocab_size), prompt[1]}); }
    catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()).find("outside the compiled vocabulary") != std::string_view::npos,
            "G4 failed-owner admission rejected for wrong reason"); rejected = true;
    }
    require(rejected, "G4 invalid owner execution was accepted");
    rejected = false;
    try { (void)window->execute(session, {prompt[0], prompt[1]}); }
    catch (const std::logic_error& error) {
        require(std::string_view(error.what()).find("cannot be reused") != std::string_view::npos,
            "G4 failed owner retry rejected for wrong reason"); rejected = true;
    }
    require(rejected, "G4 failed owner was reused");
    window.reset();

    std::vector<NegativeRecord> negatives;
    const auto miss = [&](std::string name, std::span<const std::uint32_t> tokens,
        const strix::ArithmeticLineage* history, strix::SerialSessionContext destination) {
        require(!window && !owner->prefetch.active, "cache miss requires destroyed owner and inactive PLE");
        transfer.clear(); const auto before = session_text(destination); bool hit = true;
        owner->guard->run([&] { hit = cached.entry->try_restore(destination, owner->model, owner->layout,
            owner->memory->state(), transfer, tokens, history); });
        require(!hit && transfer.restores == 0 && transfer.captures == 0 && session_text(destination) == before,
            "G4 expected cache miss transferred or changed metadata");
        negatives.push_back({std::move(name), transfer.restores + transfer.captures, true});
    };
    miss("null-history", first_two, nullptr, session);
    auto serial_history = strix::prepare_serial_row_lineage({}, 0, owner->serial.execution_identity,
        owner->serial.kernel_plan_identity);
    serial_history = strix::prepare_serial_row_lineage(serial_history, 1, owner->serial.execution_identity,
        owner->serial.kernel_plan_identity);
    miss("serial-history", first_two, &serial_history, session);
    auto cuts = strix::prepare_prefill_chunk_lineage({}, 0, 1, owner->window.window_execution_identity,
        owner->window.window_kernel_plan_identity, owner->window.window_execution_identity);
    cuts = strix::prepare_prefill_chunk_lineage(cuts, 1, 2, owner->window.window_execution_identity,
        owner->window.window_kernel_plan_identity, owner->window.window_execution_identity);
    miss("different-chunk-cuts", first_two, &cuts, session);
    for (unsigned field = 0; field < 3U; ++field) {
        auto changed = cut_oracle;
        if (field == 0U) changed.intervals[0].numerical_plan_identity ^= 1ULL;
        if (field == 1U) changed.intervals[0].reduction_plan_identity ^= 1ULL;
        if (field == 2U) changed.intervals[0].state_update_identity ^= 1ULL;
        changed.history_identity = strix::arithmetic_history_identity(changed.intervals);
        miss("history-identity-field-" + std::to_string(field), first_two, &changed, session);
    }
    auto tainted = cut_oracle; tainted.tainted = true;
    miss("tainted-history", first_two, &tainted, session);
    const auto gap = strix::prepare_prefill_chunk_lineage({}, 0, 1, owner->window.window_execution_identity,
        owner->window.window_kernel_plan_identity, owner->window.window_execution_identity);
    miss("gap-history", first_two, &gap, session);
    std::array<std::uint32_t, 2> changed_tokens{prompt[0],
        static_cast<std::uint32_t>((static_cast<std::uint64_t>(prompt[1]) + 1U) % owner->model.vocab_size)};
    miss("changed-token", changed_tokens, &cut_oracle, session);
    miss("prefix-length-three", std::span<const std::uint32_t>(prompt).first(3), &cut_oracle, session);
    for (unsigned field = 0; field < 3U; ++field) {
        auto destination = session;
        if (field == 0U) destination.runtime_compatibility_identity ^= 1ULL;
        if (field == 1U) destination.state_layout_identity ^= 1ULL;
        if (field == 2U) destination.execution_identity = owner->window.window_execution_identity;
        miss("destination-identity-field-" + std::to_string(field), first_two, &cut_oracle, destination);
    }
    transfer.clear(); rejected = false;
    require(!owner->prefetch.active, "throwing miss requires inactive PLE");
    try { owner->guard->run([&] { cached.entry->restore(session, owner->model, owner->layout,
        owner->memory->state(), transfer, first_two, nullptr); }); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && transfer.restores == 0 && transfer.captures == 0, "throwing cache miss performed transfers");
    negatives.push_back({"restore-null-history", 0, true});
    require(negatives.size() == 14, "G4 cache negative matrix is incomplete");
    require_unchanged(owner, b2_state, b2_session, session);

    const auto restore_hit = [&] {
        require(!window && !owner->prefetch.active, "cache hit requires destroyed owner and inactive PLE");
        transfer.clear(); bool hit = false;
        owner->guard->run([&] { hit = cached.entry->try_restore(session, owner->model, owner->layout,
            owner->memory->state(), transfer, first_two, &cut_oracle); });
        require(hit && transfer.restores == snapshot.ranges.size() && transfer.captures == 0,
            "G4 real cache hit missed or did not restore all ranges");
        require_unchanged(owner, cut_state, cut_session, session);
    };
    restore_hit();
    const auto first_hit_calls = transfer.restores;
    b3.oracle = cut_oracle; // independently constructed cut, never copied from live session
    window = std::make_unique<strix::HipKnownPrefixWindow>(owner->window, owner->serial, owner->model, config,
        owner->resources, owner->window_memory, owner->scope(), std::static_pointer_cast<const void>(owner));
    QualSerialBackend serial3(owner, session, dispatcher, b3, args.output);
    QualWindowBackend windows3(owner, session, window, b3, args.output);
    const auto resumed3 = strix::execute_known_prompt_greedy_request(request2, windows3, serial3);
    require(resumed3.generation && resumed3.consumed_prompt_tokens == prompt.size() - 2U &&
        resumed3.windows_executed == 1 && resumed3.serial_prompt_steps == prompt.size() % 2U,
        "G4 fresh-owner cached suffix resume coverage differs");
    b3.controller = *resumed3.generation; b3.serial_prompt_steps = resumed3.serial_prompt_steps;
    require_oracle(session, b3.oracle);
    require_position_word(owner, session);
    window.reset();
    const auto b3_session = session;
    const auto b3_state = dump_state(owner, args.output, "B3-final");
    require_controller(*b1.controller, *b3.controller, prompt.size());
    require_rows(b1, b3, 2); require_same_state_files(b1_state, b3_state); require_same_metadata(b1_session, b3_session);

    // Real D2D copy number one succeeds; the second is rejected by our wrapper.
    require(!window && !owner->prefetch.active, "partial restore requires destroyed owner and inactive PLE");
    transfer.clear(); transfer.throw_restore = 2; rejected = false;
    try { owner->guard->run([&] { (void)cached.entry->try_restore(session, owner->model, owner->layout,
        owner->memory->state(), transfer, first_two, &cut_oracle); }); }
    catch (const std::runtime_error& error) {
        require(std::string_view(error.what()) == "G4 deliberate restore failure after one real copy",
            "partial restore failed for an unexpected reason"); rejected = true;
    }
    require(rejected && transfer.restores == 2 && session.health == strix::SerialSessionHealth::poisoned &&
        session.arithmetic_lineage.tainted && session.arithmetic_lineage.intervals.empty() &&
        session.arithmetic_lineage.history_identity == 0 && session.next_position == b3_session.next_position &&
        session.committed_tokens == b3_session.committed_tokens && session.state.valid == b3_session.state.valid &&
        session.state.generations == b3_session.state.generations,
        "partial restore did not preserve transaction metadata and poison lineage");
    const auto poisoned_text = session_text(session);
    rejected = false;
    try { (void)serial3.execute(prompt[0], session.next_position); }
    catch (const std::logic_error& error) {
        require(std::string_view(error.what()).find("poisoned") != std::string_view::npos,
            "poisoned serial execution rejected for wrong reason"); rejected = true;
    }
    require(rejected && session_text(session) == poisoned_text, "poisoned serial execution modified metadata");
    transfer.clear(); rejected = false;
    require(!owner->prefetch.active, "poisoned capture requires inactive PLE");
    try { owner->guard->run([&] { auto forbidden = strix::SerialPrefixCacheEntry::capture(owner->model,
        first_two, session, owner->layout, owner->memory->state(), *owner->allocator, owner->generations, transfer);
        (void)forbidden; }); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && transfer.captures == 0 && session_text(session) == poisoned_text,
        "poisoned cache capture was accepted or changed state metadata");
    restore_hit();
    const auto recovery_hit_calls = transfer.restores;
    owner->guard->run([] {});
    require(hash_file(args.teacher) == args.teacher_sha && hash_file(args.model) == args.model_sha &&
        hash_file(executable) == executable_sha, "G4 prompt/model/executable changed during qualification");
    // Re-read all four stored dumps and every declared full-logit file before
    // publishing. Live state is now cut2 after the final explicit cache restore.
    for (const auto* state : {&b1_state, &cut_state, &b2_state, &b3_state})
        for (const auto& file : *state)
            require(std::filesystem::file_size(file.path) == file.bytes && hash_file(file.path) == file.sha,
                "G4 state artifact changed before publication");
    for (const auto* branch : {&b1, &b2, &b3})
        for (const auto& row : branch->rows)
            require(std::filesystem::file_size(row.file) == owner->model.vocab_size * sizeof(float) && hash_file(row.file) == row.sha,
                "G4 full-row artifact changed before publication");
    owner->guard->run([] {});
    cached.entry.reset(); // A9: free snapshot only after drain, before successful publication
    const auto pending = args.output / "result.pending.json";
    auto output = output_file(pending);
    output << "{\"schema\":\"known-prompt-g4-qualification-v1\",\"finished\":true,\"passed\":true"
        << ",\"profile\":" << quote(args.profile) << ",\"model_sha256\":" << quote(args.model_sha)
        << ",\"model_content_identity\":" << quote(hex(owner->mapped->content_identity()))
        << ",\"prompt_sha256\":" << quote(args.teacher_sha) << ",\"executable_sha256\":" << quote(executable_sha)
        << ",\"context_tokens\":" << context_tokens << ",\"prompt_tokens\":[";
    for (std::size_t i = 0; i < prompt.size(); ++i) output << (i ? "," : "") << prompt[i];
    output << "],\"serial_execution_identity\":" << owner->serial.execution_identity
        << ",\"serial_kernel_plan_identity\":" << owner->serial.kernel_plan_identity
        << ",\"window_execution_identity\":" << owner->window.window_execution_identity
        << ",\"window_kernel_plan_identity\":" << owner->window.window_kernel_plan_identity
        << ",\"manifest_layout_identity\":" << owner->manifest.layout_identity
        << ",\"tensor_access_plan_identity\":" << owner->access.plan_identity
        << ",\"source_counts\":{\"r2_steps\":84,\"per_window_r2_dispatches\":84,\"per_window_row_loop_dispatches\":708}"
        << ",\"capacity\":{\"disk_available\":" << free_disk << ",\"disk_required\":" << required_disk
        << ",\"device_free\":" << free_device << ",\"device_total\":" << total_device
        << ",\"snapshot_device_required\":" << capacity_plan.device_bytes << "},\"snapshot\":{\"identity\":"
        << snapshot.plan_identity << ",\"device_bytes\":" << snapshot.device_bytes << ",\"host_bytes\":" << snapshot.host_bytes
        << ",\"ranges\":[";
    for (std::size_t i = 0; i < snapshot.ranges.size(); ++i) {
        const auto& range = snapshot.ranges[i];
        output << (i ? "," : "") << "{\"component\":" << static_cast<unsigned>(range.component)
            << ",\"storage\":" << static_cast<unsigned>(range.storage) << ",\"source_offset\":" << range.source_offset
            << ",\"destination_offset\":" << range.destination_offset << ",\"bytes\":" << range.bytes
            << ",\"provisional\":" << (range.provisional ? "true" : "false") << '}';
    }
    output << "],\"capture_calls\":" << capture_calls << ",\"first_restore_calls\":" << first_hit_calls
        << ",\"partial_restore_calls\":2,\"recovery_restore_calls\":" << recovery_hit_calls << "},\"branches\":[";
    write_branch(output, b1, b1_session); output << ','; write_branch(output, b2, b2_session);
    output << ','; write_branch(output, b3, b3_session);
    output << "],\"B1_final_state\":"; write_states(output, b1_state);
    output << ",\"cut2_state\":"; write_states(output, cut_state);
    output << ",\"cut2_session\":"; write_session(output, cut_session);
    output << ",\"B2_final_state\":"; write_states(output, b2_state);
    output << ",\"B3_final_state\":"; write_states(output, b3_state);
    output << ",\"cache_negatives\":[";
    for (std::size_t i = 0; i < negatives.size(); ++i)
        output << (i ? "," : "") << "{\"name\":" << quote(negatives[i].name) << ",\"transfers\":"
            << negatives[i].transfers << ",\"unchanged\":" << (negatives[i].unchanged ? "true" : "false") << '}';
    output << "],\"cancel_predicate_calls\":" << predicate_calls
        << ",\"stopped_consumed_tokens\":" << stopped.consumed_prompt_tokens
        << ",\"stopped_windows\":" << stopped.windows_executed
        << ",\"g4_prompt_driver_cancel\":true,\"g4_cache_resume\":true,\"g4_cache_negative_matrix\":true"
        << ",\"g4_owner_admission_recovery\":true,\"g4_partial_restore_recovery\":true"
        << ",\"native_post_dispatch_fault_injection\":\"not_covered\",\"full_heads_foundation_only\":true"
        << ",\"B3_row_coverage_begin\":2,\"quality\":false,\"performance\":false}\n";
    output.close();
    require(!std::filesystem::exists(args.output / "result.json"), "G4 result destination exists");
    std::filesystem::rename(pending, args.output / "result.json");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return qualify(options(argc, argv)); }
    catch (const std::exception& error) {
        std::cerr << "known-prompt G4 qualification failed: " << error.what() << '\n';
        return 1;
    }
}

