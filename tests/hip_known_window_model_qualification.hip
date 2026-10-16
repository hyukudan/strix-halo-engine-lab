// Explicit model qualification executable. Not an automatic CTest, benchmark,
// greedy predictor, or authorization to reuse unknown future tokens.
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
    std::uint64_t start = 0;
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
        else if (key == "--teacher-u32le") result.teacher = value;
        else if (key == "--teacher-sha256") result.teacher_sha = value;
        else if (key == "--output") result.output = value;
        else if (key == "--profile") result.profile = value;
        else if (key == "--start-position") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result.start);
            require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(), "invalid start position");
        } else throw std::invalid_argument("unknown qualification option: " + key);
    }
    require(!result.model.empty() && !result.teacher.empty() && !result.output.empty(),
        "required: --model --model-sha256 --teacher-u32le --teacher-sha256 --output");
    require(digest_text(result.model_sha) && digest_text(result.teacher_sha), "expected lowercase SHA256 pins required");
    require(result.profile == "q8-wave8" || result.profile == "q8-wave8-a8-q51", "unsupported closed profile");
    require(result.start <= context_tokens - 2U, "two known rows exceed the fixed 8192-token context");
    require(!std::filesystem::exists(result.output), "output directory must be fresh");
    return result;
}
std::vector<std::uint32_t> teacher_tokens(const Options& args, std::uint64_t vocabulary) {
    require(hash_file(args.teacher) == args.teacher_sha, "teacher SHA differs before execution");
    const auto bytes = std::filesystem::file_size(args.teacher);
    require(bytes % 4U == 0 && bytes >= (args.start + 2U) * 4U && bytes <= context_tokens * 4U,
        "teacher file must contain bounded explicit u32le tokens including both requested rows");
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
void restore_state(const std::shared_ptr<Owners>& owner, const std::vector<StateFile>& initial,
                   const strix::SerialSessionContext& saved, strix::SerialSessionContext& session) {
    require(!owner->prefetch.active && saved.health == strix::SerialSessionHealth::ready,
        "restore requires completed work and the exact healthy saved metadata");
    std::vector<std::byte> buffer(transfer_chunk);
    for (const auto& file : initial) {
        require(hash_file(file.path) == file.sha && std::filesystem::file_size(file.path) == file.bytes,
            "initial snapshot file changed");
        const auto view = strix::resolve_session_state(owner->layout, owner->memory->state(), file.component);
        require(view.bytes == file.bytes && view.storage == file.storage, "snapshot state binding changed");
        std::ifstream input(file.path, std::ios::binary);
        input.exceptions(std::ios::failbit | std::ios::badbit);
        for (std::uint64_t offset = 0; offset < view.bytes;) {
            const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), view.bytes - offset));
            input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(count));
            owner->guard->run([&] {
                if (view.storage == strix::SessionStateStorage::device)
                    hip_check(hipMemcpyAsync(reinterpret_cast<void*>(view.address + offset), buffer.data(),
                        count, hipMemcpyHostToDevice, owner->stream), "restore state bytes");
                else std::memcpy(reinterpret_cast<void*>(view.address + offset), buffer.data(), count);
            });
            offset += count;
        }
    }
    // This is an in-process diagnostic restore of the saved lineage itself,
    // not the snapshot API's intentionally lineage-less P1 restore protocol.
    session = saved;
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

int qualify(const Options& args) {
    const auto teacher = teacher_tokens(args, strix::ModelContract::qwen38_flash_next().vocab_size);
    const auto executable = std::filesystem::canonical("/proc/self/exe");
    const auto executable_sha = hash_file(executable);
    require(std::filesystem::create_directory(args.output), "cannot create fresh output directory");
    strix::ExecutionBindingRequest binding;
    strix::ExecutionBufferRequest buffers;
    const auto config = configure(args, binding, buffers);
    const auto owner = load(args, config, binding, buffers);
    auto catalog = std::make_shared<strix::HipLauncherCatalog>(owner->model, config);
    strix::ExecutionLauncherRegistry registry;
    catalog->register_for_compiled_execution(owner->serial, registry, &owner->resources);
    const auto launch = strix::compile_launch_plan(owner->serial, registry);
    strix::DirectLaunchDispatcher dispatcher(launch);
    strix::SerialSessionContext session;
    session.execution_identity = owner->serial.execution_identity;
    session.state_layout_identity = owner->layout.layout_identity;
    session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
        owner->serial.execution_identity, owner->serial.arena, owner->resources).compatibility_identity;
    for (const auto& allocation : owner->layout.allocations)
        session.state.valid = session.state.valid | allocation.state_flag;
    const auto serial_step = [&](std::uint32_t token) {
        owner->guard->run([&] {
            const auto context = owner->scope();
            strix::HipLaunchScope scope(context);
            (void)strix::execute_serial_token(owner->serial, session, owner->resources, token, dispatcher,
                [&] { hip_check(hipStreamSynchronize(owner->stream), "complete serial row before publication"); });
        });
        require(!owner->prefetch.active, "serial row left an active PLE lookup");
    };
    for (std::uint64_t p = 0; p < args.start; ++p) serial_step(teacher[static_cast<std::size_t>(p)]);
    const auto prefix = session;
    const auto initial = dump_state(owner, args.output, "initial");
    auto metadata = output_file(args.output / "initial-session.json");
    write_session(metadata, prefix); metadata.close();
    std::array<std::vector<float>, 2> reference;
    std::array<std::string, 2> reference_sha, candidate_sha;
    const auto logit_use = owner->serial.io.steps.back().writes.front();
    for (std::size_t r = 0; r < 2; ++r) {
        serial_step(teacher[static_cast<std::size_t>(args.start) + r]);
        const auto path = args.output / ("serial-logits-" + std::to_string(r) + ".f32");
        reference[r] = logits(owner, strix::resolve_execution_buffer(owner->serial.execution_identity,
            owner->serial.arena, owner->memory->scratch(), logit_use), path);
        reference_sha[r] = hash_file(path);
    }
    const auto serial_session = session;
    const auto serial_state = dump_state(owner, args.output, "serial");
    restore_state(owner, initial, prefix, session);
    // Verify the actual restored bytes, including stale/inactive capacity.
    const auto restored = dump_state(owner, args.output, "restored");
    for (std::size_t i = 0; i < initial.size(); ++i)
        require(initial[i].sha == restored[i].sha && equal_files(initial[i].path, restored[i].path),
            "restored initial component differs");
    strix::HipKnownPrefixWindow window(owner->window, owner->serial, owner->model, config,
        owner->resources, owner->window_memory, owner->scope(), std::static_pointer_cast<const void>(owner));
    const auto result = window.execute(session, {teacher[static_cast<std::size_t>(args.start)],
        teacher[static_cast<std::size_t>(args.start) + 1U]});
    require(!owner->prefetch.active, "window left an active PLE lookup");
    bool full_logits_equal = true;
    for (std::size_t r = 0; r < 2; ++r) {
        const auto path = args.output / ("window-logits-" + std::to_string(r) + ".f32");
        const auto candidate = logits(owner, result.logits[r], path);
        candidate_sha[r] = hash_file(path);
        full_logits_equal = full_logits_equal && candidate.size() == reference[r].size() &&
            std::memcmp(candidate.data(), reference[r].data(), candidate.size() * sizeof(float)) == 0;
    }
    const auto window_state = dump_state(owner, args.output, "window");
    bool all_state_equal = serial_state.size() == window_state.size();
    for (std::size_t i = 0; i < serial_state.size() && i < window_state.size(); ++i)
        all_state_equal = all_state_equal && serial_state[i].component == window_state[i].component &&
            serial_state[i].bytes == window_state[i].bytes && serial_state[i].sha == window_state[i].sha &&
            equal_files(serial_state[i].path, window_state[i].path);
    const auto expected_lineage = strix::prepare_prefill_chunk_lineage(prefix.arithmetic_lineage,
        args.start, args.start + 2U, owner->window.window_execution_identity,
        owner->window.window_kernel_plan_identity, owner->window.window_execution_identity);
    const bool lineage_equal = session.arithmetic_lineage.intervals == expected_lineage.intervals &&
        session.arithmetic_lineage.history_identity == expected_lineage.history_identity &&
        session.arithmetic_lineage.tainted == expected_lineage.tainted;
    const auto position = strix::resolve_session_state(owner->layout, owner->memory->state(),
        strix::SessionStateComponent::position_history);
    require(position.storage == strix::SessionStateStorage::host && position.bytes >= sizeof(std::uint64_t),
        "unexpected position history storage");
    std::uint64_t position_word = 0;
    std::memcpy(&position_word, reinterpret_cast<const void*>(position.address), sizeof(position_word));
    const bool metadata_equal = session.next_position == args.start + 2U &&
        session.committed_tokens == serial_session.committed_tokens &&
        session.state.valid == serial_session.state.valid && session.state.generations == serial_session.state.generations &&
        session.health == strix::SerialSessionHealth::ready && position_word == args.start + 1U &&
        session.execution_identity == prefix.execution_identity &&
        session.runtime_compatibility_identity == prefix.runtime_compatibility_identity &&
        session.state_layout_identity == prefix.state_layout_identity;
    require(hash_file(args.teacher) == args.teacher_sha && hash_file(args.model) == args.model_sha &&
        hash_file(executable) == executable_sha, "input/model/executable content changed during qualification");
    const bool passed = full_logits_equal && all_state_equal && lineage_equal && metadata_equal;
    const auto pending = args.output / "result.pending.json";
    auto output = output_file(pending);
    output << "{\"schema\":\"known-prefix-model-qualification-v1\",\"finished\":true,\"passed\":"
        << (passed ? "true" : "false") << ",\"profile\":" << quote(args.profile)
        << ",\"model_sha256\":" << quote(args.model_sha) << ",\"model_content_identity\":"
        << quote(hex(owner->mapped->content_identity())) << ",\"teacher_sha256\":" << quote(args.teacher_sha)
        << ",\"executable_sha256\":" << quote(executable_sha) << ",\"context_tokens\":" << context_tokens
        << ",\"start_position\":" << args.start << ",\"tokens\":[" << teacher[static_cast<std::size_t>(args.start)]
        << ',' << teacher[static_cast<std::size_t>(args.start) + 1U] << "]"
        << ",\"serial_execution_identity\":" << owner->serial.execution_identity
        << ",\"serial_kernel_plan_identity\":" << owner->serial.kernel_plan_identity
        << ",\"manifest_layout_identity\":" << owner->manifest.layout_identity
        << ",\"tensor_access_plan_identity\":" << owner->access.plan_identity
        << ",\"window_execution_identity\":" << owner->window.window_execution_identity
        << ",\"window_kernel_plan_identity\":" << owner->window.window_kernel_plan_identity
        << ",\"r2_steps\":" << owner->window.r2_steps
        << ",\"logical_projection_weight_bytes_avoided\":" << owner->window.logical_projection_weight_bytes_avoided
        << ",\"projection_launches_saved\":" << owner->window.projection_launches_saved
        << ",\"r2_dispatches\":" << result.r2_dispatches << ",\"row_loop_dispatches\":" << result.row_loop_dispatches
        << ",\"full_logits_bits_equal\":" << (full_logits_equal ? "true" : "false")
        << ",\"all_seven_state_bytes_equal\":" << (all_state_equal ? "true" : "false")
        << ",\"six_generations_and_position_equal\":" << (metadata_equal ? "true" : "false")
        << ",\"expected_chunk_lineage_equal\":" << (lineage_equal ? "true" : "false")
        << ",\"quality_qualification\":false,\"performance_measurement\":false,\"logits\":[";
    for (std::size_t r = 0; r < 2; ++r)
        output << (r ? "," : "") << "{\"row\":" << r << ",\"serial_sha256\":" << quote(reference_sha[r])
            << ",\"window_sha256\":" << quote(candidate_sha[r]) << '}';
    output << "],\"initial_state\":"; write_states(output, initial);
    output << ",\"restored_state\":"; write_states(output, restored);
    output << ",\"serial_state\":"; write_states(output, serial_state);
    output << ",\"window_state\":"; write_states(output, window_state);
    output << ",\"initial_session\":"; write_session(output, prefix);
    output << ",\"serial_session\":"; write_session(output, serial_session);
    output << ",\"window_session\":"; write_session(output, session);
    output << "}\n";
    output.close();
    require(!std::filesystem::exists(args.output / "result.json"), "result destination already exists");
    std::filesystem::rename(pending, args.output / "result.json");
    return passed ? 0 : 2;
}
} // namespace

int main(int argc, char** argv) {
    try { return qualify(options(argc, argv)); }
    catch (const std::exception& error) {
        std::cerr << "known-prefix model qualification failed: " << error.what() << '\n';
        return 1;
    }
}
