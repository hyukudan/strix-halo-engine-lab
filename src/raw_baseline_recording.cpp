#include "strix/raw_baseline_recording.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace strix {
namespace {
constexpr std::array<const char*, 7> names{"attention_kv", "recurrent_matrix", "gdn_convolution", "ple_convolution", "sparse_index", "ngram_history", "position_history"};
constexpr std::array<const char*, 7> classes{"kv", "recurrent", "convolution", "convolution", "sparse_index", "host_ngram", "position"};
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("raw recording byte overflow");
    return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::overflow_error("raw recording byte overflow");
    return a * b;
}
std::string quote(const std::string& text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += static_cast<char>(c);
    }
    return out + '"';
}
bool digest(const std::string& text) {
    return text.size() == 64 && text != std::string(64, '0') && std::all_of(text.begin(), text.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
std::string numbers(std::span<const std::uint32_t> values) {
    std::ostringstream out; out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    return out.str() + ']';
}
std::string generations(const SerialSessionContext& session) {
    std::ostringstream out; out << '[';
    for (std::size_t i = 0; i < session.state.generations.size(); ++i) { if (i) out << ','; out << session.state.generations[i]; }
    return out.str() + ']';
}
void private_parent(std::filesystem::path path) {
    for (;;) {
        if (std::filesystem::exists(path / ".git")) throw std::invalid_argument("raw recordings must remain outside Git");
        const auto parent = path.parent_path(); if (parent == path || parent.empty()) break; path = parent;
    }
}
void word(std::uint32_t value, std::span<std::byte, 4> bytes) {
    for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<std::byte>((value >> (i * 8U)) & 255U);
}
}
void validate_raw_baseline_output_directory(const std::filesystem::path& path) {
    if (!path.is_absolute() || path.filename().empty() || std::filesystem::exists(path) || std::filesystem::is_symlink(path)) throw std::invalid_argument("raw output must be a fresh absolute private directory");
    const auto parent = std::filesystem::canonical(path.parent_path()); private_parent(parent);
}
struct RawBaselineRecording::Impl {
    std::filesystem::path root;
    RawBaselineMetadata metadata;
    RawBaselinePolicy policy;
    SerialSessionStateLayout layout;
    std::array<SessionStateAllocation, 7> allocations;
    std::ofstream steps, states, history;
    std::vector<std::uint32_t> inputs, argmax;
    std::map<std::string, std::pair<std::uint64_t, std::string>> files;
    std::uint64_t written = 0;
    bool initial = false, failed = false, finalized = false;
    SerialSessionContext last;
    std::vector<std::byte> copy_chunk;
    RawBaselineStateReader uncertain_reader_owner;
    bool unknown_copy = false;
    void current() const {
        if (failed || finalized || std::filesystem::canonical(root) != root) throw std::logic_error("raw recording failed, finalized or moved");
        private_parent(root);
    }
    void charge(std::uint64_t bytes) {
        const auto total = add(written, bytes);
        if (total > policy.maximum_bytes) throw std::out_of_range("raw recording exceeds explicit byte budget");
        written = total;
    }
    void write(std::ofstream& output, std::span<const std::byte> bytes) {
        current(); charge(bytes.size());
        for (std::size_t offset = 0; offset < bytes.size();) {
            const auto count = std::min<std::size_t>(policy.chunk_bytes, bytes.size() - offset);
            output.write(reinterpret_cast<const char*>(bytes.data() + offset), static_cast<std::streamsize>(count));
            if (!output) throw std::runtime_error("raw recording write outcome is incomplete");
            offset += count;
        }
    }
    void text(std::ofstream& output, const std::string& value) { write(output, std::as_bytes(std::span{value})); }
    void close(std::ofstream& output) {
        output.flush(); if (!output) throw std::runtime_error("raw recording flush outcome is incomplete");
        output.close(); if (output.fail()) throw std::runtime_error("raw recording close outcome is incomplete");
    }
    std::string artifact(const std::string& filename) {
        const auto path = root / filename;
        const auto bytes = std::filesystem::file_size(path);
        const auto hash = model_content_identity_hex(sha256_file_content(path));
        if (!files.emplace(filename, std::pair{bytes, hash}).second) throw std::logic_error("raw artifact reused");
        return "{\"path\":" + quote(filename) + ",\"bytes\":" + std::to_string(bytes) + ",\"sha256\":" + quote(hash) + '}';
    }
    std::ofstream fresh(const std::string& filename) {
        current();
        if (std::filesystem::exists(root / filename) || std::filesystem::is_symlink(root / filename)) throw std::runtime_error("raw artifact path is not fresh");
        std::ofstream output(root / filename, std::ios::binary);
        if (!output) throw std::runtime_error("cannot create raw artifact");
        return output;
    }
    void validate_session(const SerialSessionContext& session, std::uint64_t next) const {
        if (session.health != SerialSessionHealth::ready || session.next_position != next || session.committed_tokens != next ||
            session.execution_identity != metadata.execution_identity || session.state_layout_identity != layout.layout_identity ||
            session.runtime_compatibility_identity != metadata.runtime_compatibility_identity || static_cast<std::uint32_t>(session.state.valid) != 63)
            throw std::invalid_argument("raw snapshot session identity/position/state classes are inconsistent");
    }
    void snapshot(const std::string& label, const SerialSessionContext& session, const RawBaselineStateReader& reader) {
        if (!reader) throw std::invalid_argument("raw state copy callback is absent");
        copy_chunk.resize(policy.chunk_bytes);
        uncertain_reader_owner = reader; // Retain sources before exposing any destination.
        std::ostringstream record;
        record << "{\"kind\":" << quote(label) << ",\"next_position\":" << session.next_position
               << ",\"committed_tokens\":" << session.committed_tokens << ",\"state_valid\":63,\"state_generations\":" << generations(session) << ",\"allocations\":[";
        for (std::size_t i = 0; i < allocations.size(); ++i) {
            const auto& allocation = allocations[i];
            const auto filename = label + '-' + names[i] + ".raw";
            auto output = fresh(filename);
            for (std::uint64_t offset = 0; offset < allocation.bytes;) {
                const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(copy_chunk.size(), allocation.bytes - offset));
                try { reader(allocation, offset, std::span{copy_chunk}.first(count)); }
                catch (...) { unknown_copy = true; throw; }
                write(output, std::span{copy_chunk}.first(count)); offset += count;
            }
            close(output);
            if (i) record << ',';
            record << "{\"component\":" << quote(names[i]) << ",\"state_class\":" << quote(classes[i])
                   << ",\"state_flag\":" << static_cast<std::uint32_t>(allocation.state_flag)
                   << ",\"storage\":" << quote(allocation.storage == SessionStateStorage::host ? "host" : "device")
                   << ",\"layout_offset\":" << allocation.offset << ",\"provisional\":" << (allocation.provisional ? "true" : "false")
                   << ",\"artifact\":" << artifact(filename) << '}';
        }
        record << "]}\n"; text(states, record.str());
    }
};
RawBaselineRecording::RawBaselineRecording(std::filesystem::path output, RawBaselineMetadata metadata, RawBaselinePolicy policy, SerialSessionStateLayout layout)
    : impl_(std::make_unique<Impl>()) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    validate_raw_baseline_output_directory(output);
    if (!digest(metadata.model_sha256) || !digest(metadata.model_content_identity) || !digest(metadata.executable_sha256) ||
        metadata.gpu_identity.empty() || metadata.gpu_identity.size() > 256 || metadata.gpu_architecture.empty() || metadata.gpu_architecture.size() > 128 ||
        !metadata.execution_identity || !metadata.kernel_plan_identity || !metadata.runtime_compatibility_identity || !metadata.vocabulary_size || metadata.vocabulary_size > 1048576 ||
        metadata.supplied_tokens.empty() || !policy.maximum_steps || policy.maximum_steps > 1048576 || policy.maximum_steps > layout.context_tokens ||
        metadata.supplied_tokens.size() > policy.maximum_steps || !layout.layout_identity || !policy.maximum_bytes || policy.maximum_bytes > 1024ULL * 1024 * 1024 * 1024 ||
        policy.chunk_bytes < 4 || policy.chunk_bytes > 1048576 || policy.chunk_bytes % 4 ||
        (metadata.mode != RawBaselineMode::greedy && metadata.mode != RawBaselineMode::teacher_forced) ||
        (metadata.evidence != RawBaselineEvidence::native_eager && metadata.evidence != RawBaselineEvidence::injected_fixture) ||
        (metadata.activation_storage != RawBaselineActivationStorage::bf16 && metadata.activation_storage != RawBaselineActivationStorage::fp32) ||
        (policy.state_policy != RawBaselineStatePolicy::each && policy.state_policy != RawBaselineStatePolicy::final) ||
        (metadata.mode == RawBaselineMode::greedy ? !metadata.maximum_generated_tokens || !metadata.eos_token || *metadata.eos_token >= metadata.vocabulary_size : metadata.maximum_generated_tokens || metadata.eos_token))
        throw std::invalid_argument("raw baseline requires exact metadata and bounded eager policy");
    if (metadata.evidence == RawBaselineEvidence::native_eager && (!metadata.hip_runtime_version || !metadata.hip_driver_version || metadata.gpu_identity.size() != 32 || metadata.gpu_identity == std::string(32, '0') ||
        !std::all_of(metadata.gpu_identity.begin(), metadata.gpu_identity.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); })))
        throw std::invalid_argument("native RAW metadata requires actual nonzero UUID and runtime/driver versions");
    for (auto token : metadata.supplied_tokens) if (token >= metadata.vocabulary_size) throw std::invalid_argument("raw input token outside vocabulary");
    auto& i = *impl_; i.metadata = std::move(metadata); i.policy = policy; i.layout = std::move(layout);
    if (i.layout.allocations.size() != 7) throw std::invalid_argument("raw state layout must include all seven actual components");
    std::uint64_t state_bytes = 0;
    for (std::size_t n = 0; n < 7; ++n) {
        const auto component = static_cast<SessionStateComponent>(n);
        const auto* allocation = find_session_state_allocation(i.layout, component);
        const auto expected_flag = 1U << (n < 3 ? n : n - 1);
        if (!allocation || !allocation->bytes || static_cast<std::uint32_t>(allocation->state_flag) != expected_flag ||
            allocation->storage != (n >= 5 ? SessionStateStorage::host : SessionStateStorage::device) ||
            add(allocation->offset, allocation->bytes) > (n >= 5 ? i.layout.host_total_bytes : i.layout.device_total_bytes))
            throw std::invalid_argument("raw state allocation class/storage/range invalid");
        i.allocations[n] = *allocation; state_bytes = add(state_bytes, allocation->bytes);
        for (std::size_t prior = 0; prior < n; ++prior) if (i.allocations[prior].storage == allocation->storage &&
            allocation->offset < add(i.allocations[prior].offset, i.allocations[prior].bytes) && i.allocations[prior].offset < add(allocation->offset, allocation->bytes))
            throw std::invalid_argument("raw state allocations overlap");
    }
    const auto snapshots = policy.state_policy == RawBaselineStatePolicy::each ? add(policy.maximum_steps, 1) : 2;
    // Includes conservative bounded metadata allowance; actual writes share the same cap.
    const auto required = add(mul(state_bytes, snapshots), add(mul(policy.maximum_steps, add(mul(i.metadata.vocabulary_size, 4), 4096)), 65536));
    if (required > policy.maximum_bytes) throw std::out_of_range("raw worst-case state/logits/metadata exceed byte budget");
    i.root = std::filesystem::canonical(output.parent_path()) / output.filename();
    if (!std::filesystem::create_directory(i.root)) throw std::runtime_error("raw output directory is not fresh");
    std::ofstream marker(i.root / "INCOMPLETE", std::ios::binary); i.text(marker, "incomplete raw baseline\n"); i.close(marker);
    i.steps.open(i.root / "steps.jsonl", std::ios::binary); i.states.open(i.root / "states.jsonl", std::ios::binary); i.history.open(i.root / "history.u32le", std::ios::binary);
    if (!i.steps || !i.states || !i.history) throw std::runtime_error("cannot create raw recording metadata");
}
RawBaselineRecording::~RawBaselineRecording() {
    // A thrown reader may have exposed the destination to unfinished work.
    // Retain its bounded buffer and captured source owners until process exit;
    // this diagnostic writer has no release-proof API and never retries it.
    if (impl_ && impl_->unknown_copy) {
        struct Retained { std::mutex mutex; std::vector<std::unique_ptr<Impl>> owners; };
        static auto* retained = new Retained; // Reachable root; no automatic exit-time release proof.
        std::lock_guard guard(retained->mutex);
        retained->owners.push_back(std::move(impl_));
    }
}
void RawBaselineRecording::capture_initial(const SerialSessionContext& session, const RawBaselineStateReader& reader) {
    auto& i = *impl_;
    try { i.current(); if (i.initial) throw std::logic_error("raw initial state already captured"); i.validate_session(session, 0); i.snapshot("initial", session, reader); i.initial = true; i.last = session; }
    catch (...) { i.failed = true; throw; }
}
void RawBaselineRecording::record_token(std::uint32_t input, std::uint64_t position, std::span<const float> logits,
    const SerialSessionContext& session, const RawBaselineStateReader& reader, std::optional<std::uint32_t> target) {
    auto& i = *impl_;
    try {
        i.current();
        if (!i.initial || position != i.inputs.size() || position >= i.policy.maximum_steps || input >= i.metadata.vocabulary_size || logits.size() != i.metadata.vocabulary_size)
            throw std::invalid_argument("raw token/logits position/count mismatch");
        i.validate_session(session, add(position, 1));
        for (std::size_t n = 0; n < 6; ++n) if (session.state.generations[n] < i.last.state.generations[n]) throw std::invalid_argument("raw state generation moved backwards");
        const bool supplied = position < i.metadata.supplied_tokens.size();
        if (!supplied && i.metadata.mode == RawBaselineMode::greedy && i.argmax.back() == *i.metadata.eos_token)
            throw std::invalid_argument("raw greedy input continues after emitted EOS");
        if (supplied ? input != i.metadata.supplied_tokens[static_cast<std::size_t>(position)] : i.metadata.mode != RawBaselineMode::greedy || input != i.argmax.back())
            throw std::invalid_argument("raw input history does not match supplied/greedy sequence");
        const auto expected_target = i.metadata.mode == RawBaselineMode::teacher_forced && position + 1 < i.metadata.supplied_tokens.size()
            ? std::optional{i.metadata.supplied_tokens[static_cast<std::size_t>(position + 1)]} : std::nullopt;
        if (target != expected_target) throw std::invalid_argument("raw teacher-forced target is not explicit next supplied token");
        std::uint32_t selected = 0;
        for (std::size_t n = 0; n < logits.size(); ++n) {
            if (!std::isfinite(logits[n])) throw std::invalid_argument("raw complete logits contain nonfinite value");
            if (logits[n] > logits[selected]) selected = static_cast<std::uint32_t>(n);
        }
        const auto filename = "step-" + std::to_string(position) + ".f32le";
        auto output = i.fresh(filename);
        std::vector<std::byte> chunk(i.policy.chunk_bytes);
        for (std::size_t base = 0; base < logits.size();) {
            const auto count = std::min<std::size_t>(chunk.size() / 4, logits.size() - base);
            for (std::size_t n = 0; n < count; ++n) word(std::bit_cast<std::uint32_t>(logits[base + n]), std::span<std::byte, 4>{chunk.data() + n * 4, 4});
            i.write(output, std::span{chunk}.first(count * 4)); base += count;
        }
        i.close(output);
        std::array<std::byte, 4> token{}; word(input, token); i.write(i.history, token);
        std::ostringstream record;
        record << "{\"position\":" << position << ",\"input_token\":" << input << ",\"role\":"
               << quote(i.metadata.mode == RawBaselineMode::teacher_forced ? "teacher_forced" : supplied ? "prefix" : "generation_fed")
               << ",\"history_length\":" << position + 1 << ",\"next_position\":" << session.next_position << ",\"argmax_token\":" << selected
               << ",\"teacher_forced_target\":" << (target ? std::to_string(*target) : "null") << ",\"logits_count\":" << logits.size()
               << ",\"logits_dtype\":\"fp32-le\",\"logits\":" << i.artifact(filename) << "}\n";
        i.text(i.steps, record.str());
        if (i.policy.state_policy == RawBaselineStatePolicy::each) i.snapshot("post-" + std::to_string(position), session, reader);
        i.inputs.push_back(input); i.argmax.push_back(selected); i.last = session;
    } catch (...) { i.failed = true; throw; }
}
void RawBaselineRecording::finalize(const SerialSessionContext& session, const RawBaselineStateReader& reader,
    std::span<const std::uint32_t> generated, std::string stop_reason, std::function<void()> verify_binding) {
    auto& i = *impl_;
    try {
        i.current(); i.validate_session(session, i.inputs.size());
        if (!i.initial || i.inputs.empty() || i.inputs.size() < i.metadata.supplied_tokens.size() || session.state.generations != i.last.state.generations)
            throw std::invalid_argument("raw capture incomplete before finalization");
        if (i.metadata.mode == RawBaselineMode::teacher_forced) {
            if (!generated.empty() || stop_reason != "raw_input_exhausted" || i.inputs.size() != i.metadata.supplied_tokens.size()) throw std::invalid_argument("raw teacher-forced capture cannot invent generated outputs");
        } else {
            const auto begin = i.metadata.supplied_tokens.size() - 1;
            if (generated.size() != i.argmax.size() - begin || generated.size() > i.metadata.maximum_generated_tokens || !std::equal(generated.begin(), generated.end(), i.argmax.begin() + static_cast<std::ptrdiff_t>(begin)) ||
                (stop_reason != "eos" && stop_reason != "max_new_tokens" && stop_reason != "context_capacity")) throw std::invalid_argument("raw emitted greedy tokens/stop do not match executed logits");
            if (stop_reason == "eos" ? generated.back() != *i.metadata.eos_token || std::find(generated.begin(), generated.end() - 1, *i.metadata.eos_token) != generated.end() - 1 :
                (std::find(generated.begin(), generated.end(), *i.metadata.eos_token) != generated.end() || (stop_reason == "max_new_tokens" ? generated.size() != i.metadata.maximum_generated_tokens || session.next_position >= i.layout.context_tokens : session.next_position != i.layout.context_tokens)))
                throw std::invalid_argument("raw greedy stop reason contradicts exact token/context policy");
        }
        if (i.policy.state_policy == RawBaselineStatePolicy::final) i.snapshot("final", session, reader);
        i.close(i.steps); i.close(i.states); i.close(i.history);
        const auto steps = i.artifact("steps.jsonl"), states = i.artifact("states.jsonl"), history = i.artifact("history.u32le");
        for (const auto& [name, expected] : i.files) if (std::filesystem::file_size(i.root / name) != expected.first || model_content_identity_hex(sha256_file_content(i.root / name)) != expected.second)
            throw std::runtime_error("raw artifact changed before receipt publication");
        if (verify_binding) verify_binding();
        std::ostringstream record;
        record << "{\"schema\":\"strix-raw-eager-baseline-v1\",\"complete\":true,\"qualified\":false,\"performance_evidence\":false,\"source_binding\":\"external_compile_receipt_required\",\"tokenizer_binding\":\"external_raw_tokens_only\",\"evidence_kind\":"
               << quote(i.metadata.evidence == RawBaselineEvidence::native_eager ? "native-eager-raw" : "injected-fixture")
               << ",\"mode\":" << quote(i.metadata.mode == RawBaselineMode::greedy ? "raw_greedy" : "raw_teacher_forced")
               << ",\"activation_storage\":" << quote(i.metadata.activation_storage == RawBaselineActivationStorage::bf16 ? "bf16" : "fp32")
               << ",\"state_policy\":" << quote(i.policy.state_policy == RawBaselineStatePolicy::each ? "each" : "final")
               << ",\"model_sha256\":" << quote(i.metadata.model_sha256) << ",\"model_content_identity\":" << quote(i.metadata.model_content_identity)
               << ",\"executable_sha256\":" << quote(i.metadata.executable_sha256) << ",\"gpu_identity\":" << quote(i.metadata.gpu_identity)
               << ",\"gpu_architecture\":" << quote(i.metadata.gpu_architecture) << ",\"hip_runtime_version\":" << i.metadata.hip_runtime_version
               << ",\"hip_driver_version\":" << i.metadata.hip_driver_version << ",\"execution_identity\":" << i.metadata.execution_identity
               << ",\"kernel_plan_identity\":" << i.metadata.kernel_plan_identity << ",\"runtime_compatibility_identity\":" << i.metadata.runtime_compatibility_identity
               << ",\"layout_identity\":" << i.layout.layout_identity << ",\"context_tokens\":" << i.layout.context_tokens
               << ",\"vocabulary_size\":" << i.metadata.vocabulary_size << ",\"supplied_tokens\":" << numbers(i.metadata.supplied_tokens)
               << ",\"generated_output_tokens\":" << numbers(generated) << ",\"maximum_generated_tokens\":" << i.metadata.maximum_generated_tokens
               << ",\"eos_token\":" << (i.metadata.eos_token ? std::to_string(*i.metadata.eos_token) : "null")
               << ",\"stop_reason\":" << quote(stop_reason) << ",\"steps_executed\":" << i.inputs.size() << ",\"final_position\":" << session.next_position
               << ",\"argmax_rule\":\"first_index_on_exact_tie\",\"history_contract\":\"single_row_eager_from_fresh_initialized_state\",\"state_encoding\":\"opaque_native_bytes\",\"native_byte_order\":"
               << quote(std::endian::native == std::endian::little ? "little" : "big") << ",\"steps\":" << steps << ",\"states\":" << states << ",\"history\":" << history << "}\n";
        auto receipt = i.fresh("receipt.pending");
        i.text(receipt, record.str()); i.close(receipt);
        if (!std::filesystem::remove(i.root / "INCOMPLETE")) throw std::runtime_error("raw incomplete marker lost before publication");
        std::filesystem::rename(i.root / "receipt.pending", i.root / "receipt.json");
        i.finalized = true;
    } catch (...) { i.failed = true; throw; }
}
} // namespace strix
