#include "deferred_moe_receipt.hpp"
#include "strix/lane_chain_hyper_output.hpp"
#include "qsa_input_receipt.hpp"
#include "shared_prefetch_receipt.hpp"
#include "strix/gdn_output_admission.hpp"
#include "strix/compiled_execution.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_gdn_islands.hpp"
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_known_prompt_backend.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/model_manifest.hpp"
#include "strix/wave_router_contract.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/raw_baseline_recording.hpp"
#include "strix/final_generation_state_export.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include "strix/shared_expert_prefetch.hpp"
#include "strix/decode_execution_options.hpp"
#include "strix/decode_integer_head.hpp"
#include "strix/hip_integer_projection_runtime.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/generation_output_dispatch.hpp"
#include "strix/prepared_serial_session.hpp"
#include "strix/moe_fork_join_arena.hpp"
#include "strix/hip_moe_overlap_dispatcher.hpp"
#include "hip_decode_experiment.hpp"
#include "hip_decode_options.hpp"
#include "strix/q8_projection_organization.hpp"
#include "strix/q8_projection_manifest.hpp"
#include "strix/hc_projection_prefetch.hpp"
#include "hip_deferred_moe_status.hpp"
#include "hip_decode_receipt_json.hpp"
#include "execution_inventory_receipt.hpp"
#include "strix/serial_decode_reference.hpp"
#include "strix/serial_decode_replay.hpp"
#include "strix/serial_prefix_cache.hpp"
#include "strix/serial_request.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_state_layout.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <memory>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using strix::decode_cli::parse_options;

// One optional stderr JSON record; stdout keeps its existing decoder receipt.
// Runs during failure unwinding too. Observability must not mask an execution
// error or change publication/cleanup ordering.
struct HostTelemetryReport {
    strix::HostDispatchCounters counters;
    bool enabled = false;
    ~HostTelemetryReport() noexcept {
        if (!enabled) return;
        try {
            strix::write_host_dispatch_telemetry(std::cerr, counters);
            std::cerr << '\n';
        } catch (...) {}
    }
};

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

void check_raw_copy(hipError_t error, const char* operation, const std::function<void()>& drain) {
    if (error != hipSuccess && drain) drain();
    check_hip(error, operation);
}

strix::PersistentState all_persistent_state() {
    using strix::PersistentState;
    return
        PersistentState::attention_kv |
        PersistentState::recurrent_matrix |
        PersistentState::convolution_history |
        PersistentState::sparse_index |
        PersistentState::ngram_history |
        PersistentState::position_history;
}

class MappedModelIoSource final
    : public strix::SparseLookupIoSource {
public:
    explicit MappedModelIoSource(
        const strix::MappedModelFile& mapped)
        : mapped_(&mapped) {}

    void read(
        std::uint64_t byte_offset,
        std::span<std::byte> destination) override {
        mapped_->read_at(
            byte_offset,
            destination);
    }

private:
    const strix::MappedModelFile* mapped_ = nullptr;
};


struct RankedLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

std::vector<RankedLogit> top_logits(
    const std::vector<float>& logits,
    std::size_t count) {
    std::vector<std::uint32_t> order(
        logits.size());
    for (std::size_t index = 0;
         index < order.size();
         ++index) {
        order[index] =
            static_cast<std::uint32_t>(
                index);
    }

    count =
        std::min(
            count,
            order.size());
    std::partial_sort(
        order.begin(),
        order.begin() +
            static_cast<
                std::ptrdiff_t>(
                count),
        order.end(),
        [&](std::uint32_t left,
            std::uint32_t right) {
            if (logits[left] !=
                logits[right]) {
                return logits[left] >
                       logits[right];
            }
            return left < right;
        });

    std::vector<RankedLogit> result;
    result.reserve(count);
    for (std::size_t index = 0;
         index < count;
         ++index) {
        const auto token =
            order[index];
        result.push_back(
            RankedLogit{
                token,
                logits[token],
            });
    }
    return result;
}

class HipSerialGreedyStepBackend final
    : public strix::SerialGreedyStepBackend {
public:
    HipSerialGreedyStepBackend(
        const strix::CompiledSerialExecution& compiled,
        strix::SerialSessionContext& session,
        const strix::ExecutionRuntimeResources& resources,
        strix::EagerExecutionDispatcher& dispatcher,
        const strix::ExecutionMemoryBinding& scratch,
        const strix::ExecutionStepIo& lm_io,
        std::uint64_t vocab_size,
        std::function<void(std::uint32_t, std::uint64_t, std::span<const float>)> recording = {},
        std::function<void()> recording_binding = {}, std::function<void()> recording_drain = {},
        std::function<void(std::uint32_t)> execute_override = {}, bool track_logit_hashes = false,
        bool prepublication_completion = false, strix::HipGreedyOutput* greedy_output = nullptr,
        strix::GenerationOutputDispatcher* output_dispatcher = nullptr)
        : compiled_(&compiled),
          session_(&session),
          resources_(&resources),
          dispatcher_(&dispatcher),
          scratch_(&scratch),
          lm_io_(&lm_io),
          vocab_size_(vocab_size), recording_(std::move(recording)), recording_binding_(std::move(recording_binding)), recording_drain_(std::move(recording_drain)), execute_override_(std::move(execute_override)), track_logit_hashes_(track_logit_hashes), prepublication_completion_(prepublication_completion), greedy_output_(greedy_output), output_dispatcher_(output_dispatcher) {
        if(prepublication_completion_&&!execute_override_)
            throw std::invalid_argument("checked completion requires the owned native token executor");
        if (greedy_output_ && (recording_ || track_logit_hashes_))
            throw std::invalid_argument("scalar greedy output cannot omit requested full-logit evidence");
        if (greedy_output_ && !output_dispatcher_)
            throw std::invalid_argument("scalar greedy output requires its prefix output dispatcher");
    }

    void execute_state(std::uint32_t input_token, std::uint64_t position) override {
        if (!greedy_output_) {
            strix::SerialGreedyStepBackend::execute_state(input_token, position);
            return;
        }
        if (position != session_->next_position || session_->health != strix::SerialSessionHealth::ready)
            throw std::invalid_argument("state-only prefix differs from live session");
        const auto start = std::chrono::steady_clock::now();
        strix::GenerationOutputDispatcher::StateOnlyScope state_only(*output_dispatcher_);
        const auto committed_before = session_->committed_tokens;
        try {
            if (execute_override_) execute_override_(input_token);
            else static_cast<void>(strix::execute_serial_token(
                *compiled_, *session_, *resources_, input_token, *dispatcher_));
            if (!prepublication_completion_)
                check_hip(hipDeviceSynchronize(), "state-only prefix producer completion");
            prefix_state_elapsed_ms_.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        } catch (...) {
            if (session_->committed_tokens != committed_before) {
                session_->health = strix::SerialSessionHealth::poisoned;
                session_->arithmetic_lineage.tainted = true;
            }
            if (recording_drain_) recording_drain_();
            throw;
        }
    }

    std::uint32_t execute_greedy(std::uint32_t input_token, std::uint64_t position,
                                 std::uint64_t vocab_size) override {
        if (!greedy_output_)
            return strix::SerialGreedyStepBackend::execute_greedy(input_token, position, vocab_size);
        if (vocab_size != vocab_size_ || position != session_->next_position ||
            session_->health != strix::SerialSessionHealth::ready)
            throw std::invalid_argument("scalar greedy request differs from live session");
        if (recording_binding_) recording_binding_();
        const auto start = std::chrono::steady_clock::now();
        const auto committed_before = session_->committed_tokens;
        try {
            if (execute_override_) execute_override_(input_token);
            else static_cast<void>(strix::execute_serial_token(
                *compiled_, *session_, *resources_, input_token, *dispatcher_));
            // The output owner uses its own stream. The producer must have
            // completed, including deferred status, before it may read logits.
            if (!prepublication_completion_)
                check_hip(hipDeviceSynchronize(), "scalar greedy producer completion");
            const auto logits = strix::resolve_execution_buffer(compiled_->execution_identity,
                compiled_->arena, *scratch_, lm_io_->writes[0]);
            if (recording_binding_) recording_binding_();
            const auto token = greedy_output_->select(logits, static_cast<std::uint32_t>(vocab_size_),
                position, session_->committed_tokens);
            if (recording_binding_) recording_binding_();
            elapsed_ms_.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            return token;
        } catch (...) {
            // State may already be consumed. Do not roll counters back or
            // allow another token after an output failure. The result owner
            // drains its own work before throwing; retain the original path's
            // producer drain as well.
            if (session_->committed_tokens != committed_before) {
                session_->health = strix::SerialSessionHealth::poisoned;
                session_->arithmetic_lineage.tainted = true;
            }
            if (recording_drain_) recording_drain_();
            throw;
        }
    }

    std::vector<float> execute(
        std::uint32_t input_token,
        std::uint64_t position) override {
        if (position != session_->next_position) {
            throw std::invalid_argument(
                "greedy request position does not match serial session");
        }
        if (recording_binding_) recording_binding_();

        const auto start =
            std::chrono::steady_clock::now();

        try {
        if (execute_override_) execute_override_(input_token);
        else
        static_cast<void>(
            strix::execute_serial_token(
                *compiled_,
                *session_,
                *resources_,
                input_token,
                *dispatcher_));

        if (!prepublication_completion_) check_hip(
            hipDeviceSynchronize(),
            "hipDeviceSynchronize(greedy serial token)");
        } catch (...) { if (recording_drain_) recording_drain_(); throw; }

        const auto stop =
            std::chrono::steady_clock::now();
        elapsed_ms_.push_back(
            std::chrono::duration<double, std::milli>(
                stop - start).count());

        const auto logits_address =
            strix::resolve_execution_buffer(
                compiled_->execution_identity,
                compiled_->arena,
                *scratch_,
                lm_io_->writes[0]);

        if (vocab_size_ >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(
                "greedy vocabulary size does not fit size_t");
        }
        std::vector<float> logits(
            static_cast<std::size_t>(
                vocab_size_));
        if (logits_address.bytes <
            logits.size() * sizeof(float)) {
            throw std::logic_error(
                "compiled logits allocation is undersized");
        }

        if (recording_binding_) recording_binding_();
        check_raw_copy(
            hipMemcpy(
                logits.data(),
                reinterpret_cast<const void*>(
                    logits_address.address),
                logits.size() * sizeof(float),
                hipMemcpyDeviceToHost),
            "hipMemcpy(greedy logits D2H)", recording_drain_);
        if (recording_binding_) recording_binding_();

        strix::validate_serial_decode_engine_logits(
            logits,
            vocab_size_);
        if (track_logit_hashes_) logit_hashes_.push_back(strix::model_content_identity_hex(
            strix::sha256_bytes(std::as_bytes(std::span<const float>(logits)))));
        if (recording_) recording_(input_token, position, logits);
        return logits;
    }

    [[nodiscard]] const std::vector<double>&
    elapsed_ms() const noexcept {
        return elapsed_ms_;
    }
    [[nodiscard]] const std::vector<std::string>& logit_hashes() const noexcept {
        return logit_hashes_;
    }
    [[nodiscard]] const std::vector<double>& prefix_state_elapsed_ms() const noexcept {
        return prefix_state_elapsed_ms_;
    }

private:
    const strix::CompiledSerialExecution* compiled_ = nullptr;
    strix::SerialSessionContext* session_ = nullptr;
    const strix::ExecutionRuntimeResources* resources_ = nullptr;
    strix::EagerExecutionDispatcher* dispatcher_ = nullptr;
    const strix::ExecutionMemoryBinding* scratch_ = nullptr;
    const strix::ExecutionStepIo* lm_io_ = nullptr;
    std::uint64_t vocab_size_ = 0;
    std::vector<double> elapsed_ms_;
    std::vector<double> prefix_state_elapsed_ms_;
    std::vector<std::string> logit_hashes_;
    std::function<void(std::uint32_t, std::uint64_t, std::span<const float>)> recording_;
    std::function<void()> recording_binding_;
    std::function<void()> recording_drain_;
    std::function<void(std::uint32_t)> execute_override_;
    bool track_logit_hashes_ = false;
    bool prepublication_completion_ = false;
    strix::HipGreedyOutput* greedy_output_ = nullptr;
    strix::GenerationOutputDispatcher* output_dispatcher_ = nullptr;
};

// A separate physical window arena and completion record. Serial D1 keeps its
// existing shadow transaction; known pairs never call its begin/complete path.
class KnownPromptMemory final {
public:
    KnownPromptMemory(const strix::CompiledKnownPrefixWindow& plan, std::uint64_t arena_generation,
        std::uint64_t status_generation, std::uint64_t context, std::uint32_t vocabulary,
        std::shared_ptr<const void> retained, std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard)
        : retained_(std::move(retained)), guard_(std::move(guard)) {
        if (!retained_ || !guard_) throw std::invalid_argument("known prompt requires retained original resources");
        scratch = {plan.window_execution_identity, 0, plan.arena.physical.total_bytes, arena_generation};
        if (!scratch.execution_identity || !scratch.arena_bytes || !arena_generation || !status_generation)
            throw std::invalid_argument("known prompt memory requires current nonempty bindings");
        try {
            guard_->run([&] {
                check_hip(hipStreamCreateWithFlags(&scope.stream, hipStreamNonBlocking), "create known-prompt stream");
                check_hip(hipMalloc(&arena_, static_cast<std::size_t>(scratch.arena_bytes)), "allocate known-prompt arena");
                scratch.arena_base = reinterpret_cast<std::uintptr_t>(arena_);
                check_hip(hipMalloc(&status_, 8), "allocate known-prompt completion record");
                check_hip(hipMemsetAsync(arena_, 0, static_cast<std::size_t>(scratch.arena_bytes), scope.stream), "initialize known-prompt arena");
                check_hip(hipMemsetAsync(status_, 0, 8, scope.stream), "initialize known-prompt completion record");
            });
            scope.context_tokens = context; scope.vocabulary_size = vocabulary;
            scope.deferred_moe_status = {plan.window_execution_identity,
                reinterpret_cast<std::uintptr_t>(status_), 8, status_generation};
        } catch (...) { release(); throw; }
    }
    ~KnownPromptMemory() { release(); }
    KnownPromptMemory(const KnownPromptMemory&) = delete;
    KnownPromptMemory& operator=(const KnownPromptMemory&) = delete;
    strix::ExecutionMemoryBinding scratch;
    strix::HipLaunchContext scope;
private:
    void release() noexcept {
        guard_->recover_or_fail_stop();
        if (arena_ && hipFree(arena_) != hipSuccess) std::_Exit(3);
        arena_ = nullptr;
        if (status_ && hipFree(status_) != hipSuccess) std::_Exit(3);
        status_ = nullptr;
        if (scope.stream && hipStreamDestroy(scope.stream) != hipSuccess) std::_Exit(3);
        scope.stream = nullptr;
    }
    std::shared_ptr<const void> retained_;
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard_;
    void* arena_ = nullptr; void* status_ = nullptr;
};

// Drain inside the transfer call before the caller's snapshot owner can unwind.
class GuardedHipSnapshotTransfer final : public strix::SessionSnapshotTransferBackend {
public:
    GuardedHipSnapshotTransfer(std::function<void()> current, std::function<void()> drain)
        : current_(std::move(current)), drain_(std::move(drain)) {}
    void capture_device(std::uintptr_t source, std::uintptr_t destination, std::uint64_t bytes) override { call([&] { backend_.capture_device(source, destination, bytes); }); }
    void capture_host(std::uintptr_t source, std::uintptr_t destination, std::uint64_t bytes) override { call([&] { backend_.capture_host(source, destination, bytes); }); }
    void restore_device(std::uintptr_t source, std::uintptr_t destination, std::uint64_t bytes) override { call([&] { backend_.restore_device(source, destination, bytes); }); }
    void restore_host(std::uintptr_t source, std::uintptr_t destination, std::uint64_t bytes) override { call([&] { backend_.restore_host(source, destination, bytes); }); }
private:
    template<class Function> void call(Function&& function) {
        try { if (current_) current_(); function(); if (current_) current_(); }
        catch (...) { if (drain_) drain_(); throw; }
    }
    strix::HipSessionSnapshotTransferBackend backend_;
    std::function<void()> current_, drain_;
};

std::string_view stop_reason_name(
    strix::SerialRequestStopReason reason) noexcept {
    switch (reason) {
    case strix::SerialRequestStopReason::eos:
        return "eos";
    case strix::SerialRequestStopReason::max_new_tokens:
        return "max_new_tokens";
    case strix::SerialRequestStopReason::context_capacity:
        return "context_capacity";
    }
    return "unknown";
}

void write_logit_hashes(std::span<const std::string> hashes) {
    std::cout << '[';
    for (std::size_t index = 0; index < hashes.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << '"' << hashes[index] << '"';
    }
    std::cout << ']';
}

}  // namespace

int main(
    int argc,
    char** argv) try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: strix-hip-decode-token MODEL.gguf TOKEN_ID ... [options]\n"
            "  --context N --generate N --eos-token ID --residency register|copy\n"
            "  --optimization-profile auto|manual (default auto, qualified checkpoint only)\n"
            "  --host-dispatch-telemetry (eager host launcher counters on stderr; profiling only)\n"
            "  --execution-inventory (include selected host launcher and declared variant metadata)\n"
            "  --gpu-greedy-token (experimental eager generation with small GPU selection result)\n"
            "  --batched-moe-activation (experimental eager A8 slot activation batching)\n"
            "  --qsa-gqa-heads-per-block 1|2|4 (experimental eager grouped attention; default1)\n"
            "  --qsa-hierarchical-selection (experimental exact top512, eager D1, context<=131072)\n"
            "  --gdn-recurrent-layout baseline|stride32|stride33|transposed (experimental eager LDS organization)\n"
            "  --known-prefix-window (experimental paired known prompt; complete heads, separate window owner)\n"
            "  --gdn-islands (experimental eager D1 GDN launch organization; paired wave8 GDN supported)\n"
            "  --prepared-eager (experimental immutable static submission plan)\n"
            "  --moe-shared-overlap [--moe-overlap-arena-budget-bytes N] (experimental eager D1)\n"
            "  --virtual-q8-gdn-output (exact original256-tree Q8 output)\n"
            "  --wave8-q8-gdn-output (eager wave8 output; requires resolved virtual Q8 GDN output)\n"
            "  --wave8-q8-qsa-output (eager all-layer Q8 output organization)\n"
            "  --prefetch-hc-projection-next-tile (canonical eager typed HC projection overlay)\n"
            "  --q51-slot-parallel-down (experimental eager grouped Q5_1 raw-dot overlay)\n"
            "  --lane-chain-hyper-output (requires --prefetch-q8-hyper-output)\n"
            "  --record-raw ABS_NEW_PRIVATE_DIRECTORY --record-max-bytes BYTES\n"
            "  --record-state-policy final|each --prefix-cache-roundtrip\n"
            "  --prefetch-q8-qsa-input (opt-in exact-tree Q8 input projections)\n"
            "  --prefetch-q8-shared-expert (opt-in exact-tree Q8 gate/up/down)\n"
            "  --execution-mode eager|graph (default eager)\n"
            "  --deferred-moe-status (owned eager C1 completion before token publication)\n"
            "  --graph-workspace-bytes BYTES --graph-executable-bytes BYTES\n"
            "  --graph-capture-peak-bytes BYTES --graph-budget-bytes BYTES\n"
            "  --integer-lm-head a8-iu8|a4-iu8|a4-iu4\n"
            "  --integer-internal-bytes BYTES --integer-retained-bytes BYTES\n"
            "Graph requires --residency copy, including the complete PLE table.\n"
            "Integer mode changes only the eager LM head; IQ4_XS requires IU8.\n"
            "Graph and integer modes require explicit memory ceilings.\n"
            "All paths remain experimental; RAW recording is not performance evidence.\n";
        return 0;
    }
    auto options =
        parse_options(
            argc,
            argv);

    const auto model =
        strix::ModelContract::
            qwen38_flash_next();

    const auto generation_eos =
        options.eos_token.value_or(
            static_cast<std::uint32_t>(
                model.ple_eos_token_id));
    if (static_cast<std::uint64_t>(
            generation_eos) >=
        model.vocab_size) {
        throw std::invalid_argument(
            "generation EOS token is outside target model vocabulary");
    }

    std::optional<strix::SerialDecodeReference>
        reference;
    std::uint64_t required_reference_logit_values = 0;
    if (!options.reference_path.empty()) {
        reference =
            strix::read_serial_decode_reference(
                options.reference_path);
        strix::validate_serial_decode_reference_vocabulary(
            *reference,
            model.vocab_size);
        if (reference->steps.size() < 2) {
            throw std::invalid_argument(
                "serial replay qualification requires at least two reference steps");
        }
        for (const auto& step : reference->steps) {
            const auto probes =
                static_cast<std::uint64_t>(
                    step.logits.size());
            if (probes >
                std::numeric_limits<std::uint64_t>::max() -
                    required_reference_logit_values) {
                throw std::overflow_error(
                    "reference logit probe count overflow");
            }
            required_reference_logit_values += probes;
        }
        if (required_reference_logit_values == 0) {
            throw std::invalid_argument(
                "serial replay reference requires logit probes");
        }
        if (options.tokens.empty()) {
            options.tokens.reserve(
                reference->steps.size());
            for (const auto& step :
                 reference->steps) {
                options.tokens.push_back(
                    step.input_token);
            }
        } else {
            if (options.tokens.size() !=
                reference->steps.size()) {
                throw std::invalid_argument(
                    "CLI token sequence length does not match replay reference");
            }
            for (std::size_t index = 0;
                 index < options.tokens.size();
                 ++index) {
                if (options.tokens[index] !=
                    reference->steps[index].input_token) {
                    throw std::invalid_argument(
                        "CLI token sequence does not match replay reference");
                }
            }
        }
    }

    for (const auto token : options.tokens) {
        if (static_cast<std::uint64_t>(token) >=
            model.vocab_size) {
            throw std::invalid_argument(
                "input token is outside target model vocabulary");
        }
    }


    const auto info =
        strix::gguf::read_file(
            options.model_path);
    const auto manifest =
        strix::build_model_manifest(
            info);
    strix::AllocationGenerationSource
        generations;
    auto mapped_owner = std::make_shared<strix::MappedModelFile>(
        strix::MappedModelFile::open_read_only(options.model_path, "model", generations));
    auto& mapped = *mapped_owner;

#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    constexpr bool has_auto_a8_runtime = true;
#else
    constexpr bool has_auto_a8_runtime = false;
#endif
    const auto optimization_profile = strix::decode_cli::resolve_optimization_profile(
        options, model, info, manifest,
        strix::model_content_identity_hex(mapped.content_identity()), has_auto_a8_runtime);
    strix::decode_cli::validate_resolved_prefix_roundtrip(options);
    strix::decode_cli::validate_resolved_q8_projection_organization(options);
    strix::decode_cli::validate_resolved_hc_projection_next_tile(options);
    strix::decode_cli::validate_resolved_q51_slot_parallel_down(options);
    strix::decode_cli::validate_resolved_gdn_islands(options);
    strix::decode_cli::validate_resolved_gdn_recurrent_layout(options);
    strix::decode_cli::validate_resolved_qsa_hierarchical_selection(options);
    strix::decode_cli::validate_resolved_qsa_gqa(options);
    strix::decode_cli::validate_resolved_known_prefix_window(options);
    if (options.batched_moe_activation && !options.q4k_a8g32_gate_up)
        throw std::invalid_argument("batched MoE activation requires resolved Q4K A8 gate/up execution");
    if (options.moe_shared_overlap && !options.deferred_moe_status)
        throw std::invalid_argument("MoE shared overlap requires the resolved eager deferred-status owner");
    const std::string optimization_profile_receipt =
        "  \"optimization_profile_requested\": \"" + optimization_profile.requested + "\",\n" +
        "  \"optimization_profile_selected\": \"" + optimization_profile.selected + "\",\n" +
        "  \"optimization_profile_reason\": \"" + optimization_profile.reason + "\",\n" +
        "  \"optimization_profile_automatic_applied\": " +
        (optimization_profile.automatic_applied ? "true" : "false") + ",\n";
    const std::string eager_submission_receipt = options.prepared_eager
        ? "  \"eager_submission_mode\":\"prepared-static-plan-experimental-v1\",\n"
        : std::string{};
    if (options.context_tokens >
        model.max_position_embeddings) {
        throw std::invalid_argument(
            "requested context exceeds model maximum");
    }
    if (options.tokens.size() >
        options.context_tokens) {
        throw std::invalid_argument(
            "input token sequence exceeds allocated context");
    }

    if (options.wave_moe_router_topk) strix::validate_wave_router_model_inputs(info);
    // Admission validates all names and the actual layout hash once, before
    // any GPU setup or enqueue, then retains its immutable snapshot.
    const auto wave_router_admission = options.wave_moe_router_topk ?
        strix::WaveRouterManifestAdmission::create(manifest) : nullptr;
    const auto q4k_a8_layers = options.q4k_a8g32_gate_up ?
        strix::moe_q4k_a8::classify_runtime_manifest(model,manifest) :
        strix::moe_q4k_a8::RuntimeLayerClassification{};
    if(options.grouped_exact_q51_down && q4k_a8_layers.q4k_with_q51_down_layers.empty())
        throw std::invalid_argument("--grouped-exact-q51-down requires a Q4_K gate/up layer with Q5_1 down");
    if (options.q51_slot_parallel_down)
        strix::moe_q4k_a8::validate_slot_parallel_manifest(model, q4k_a8_layers);
    const auto q4k_a8_stats = options.q4k_a8g32_gate_up ?
        std::make_shared<strix::moe_q4k_a8::RuntimeDispatchStats>() : nullptr;
    const auto emit_q51_slot_receipt = [&] {
        if (options.q51_slot_parallel_down)
            std::cout << "  \"q51_slot_parallel_down\": true,\n"
                         "  \"q51_slot_parallel_down_qualification\": \"not_assessed\",\n";
    };
    const auto emit_q51_slot_counts = [&] {
        if (options.q51_slot_parallel_down) std::cout << "  \"slot_parallel_q51_submissions\": "
            << (q4k_a8_stats ? q4k_a8_stats->slot_parallel_q51_submissions.load(std::memory_order_relaxed) : 0) << ",\n";
    };
    const auto emit_q4k_a8_receipt = [&] {
        emit_q51_slot_receipt();
        std::cout << "  \"grouped_exact_q51_down\": " << (options.grouped_exact_q51_down ? "true" : "false") << ",\n"
            << "  \"grouped_q51_numerical_identity\": \""
            << (options.grouped_exact_q51_down ? strix::moe_q4k_a8::grouped_q51_numerical_identity : std::string_view{"legacy"}) << "\",\n"
            << "  \"q4k_a8g32_gate_up\": " << (options.q4k_a8g32_gate_up ? "true" : "false") << ",\n"
            << "  \"q4k_a8g32_numerical_identity\": \""
            << (options.q4k_a8g32_gate_up ? strix::moe_q4k_a8::runtime_numerical_identity : std::string_view{"legacy"}) << "\",\n"
            << "  \"q4k_a8g32_quantized_layers\": [";
        for (std::size_t i=0;i<q4k_a8_layers.quantized_layers.size();++i)
            std::cout << (i ? "," : "") << q4k_a8_layers.quantized_layers[i];
        std::cout << "],\n  \"q4k_a8g32_legacy_q5k_layers\": [";
        for (std::size_t i=0;i<q4k_a8_layers.legacy_q5k_layers.size();++i)
            std::cout << (i ? "," : "") << q4k_a8_layers.legacy_q5k_layers[i];
        std::cout << "],\n  \"grouped_q51_eligible_layers\": [";
        for(std::size_t i=0;i<q4k_a8_layers.q4k_with_q51_down_layers.size();++i)
            std::cout << (i ? "," : "") << q4k_a8_layers.q4k_with_q51_down_layers[i];
        std::cout << "],\n";
    };
    const auto emit_q4k_a8_counts = [&] {
        std::cout << "  \"q4k_a8g32_host_quantized_submissions\": "
            << (q4k_a8_stats ? q4k_a8_stats->quantized_submissions.load(std::memory_order_relaxed) : 0)
            << ",\n  \"q4k_a8g32_host_legacy_q5k_submissions\": "
            << (q4k_a8_stats ? q4k_a8_stats->legacy_q5k_submissions.load(std::memory_order_relaxed) : 0)
            << ",\n  \"grouped_q51_host_submissions\": "
            << (q4k_a8_stats ? q4k_a8_stats->grouped_q51_submissions.load(std::memory_order_relaxed) : 0) << ",\n";
        emit_q51_slot_counts();
        if (options.batched_moe_activation) std::cout << "  \"batched_moe_activation\":true,\n"
            << "  \"batched_moe_activation_launches\":"
            << (q4k_a8_stats ? q4k_a8_stats->batched_activation_launches.load(std::memory_order_relaxed) : 0) << ",\n"
            << "  \"batched_moe_activation_slots\":"
            << (q4k_a8_stats ? q4k_a8_stats->batched_activation_slots.load(std::memory_order_relaxed) : 0) << ",\n";

    };
    strix::HipModelResidencyPolicy lane_chain_policy;lane_chain_policy.mode=options.residency;lane_chain_policy.coalesce_gap_bytes=options.coalesce_gap_bytes;lane_chain_policy.promote_file_backed_candidates=options.execution.mode==strix::DecodeExecutionMode::graph;
    std::uint32_t lane_chain_total=0;
    const auto lane_chain_steps=options.lane_chain_hyper_output?strix::lane_chain_hyper_output_steps(model,manifest,lane_chain_policy,&lane_chain_total):std::vector<std::uint32_t>{};
    const auto tensor_index =
        strix::build_model_tensor_index(
            manifest);
    const auto q8_organization_plan = strix::q8_projection_organization::build_manifest_plan(
        model, manifest, options.prefetch_q8_lm_head, options.wave8_q8_gdn_input,
        options.wave8_q8_gdn_output, options.wave8_q8_qsa_output);
    if (options.virtual_q8_lm_head) {
        std::size_t output_heads = 0;
        for (const auto& tensor : manifest.tensors) {
            if (tensor.role != strix::TensorRole::output_head) continue;
            ++output_heads;
            if (tensor.ggml_type != 8U || model.hidden_size % 32U != 0U)
                throw std::invalid_argument("--virtual-q8-lm-head requires an aligned Q8_0 output head");
        }
        if (output_heads != 1U)
            throw std::invalid_argument("--virtual-q8-lm-head requires exactly one output head");
    }
    if (options.virtual_q8_gdn_input) {
        if (model.full_attention_interval == 0U) throw std::invalid_argument("invalid GDN layer interval");
        bool meaningful_q8 = false;
        for (const auto& tensor : manifest.tensors) {
            if (tensor.role != strix::TensorRole::attention_projection || !tensor.layer.has_value() ||
                ((*tensor.layer + 1U) % model.full_attention_interval) == 0U) continue;
            if (tensor.ggml_type == 8U &&
                (tensor.name.ends_with(".attn_qkv.weight") || tensor.name.ends_with(".attn_gate.weight")))
                meaningful_q8 = true;
        }
        if (!meaningful_q8 || model.hidden_size % 32U != 0U)
            throw std::invalid_argument("--virtual-q8-gdn-input requires an aligned Q8_0 GDN input projection");
    }
    if(options.virtual_q8_gdn_output) strix::validate_virtual_q8_gdn_output_manifest(model,manifest);
    const std::string lane_chain_mode_receipt=
        std::string("  \"lane_chain_hyper_output\": ") + (options.lane_chain_hyper_output ? "true" : "false") + ",\n" +
        "  \"lane_chain_hyper_output_eligible_steps\": " + std::to_string(lane_chain_steps.size()) + ",\n" +
        "  \"lane_chain_hyper_output_fallback_steps\": " + std::to_string(options.lane_chain_hyper_output ? lane_chain_total-lane_chain_steps.size() : 0U) + ",\n" +
        (options.prefetch_hc_projection_next_tile ?
            "  \"prefetch_hc_projection_next_tile\": true,\n"
            "  \"hc_projection_next_tile_qualification\": \"not_assessed\",\n" : "");
    const auto qsa_prefetch_plan=options.prefetch_q8_qsa_input ? strix::qsa_input_prefetch::build_plan(model,manifest) : strix::qsa_input_prefetch::Plan{};
    std::string qsa_input_mode_receipt="  \"prefetch_q8_qsa_input\": "+std::string(options.prefetch_q8_qsa_input?"true":"false")+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_eligible_steps\": "+std::to_string(qsa_prefetch_plan.eligible_steps)+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_fallback_steps\": "+std::to_string(qsa_prefetch_plan.fallback_steps)+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_eligible_projections\": "+std::to_string(qsa_prefetch_plan.eligible_projections)+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_fallback_projections\": "+std::to_string(qsa_prefetch_plan.fallback_projections)+",\n";
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="  \"prefetch_q8_qsa_input_projection_choices\": [";
    for(std::size_t i=0;i<qsa_prefetch_plan.projections.size();++i){const auto& choice=qsa_prefetch_plan.projections[i];if(i)qsa_input_mode_receipt+=",";qsa_input_mode_receipt+="{\"layer\":"+std::to_string(choice.layer)+",\"projection\":"+std::to_string(i%5U)+",\"tensor_name\":\""+strix::decode_receipt::escape_json_string(choice.tensor_name)+"\",\"projection_kernel\":\""+std::string(choice.eligible?strix::qsa_input_prefetch::projection_prefetch_id:strix::qsa_input_prefetch::projection_base_id)+"\"}";}
    if(options.prefetch_q8_qsa_input) qsa_input_mode_receipt+="],\n";
    const auto qsa_prefetch_stats=options.prefetch_q8_qsa_input ?
        std::make_shared<strix::qsa_input_prefetch::RuntimeDispatchStats>() : nullptr;
    const auto qsa_gqa_stats = options.qsa_gqa_heads_per_block != 1 ?
        std::make_shared<strix::QsaHierarchicalSelectionStats>() : nullptr;
    const auto qsa_selection_stats = options.qsa_hierarchical_selection ?
        std::make_shared<strix::QsaHierarchicalSelectionStats>() : nullptr;
    const auto shared_prefetch_plan=options.prefetch_q8_shared_expert?strix::shared_expert_prefetch::build_plan(model,manifest):strix::shared_expert_prefetch::Plan{};
    std::string shared_mode_receipt="  \"prefetch_q8_shared_expert\": "+std::string(options.prefetch_q8_shared_expert?"true":"false")+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_eligible_steps\": "+std::to_string(shared_prefetch_plan.eligible_steps)+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_fallback_steps\": "+std::to_string(shared_prefetch_plan.fallback_steps)+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_eligible_projections\": "+std::to_string(shared_prefetch_plan.eligible_projections)+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_fallback_projections\": "+std::to_string(shared_prefetch_plan.fallback_projections)+",\n";
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="  \"prefetch_q8_shared_expert_projection_choices\": [";
    for(std::size_t i=0;i<shared_prefetch_plan.projections.size();++i){const auto& c=shared_prefetch_plan.projections[i];if(i)shared_mode_receipt+=",";shared_mode_receipt+="{\"layer\":"+std::to_string(c.layer)+",\"projection\":"+std::to_string(i%3U)+",\"tensor_name\":\""+strix::decode_receipt::escape_json_string(c.tensor_name)+"\",\"projection_kernel\":\""+std::string(c.eligible?strix::shared_expert_prefetch::projection_prefetch_id:strix::shared_expert_prefetch::projection_base_id)+"\"}";}
    if(options.prefetch_q8_shared_expert) shared_mode_receipt+="],\n";
    const auto shared_prefetch_stats=options.prefetch_q8_shared_expert ?
        std::make_shared<strix::shared_expert_prefetch::RuntimeDispatchStats>() : nullptr;
    // Static mode/plan metadata may be emitted before execution. Dispatch
    // counts must be read separately after the selected workload completes.
    const auto gdn_output_mode_receipt=[&] {
        std::string receipt=(options.virtual_q8_gdn_output ? "  \"virtual_q8_gdn_output\": true,\n" : "")+shared_mode_receipt+qsa_input_mode_receipt+lane_chain_mode_receipt;
        if (options.prefetch_q8_lm_head || options.wave8_q8_lm_head || options.wave8_q8_gdn_input ||
            options.wave8_q8_gdn_output || options.wave8_q8_qsa_output) {
            receipt += "  \"prefetch_q8_lm_head\": " + std::string(options.prefetch_q8_lm_head ? "true" : "false") + ",\n";
            receipt += "  \"wave8_q8_lm_head\": " + std::string(options.wave8_q8_lm_head ? "true" : "false") + ",\n";
            receipt += "  \"wave8_q8_gdn_input\": " + std::string(options.wave8_q8_gdn_input ? "true" : "false") + ",\n";
            receipt += "  \"wave8_q8_gdn_output\": " + std::string(options.wave8_q8_gdn_output ? "true" : "false") + ",\n";
            receipt += "  \"wave8_q8_qsa_output\": " + std::string(options.wave8_q8_qsa_output ? "true" : "false") + ",\n";
            receipt += "  \"q8_projection_organization_qualification\": \"not_assessed\",\n";
        }
        return receipt;
    };
    const std::string deferred_moe_mode_receipt = "  \"deferred_moe_status\": " +
        std::string(options.deferred_moe_status ? "true" : "false") + ",\n";
    const strix::decode_detail::SharedPrefetchDispatchReceipt shared_dispatch_receipt(shared_prefetch_stats);
    const auto original_work_device = std::make_shared<strix::HipSetupQuiescenceGuard>();
    auto target_architecture = original_work_device->architecture();
    if (const auto suffix = target_architecture.find(':'); suffix != std::string::npos)
        target_architecture.resize(suffix);
    if (target_architecture != "gfx1151")
        throw std::invalid_argument("native decoder requires observed gfx1151 without architecture override");

    strix::ExecutionBindingRequest
        binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;
    binding_request.minimum_qualification =
        strix::kernels::
            QualificationStatus::
                experimental;

    const bool graph_mode = options.execution.mode == strix::DecodeExecutionMode::graph;
    binding_request.require_graph_safe = graph_mode;
    binding_request.device_graph_inputs = graph_mode;
    strix::HipLauncherCatalogConfig catalog_config;
    catalog_config.activation = strix::HipActivationStorage::bf16;
    catalog_config.enable_device_graph_inputs = graph_mode;
    catalog_config.parallel_hyper_mix = options.parallel_hyper_mix;
    catalog_config.staged_hyper_projections = options.staged_hyper_projections;
    catalog_config.fused_staged_hyper_mix = options.fused_staged_hyper_mix;
    catalog_config.typed_hyper_projection = options.typed_hyper_projection;
    catalog_config.prefetch_typed_hyper_rows = options.prefetch_typed_hyper_rows;
    catalog_config.prefetch_hc_projection_next_tile = options.prefetch_hc_projection_next_tile;
    binding_request.prefetch_hc_projection_next_tile = options.prefetch_hc_projection_next_tile;
    catalog_config.prefetch_q8_hyper_output = options.prefetch_q8_hyper_output;
    catalog_config.lane_chain_hyper_output=options.lane_chain_hyper_output;
    binding_request.lane_chain_hyper_output=options.lane_chain_hyper_output;
    binding_request.lane_chain_hyper_output_eligible_steps=lane_chain_steps;
    catalog_config.virtual_q8_lm_head = options.virtual_q8_lm_head;
    catalog_config.prefetch_q8_lm_head = options.prefetch_q8_lm_head;
    catalog_config.wave8_q8_lm_head = options.wave8_q8_lm_head;
    catalog_config.wave8_q8_gdn_input = options.wave8_q8_gdn_input;
    catalog_config.gdn_recurrent_layout = binding_request.gdn_recurrent_layout = options.gdn_recurrent_layout;
    catalog_config.wave8_q8_gdn_output = options.wave8_q8_gdn_output;
    catalog_config.wave8_q8_qsa_output = options.wave8_q8_qsa_output;
    binding_request.prefetch_q8_lm_head = options.prefetch_q8_lm_head;
    binding_request.wave8_q8_lm_head = options.wave8_q8_lm_head;
    binding_request.wave8_q8_gdn_input = options.wave8_q8_gdn_input;
    binding_request.wave8_q8_gdn_output = options.wave8_q8_gdn_output;
    binding_request.wave8_q8_qsa_output = options.wave8_q8_qsa_output;
    catalog_config.batched_moe_activation = options.batched_moe_activation;
    binding_request.batched_moe_activation = options.batched_moe_activation;
    catalog_config.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    catalog_config.grouped_exact_q51_down = options.grouped_exact_q51_down;
    catalog_config.q51_slot_parallel_down = options.q51_slot_parallel_down;
    catalog_config.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    catalog_config.qsa_selection_stats = qsa_selection_stats;
    catalog_config.qsa_gqa_heads_per_block = binding_request.qsa_gqa_heads_per_block = options.qsa_gqa_heads_per_block;
    catalog_config.qsa_gqa_stats = qsa_gqa_stats;
    binding_request.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    catalog_config.q4k_a8_stats = q4k_a8_stats;
    binding_request.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    binding_request.grouped_exact_q51_down = options.grouped_exact_q51_down;
    binding_request.q51_slot_parallel_down = options.q51_slot_parallel_down;
    catalog_config.deferred_moe_status = options.deferred_moe_status;
    binding_request.deferred_moe_status = options.deferred_moe_status;
    catalog_config.ranked_moe_router_topk = options.ranked_moe_router_topk;
    binding_request.ranked_moe_router_topk = options.ranked_moe_router_topk;
    catalog_config.wave_moe_router_topk = options.wave_moe_router_topk;
    catalog_config.wave_router_manifest_admission = wave_router_admission;
    binding_request.wave_moe_router_topk = options.wave_moe_router_topk;
    catalog_config.parallel_moe_router = options.parallel_moe_router;
    binding_request.parallel_moe_router = options.parallel_moe_router;
    catalog_config.virtual_q8_gdn_input = options.virtual_q8_gdn_input;
    catalog_config.prefetch_q8_gdn_input = options.prefetch_q8_gdn_input;
    catalog_config.prefetch_q8_qsa_input=options.prefetch_q8_qsa_input;
    catalog_config.qsa_input_prefetch_layers=qsa_prefetch_plan.layers;
    catalog_config.qsa_input_prefetch_stats=qsa_prefetch_stats;
    binding_request.prefetch_q8_qsa_input=options.prefetch_q8_qsa_input;
    binding_request.qsa_input_prefetch_layers=qsa_prefetch_plan.layers;
    catalog_config.prefetch_q8_shared_expert=options.prefetch_q8_shared_expert;
    catalog_config.shared_expert_prefetch_layers=shared_prefetch_plan.layers;
    catalog_config.shared_expert_prefetch_stats=shared_prefetch_stats;
    binding_request.prefetch_q8_shared_expert=options.prefetch_q8_shared_expert;
    binding_request.shared_expert_prefetch_layers=shared_prefetch_plan.layers;
    catalog_config.virtual_q8_gdn_output = options.virtual_q8_gdn_output;
    std::optional<strix::integer_projection::RuntimeConfig> integer_config;
    if (options.execution.integer_head != strix::DecodeIntegerHead::none) {
        integer_config = strix::make_decode_integer_head_config(model, manifest, options.execution,
            {original_work_device->device_index(), original_work_device->observed_binding_identity(), target_architecture});
    }

    strix::kernels::KernelRegistry
        kernel_registry;
    strix::register_complete_hip_kernel_catalog(
        model,
        binding_request,
        catalog_config,
        kernel_registry, integer_config ? &*integer_config : nullptr);

    strix::ExecutionBufferRequest
        buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.parallel_moe_router = options.parallel_moe_router;
    buffer_request.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    buffer_request.grouped_exact_q51_down = options.grouped_exact_q51_down;
    buffer_request.q51_slot_parallel_down = options.q51_slot_parallel_down;
    buffer_request.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    buffer_request.qsa_gqa_heads_per_block = options.qsa_gqa_heads_per_block;
    buffer_request.activation_element_bytes =
        2;
    buffer_request.control_element_bytes =
        4;
    buffer_request.index_element_bytes =
        4;
    buffer_request.logit_element_bytes =
        4;
    buffer_request.mlp_intermediate_element_bytes =
        4;

    auto compiled =
        strix::compile_serial_execution(
            model,
            kernel_registry,
            binding_request,
            buffer_request,
            256);
    std::optional<strix::MoeForkJoinArenaPlan> overlap_arena;
    if (options.moe_shared_overlap) {
        overlap_arena.emplace(strix::prepare_moe_fork_join_arena(
            compiled, options.moe_overlap_arena_budget_bytes));
        compiled = overlap_arena->rebind_execution(compiled);
    }

    std::uint32_t wave_eligible_steps = 0;
    for (const auto& binding : compiled.bindings.bindings)
        if(binding.op == strix::SerialOp::moe_route && binding.kernel_id && binding.kernel_id->ends_with(".wave.parallel"))
            ++wave_eligible_steps;
    if (wave_eligible_steps > compiled.execution.moe_route_steps ||
        (options.wave_moe_router_topk && (!wave_eligible_steps || wave_eligible_steps != compiled.execution.moe_route_steps)))
        throw std::invalid_argument("wave router compiled bindings do not cover every route step");
    const auto wave_fallback_steps = compiled.execution.moe_route_steps - wave_eligible_steps;
    const std::string wave_router_mode_receipt =
        "  \"wave_moe_router_topk\": " + std::string(options.wave_moe_router_topk ? "true" : "false") + ",\n" +
        "  \"wave_moe_router_topk_eligible_steps\": " + std::to_string(wave_eligible_steps) + ",\n" +
        "  \"wave_moe_router_topk_fallback_steps\": " + std::to_string(wave_fallback_steps) + ",\n";
    const auto tensor_access = strix::compile_execution_tensor_access_plan(
        compiled.execution, manifest, tensor_index,
        strix::build_serial_execution_tensor_requirements(compiled.execution));
    strix::qsa_input_prefetch::validate_tensor_access(qsa_prefetch_plan,manifest,compiled.execution,tensor_access);
    strix::shared_expert_prefetch::validate_tensor_access(shared_prefetch_plan,manifest,compiled.execution,tensor_access);
    const auto hc_next_tile_plan = strix::hc_projection_prefetch::build_plan(
        model, manifest, compiled, tensor_access, options.prefetch_hc_projection_next_tile);

    strix::SerialSessionStateLayoutRequest
        state_request;
    state_request.context_tokens =
        options.context_tokens;
    const auto state_layout =
        strix::build_serial_session_state_layout(
            model,
            state_request);


    if (!options.validation_final_state_path.empty())
        strix::validate_final_generation_state_budget(state_layout, {options.validation_final_state_maximum_bytes});
    strix::HipModelResidencyPolicy
        residency_policy;
    residency_policy.mode =
        options.residency;
    residency_policy.coalesce_gap_bytes =
        options.coalesce_gap_bytes;
    // Graph mode explicitly opts into a completely resident model, including
    // its PLE table; registered/file-backed views cannot enter capture.
    residency_policy.promote_file_backed_candidates = graph_mode;

    auto model_owner = std::make_shared<strix::HipModelResidency>(
        strix::HipModelResidency::create(mapped, manifest, generations, residency_policy));
    auto& model_residency = *model_owner;
    strix::q8_projection_organization::validate_resident_weights(q8_organization_plan, model_residency.binding());
    // Admission of actual device views happens before any token can enqueue work.
    for(const auto& projection:qsa_prefetch_plan.projections) {
        if(!projection.eligible) continue;
        const auto found=std::find_if(manifest.tensors.begin(),manifest.tensors.end(),
            [&](const auto& tensor){return tensor.name==projection.tensor_name;});
        if(found==manifest.tensors.end()) throw std::logic_error("QSA admitted tensor disappeared");
        const auto index=static_cast<std::uint32_t>(found-manifest.tensors.begin());
        const auto* runtime=strix::find_runtime_tensor_binding(model_residency.binding(),index);
        const auto bytes=strix::qsa_input_prefetch::required_weight_bytes(found->dimensions[0],found->dimensions[1]);
        if(!runtime || runtime->address==0 || runtime->address%16U!=0 || runtime->bytes<bytes)
            throw std::invalid_argument("QSA prefetch resident weight alignment/extent invalid");
    }
    // Admit all actual eligible device views before the first token can enqueue.
    for(const auto& projection:shared_prefetch_plan.projections) {
        if(!projection.eligible) continue;
        const auto tensor=std::find_if(manifest.tensors.begin(),manifest.tensors.end(),
            [&](const auto& t){return t.name==projection.tensor_name;});
        if(tensor==manifest.tensors.end())throw std::logic_error("shared admitted tensor disappeared");
        const auto index=static_cast<std::uint32_t>(tensor-manifest.tensors.begin());
        const auto* runtime=strix::find_runtime_tensor_binding(model_residency.binding(),index);
        const auto bytes=strix::shared_expert_prefetch::required_weight_bytes(tensor->dimensions[0],tensor->dimensions[1]);
        if(!runtime || runtime->address==0 || runtime->address%16U || runtime->bytes<bytes)
            throw std::invalid_argument("shared prefetch resident weight alignment/extent invalid");
    }


    if (reference.has_value() &&
        reference->model_content_identity !=
            model_residency.binding().
                model_content_identity) {
        throw std::invalid_argument(
            "reference checkpoint SHA-256 does not match loaded model");
    }

    auto allocator_owner = std::make_shared<strix::HipRuntimeAllocator>(
        strix::HipHostAllocationMode::pageable_aligned);
    auto& allocator = *allocator_owner;
    auto memory_owner = std::make_shared<strix::RuntimeExecutionMemoryOwner>(
        strix::RuntimeExecutionMemoryOwner::allocate(compiled, state_layout, allocator, generations));
    auto& memory = *memory_owner;
    strix::q8_projection_organization::validate_output_steps(
        q8_organization_plan, model, compiled, tensor_access, memory.scratch(), model_residency.binding());
    strix::hc_projection_prefetch::validate_runtime(
        hc_next_tile_plan, model, manifest, compiled, tensor_access, memory.scratch(), model_residency.binding());

    strix::initialize_hip_fresh_runtime_memory(
        memory.scratch(),
        memory.state());

    strix::FileMappedPleLookupBackend
        mapped_ple_lookup;
    MappedModelIoSource
        mapped_model_io(
            mapped);
    strix::PlannedPleLookupPolicy
        planned_ple_policy;
    planned_ple_policy.max_rows_per_io =
        options.ple_max_rows_per_io;
    planned_ple_policy.max_inflight_ios =
        options.ple_max_inflight_ios;
    strix::PlannedPleLookupBackend
        planned_ple_lookup(
            mapped_model_io,
            planned_ple_policy);
    strix::PleLookupBackend* ple_lookup =
        options.planned_ple_io
            ? static_cast<strix::PleLookupBackend*>(
                  &planned_ple_lookup)
            : static_cast<strix::PleLookupBackend*>(
                  &mapped_ple_lookup);

    const auto ple_hash_contract =
        strix::build_ple_hash_contract(
            model);
    strix::PleLookupPrefetchState
        ple_prefetch;

    strix::ExecutionRuntimeResources
        resources;
    resources.scratch =
        &memory.scratch();
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &model_residency.binding();
    resources.tensor_access =
        &tensor_access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &memory.state();
    resources.ple_lookup =
        ple_lookup;
    resources.ple_hash_contract =
        &ple_hash_contract;
    resources.ple_prefetch =
        &ple_prefetch;

    auto experiment_owner = std::make_shared<HipDecodeExperimentResources>(
        mapped_owner, model_owner, allocator_owner, memory_owner, manifest, tensor_access, state_layout);
    if (graph_mode) resources = experiment_owner->resources;
    if(wave_router_admission) {
        resources.model_manifest = &wave_router_admission->manifest();
        experiment_owner->resources.model_manifest = &wave_router_admission->manifest();
    }
    std::shared_ptr<strix::integer_projection::HipIntegerRuntimeAllocationOwner> arena_pin, weight_pin;
    std::shared_ptr<strix::integer_projection::HipIntegerProjectionRuntimeBundle> integer_bundle;
    if (integer_config) {
        const auto head = std::find_if(manifest.tensors.begin(), manifest.tensors.end(),
            [](const auto& tensor) { return tensor.role == strix::TensorRole::output_head; });
        if (head == manifest.tensors.end()) throw std::logic_error("integer output head disappeared");
        const auto index = static_cast<std::uint32_t>(head - manifest.tensors.begin());
        const auto* binding = strix::find_runtime_tensor_binding(model_residency.binding(), index);
        if (!binding) throw std::logic_error("integer output head has no actual allocation binding");
        arena_pin = strix::integer_projection::HipIntegerRuntimeAllocationOwner::pin(
            integer_config->device, memory.scratch().arena_base,
            memory.scratch().allocation_generation, experiment_owner);
        weight_pin = strix::integer_projection::HipIntegerRuntimeAllocationOwner::pin(
            integer_config->device, binding->address, binding->allocation_generation, experiment_owner);
        const strix::integer_projection::RuntimeOwners pins{arena_pin, weight_pin};
        const auto plan = strix::integer_projection::RuntimePlan::bind(compiled, resources, *integer_config, pins);
        integer_bundle = std::make_shared<strix::integer_projection::HipIntegerProjectionRuntimeBundle>(
            plan, pins, hipStreamLegacy);
    }
    std::shared_ptr<HipDeferredMoeStatus> deferred_moe;
    if(options.deferred_moe_status) {
        const auto deferred_status_generation=generations.next();
        const auto deferred_shadow_generation=generations.next();
        deferred_moe=std::make_shared<HipDeferredMoeStatus>(resources,compiled.execution_identity,
            deferred_status_generation,deferred_shadow_generation,experiment_owner,original_work_device);
    }
    auto launcher_catalog = std::make_shared<strix::HipLauncherCatalog>(model, catalog_config, integer_bundle);
    std::shared_ptr<strix::HipGdnIslands> gdn_islands;
    if (options.gdn_islands) {
        if (!deferred_moe) throw std::logic_error("GDN islands are missing their D1 owner");
        gdn_islands = strix::HipGdnIslands::create(compiled, launcher_catalog->gdn_config(),
            deferred_moe->resources(), deferred_moe->scope(), model_owner,
            std::static_pointer_cast<const void>(deferred_moe), original_work_device);
    }

    strix::ExecutionLauncherRegistry launchers;
    launcher_catalog->register_for_compiled_execution(compiled, launchers,
        gdn_islands ? &deferred_moe->resources() : &resources, gdn_islands);
    if (launcher_catalog->audit(compiled, launchers).missing_steps)
        throw std::logic_error("HIP launcher catalog is incomplete for actual compiled execution");
    const auto launch_plan = strix::compile_launch_plan(compiled, launchers);
    const auto execution_inventory_receipt = options.execution_inventory
        ? strix::decode_receipt::execution_inventory_receipt(compiled, launch_plan, kernel_registry) +
            strix::decode_receipt::hip_runtime_environment_receipt()
        : std::string{};
    HostTelemetryReport host_telemetry;
    host_telemetry.enabled = options.host_dispatch_telemetry;
    strix::DirectLaunchDispatcher direct_dispatcher(launch_plan,
        host_telemetry.enabled ? &host_telemetry.counters : nullptr);
    std::unique_ptr<strix::HipMoeOverlapDispatcher> moe_overlap;
    if (overlap_arena) {
        if (!deferred_moe) throw std::logic_error("MoE overlap is missing its main stream owner");
        moe_overlap = std::make_unique<strix::HipMoeOverlapDispatcher>(compiled, *overlap_arena,
            launch_plan, deferred_moe->resources(), deferred_moe->scope().stream,
            std::static_pointer_cast<const void>(deferred_moe),
            std::static_pointer_cast<const void>(launcher_catalog), original_work_device,
            host_telemetry.enabled ? &host_telemetry.counters : nullptr);
    }
    strix::EagerExecutionDispatcher& native_dispatcher = moe_overlap
        ? static_cast<strix::EagerExecutionDispatcher&>(*moe_overlap)
        : static_cast<strix::EagerExecutionDispatcher&>(direct_dispatcher);
    std::unique_ptr<strix::GenerationOutputDispatcher> output_dispatcher;
    if (options.gpu_greedy_token)
        output_dispatcher = std::make_unique<strix::GenerationOutputDispatcher>(native_dispatcher, compiled);
    strix::EagerExecutionDispatcher& dispatcher = output_dispatcher
        ? static_cast<strix::EagerExecutionDispatcher&>(*output_dispatcher)
        : native_dispatcher;
    std::unique_ptr<strix::HipSerialGraph> graph;
    if (graph_mode) {
        strix::GraphMemoryRequest graph_memory;
        graph_memory.workspace_bytes_per_instance = options.execution.graph_workspace_bytes;
        graph_memory.executable_bytes_per_instance = options.execution.graph_executable_bytes;
        graph_memory.capture_peak_bytes = options.execution.graph_capture_peak_bytes;
        graph_memory.budget_bytes = options.execution.graph_budget_bytes;
        graph = std::make_unique<strix::HipSerialGraph>(compiled, launcher_catalog, experiment_owner, graph_memory);
    }

    const auto identities =
        strix::build_runtime_execution_identities(
            compiled.execution_identity,
            compiled.arena,
            resources);

    strix::SerialSessionContext
        session;
    session.execution_identity =
        compiled.execution_identity;
    session.state_layout_identity =
        state_layout.layout_identity;
    session.runtime_compatibility_identity =
        identities.compatibility_identity;
    session.state.valid =
        all_persistent_state();

    const auto& lm_io =
        compiled.io.steps.back();
    if (lm_io.op !=
            strix::SerialOp::lm_head ||
        lm_io.writes.size() != 1) {
        throw std::logic_error(
            "compiled execution does not end in one logits write");
    }

    std::unique_ptr<strix::PreparedEagerExecution> prepared_eager;
    if (options.prepared_eager) {
        strix::EagerExecutionRequest prototype;
        prototype.input_token_id = options.tokens.empty() ? 0U : options.tokens.front();
        prototype.position = session.next_position;
        prototype.state = session.state;
        const auto& current_resources = deferred_moe ? deferred_moe->resources() : resources;
        const auto owner = deferred_moe ? std::static_pointer_cast<const void>(deferred_moe)
                                        : std::static_pointer_cast<const void>(experiment_owner);
        prepared_eager = std::make_unique<strix::PreparedEagerExecution>(
            compiled, current_resources, prototype, owner);
    }
    const auto gdn_recurrent_layout_receipt = [&]() -> std::string {
        if (options.gdn_recurrent_layout == strix::HipGdnRecurrentLayout::baseline) return {};
        return "  \"gdn_recurrent_layout\": \"" +
            std::string(strix::gdn_recurrent_layout_marker(options.gdn_recurrent_layout)) + "\",\n"
            "  \"gdn_recurrent_layout_qualification\": \"not_assessed\",\n";
    };
    const auto qsa_gqa_receipt = [&]() -> std::string {
        if (!qsa_gqa_stats) return {};
        std::string result = "  \"qsa_gqa_heads_per_block\": " + std::to_string(options.qsa_gqa_heads_per_block) + ",\n"
            "  \"qsa_gqa_qualification\": \"not_assessed\",\n"
            "  \"qsa_gqa_submissions\": " + std::to_string(qsa_gqa_stats->submissions.load(std::memory_order_relaxed)) + ",\n";
        result += "  \"qsa_gqa_layer_submissions\": [";
        for (std::size_t layer = 0; layer < qsa_gqa_stats->layer_submissions.size(); ++layer) {
            if (layer) result += ',';
            result += std::to_string(qsa_gqa_stats->layer_submissions[layer].load(std::memory_order_relaxed));
        }
        return result + "],\n";
    };
    const auto qsa_selection_receipt = [&]() -> std::string {
        if (!qsa_selection_stats) return {};
        std::string result = "  \"qsa_hierarchical_selection\": true,\n"
            "  \"qsa_hierarchical_selection_qualification\": \"not_assessed\",\n"
            "  \"qsa_hierarchical_selection_submissions\": " +
            std::to_string(qsa_selection_stats->submissions.load(std::memory_order_relaxed)) + ",\n";
        result += "  \"qsa_hierarchical_selection_layer_submissions\": [";
        for (std::size_t layer = 0; layer < qsa_selection_stats->layer_submissions.size(); ++layer) {
            if (layer) result += ',';
            result += std::to_string(qsa_selection_stats->layer_submissions[layer].load(std::memory_order_relaxed));
        }
        return result + "],\n";
    };
    const auto gdn_islands_receipt = [&]() -> std::string {
        if (!gdn_islands) return {};
        const auto observed = gdn_islands->receipt();
        return "  \"gdn_islands\": true,\n  \"gdn_islands_qualification\": \"not_assessed\",\n" +
            std::string("  \"gdn_islands_organization_identity\": ") + std::to_string(observed.organization_identity) + ",\n" +
            "  \"gdn_islands_count\": " + std::to_string(observed.islands) + ",\n" +
            "  \"gdn_islands_kernel_nodes\": " + std::to_string(observed.kernel_nodes) + ",\n" +
            "  \"gdn_islands_setup_memory_observed\": " + (observed.memory_observed ? "true" : "false") + ",\n" +
            "  \"gdn_islands_setup_free_delta_bytes\": " + std::to_string(observed.observed_setup_free_delta_bytes) + ",\n" +
            "  \"gdn_islands_replays\": " + std::to_string(observed.replays) + ",\n";
    };
    const auto overlap_receipt = [&]() -> std::string {
        if (!moe_overlap) return {};
        const auto stats = moe_overlap->stats();
        return "  \"moe_shared_overlap_mode\":\"eager-sibling-streams-experimental-v1\",\n"
            "  \"moe_shared_overlap_layout_identity\":" + std::to_string(overlap_arena->layout_identity()) + ",\n" +
            "  \"moe_shared_overlap_source_execution_identity\":" + std::to_string(overlap_arena->source_execution_identity()) + ",\n" +
            "  \"moe_shared_overlap_arena_bytes\":" + std::to_string(compiled.arena.total_bytes) + ",\n" +
            "  \"moe_shared_overlap_additional_arena_bytes\":" + std::to_string(overlap_arena->additional_bytes()) + ",\n" +
            "  \"moe_shared_overlap_fork_records\":" + std::to_string(stats.fork_records) + ",\n" +
            "  \"moe_shared_overlap_join_waits\":" + std::to_string(stats.join_waits) + ",\n" +
            "  \"moe_shared_overlap_verified_tokens\":" + std::to_string(stats.verified_tokens) + ",\n" +
            "  \"moe_shared_overlap_committed_tokens\":" + std::to_string(stats.committed_tokens) + ",\n" +
            "  \"moe_shared_overlap_committed_layers\":" + std::to_string(stats.committed_layers) + ",\n" +
            "  \"moe_shared_overlap_aborted_tokens\":" + std::to_string(stats.aborted_tokens) + ",\n";
    };
    std::uint64_t input_access_epoch = 1;
    const std::function<void(std::uint32_t)> execute_native_token = [&](std::uint32_t token) {
        original_work_device->require_original();
        if (arena_pin) {
            if (input_access_epoch == UINT64_MAX) throw std::overflow_error("integer input epochs exhausted");
            arena_pin->advance_access(++input_access_epoch);
        }
        if (deferred_moe) {
            const auto committed_before = session.committed_tokens;
            deferred_moe->begin();
            try {
                strix::HipLaunchScope scope(deferred_moe->scope());
                if (moe_overlap) moe_overlap->begin_token(session);
                const auto complete_before_commit = [&] {
                    if (moe_overlap) moe_overlap->verify_before_commit(
                        output_dispatcher && output_dispatcher->state_only_active());
                    deferred_moe->complete_before_commit();
                };
                if (prepared_eager) static_cast<void>(strix::execute_prepared_serial_token(
                    *prepared_eager, session, deferred_moe->resources(), token, dispatcher, complete_before_commit));
                else static_cast<void>(strix::execute_serial_token(compiled, session, deferred_moe->resources(),
                    token, dispatcher, complete_before_commit));
                if (moe_overlap) moe_overlap->mark_committed(session);
            } catch (...) {
                if (moe_overlap) moe_overlap->abort();
                deferred_moe->abort();
                if (session.committed_tokens != committed_before) {
                    session.health = strix::SerialSessionHealth::poisoned;
                    session.arithmetic_lineage.tainted = true;
                }
                throw;
            }
        } else if (graph) static_cast<void>(graph->execute(session, token));
        else if (prepared_eager) static_cast<void>(strix::execute_prepared_serial_token(
            *prepared_eager, session, resources, token, dispatcher));
        else static_cast<void>(strix::execute_serial_token(compiled, session, resources, token, dispatcher));
        // Deferred mode completed its owned stream inside the session's
        // prepublication boundary. Keep the legacy completion path unchanged.
        if (!deferred_moe) {
            check_hip(hipDeviceSynchronize(), "native token original-device completion");
            original_work_device->require_original();
        }
    };

    std::unique_ptr<strix::RawBaselineRecording> raw_recording;
    strix::RawBaselineStateReader raw_state_reader;
    std::function<void()> raw_require_identity = [original_work_device] {
        original_work_device->require_original();
    };
    std::function<void()> raw_drain_or_fail = [original_work_device] {
        original_work_device->recover_or_fail_stop();
    };
    std::string raw_model_sha, raw_executable_sha;
    std::filesystem::path raw_executable;
    std::optional<strix::FinalGenerationStateMetadata> final_state_metadata;
    if (!options.raw_recording_path.empty() || !options.validation_final_state_path.empty()) {
        if (const char* override_arch = std::getenv("HSA_OVERRIDE_GFX_VERSION"); override_arch && *override_arch) throw std::invalid_argument("RAW native target recording rejects architecture override");
#if defined(__linux__)
        raw_executable = "/proc/self/exe";
#else
        raw_executable = std::filesystem::canonical(argv[0]);
#endif
        raw_model_sha = strix::model_content_identity_hex(strix::sha256_file_content(options.model_path));
        raw_executable_sha = strix::model_content_identity_hex(strix::sha256_file_content(raw_executable));
        int device = 0, runtime_version = 0, driver_version = 0;
        hipDeviceProp_t properties{}; hipUUID uuid{};
        check_hip(hipGetDevice(&device), "hipGetDevice(raw recording)");
        check_hip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties(raw recording)");
        check_hip(hipDeviceGetUuid(&uuid, device), "hipDeviceGetUuid(raw recording)");
        check_hip(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion(raw recording)");
        check_hip(hipDriverGetVersion(&driver_version), "hipDriverGetVersion(raw recording)");
        if (runtime_version <= 0 || driver_version <= 0) throw std::runtime_error("RAW HIP runtime/driver version unavailable");
        constexpr char hex[] = "0123456789abcdef"; std::string gpu_uuid;
        for (const auto byte : uuid.bytes) { const auto value = static_cast<unsigned char>(byte); gpu_uuid += hex[value >> 4]; gpu_uuid += hex[value & 15]; }
        const auto expected_state_owner = memory.state();
        const std::string expected_architecture = properties.gcnArchName;
        raw_require_identity = [&, device, runtime_version, driver_version, gpu_uuid, expected_architecture, expected_state_owner] {
            int actual_device = 0, actual_runtime = 0, actual_driver = 0; hipUUID actual_uuid{}; hipDeviceProp_t actual_properties{};
            check_hip(hipGetDevice(&actual_device), "hipGetDevice(raw binding)");
            check_hip(hipRuntimeGetVersion(&actual_runtime), "hipRuntimeGetVersion(raw binding)");
            check_hip(hipDriverGetVersion(&actual_driver), "hipDriverGetVersion(raw binding)");
            check_hip(hipDeviceGetUuid(&actual_uuid, actual_device), "hipDeviceGetUuid(raw binding)");
            check_hip(hipGetDeviceProperties(&actual_properties, actual_device), "hipGetDeviceProperties(raw binding)");
            std::string actual_hex; constexpr char digits[] = "0123456789abcdef";
            for (auto byte : actual_uuid.bytes) { const auto value = static_cast<unsigned char>(byte); actual_hex += digits[value >> 4]; actual_hex += digits[value & 15]; }
            const auto& actual_owner = memory.state();
            if (actual_device != device || actual_runtime != runtime_version || actual_driver != driver_version || actual_hex != gpu_uuid ||
                std::string(actual_properties.gcnArchName) != expected_architecture || actual_owner.layout_identity != expected_state_owner.layout_identity ||
                actual_owner.device_base != expected_state_owner.device_base || actual_owner.device_bytes != expected_state_owner.device_bytes ||
                actual_owner.device_allocation_generation != expected_state_owner.device_allocation_generation || actual_owner.host_base != expected_state_owner.host_base ||
                actual_owner.host_bytes != expected_state_owner.host_bytes || actual_owner.host_allocation_generation != expected_state_owner.host_allocation_generation)
                throw std::runtime_error("RAW native device/runtime/state owner identity changed");
            if (const char* override_arch = std::getenv("HSA_OVERRIDE_GFX_VERSION"); override_arch && *override_arch) throw std::runtime_error("RAW architecture override changed");
        };
        raw_drain_or_fail = [device, require_original_binding = raw_require_identity] {
            try {
                check_hip(hipSetDevice(device), "hipSetDevice(raw exact drain)");
                require_original_binding();
                check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(raw exact drain)");
                require_original_binding();
            } catch (...) {
                std::fputs("RAW capture failed with unproved original GPU quiescence; fail-stop, no finalized recording\n", stderr);
                std::fflush(nullptr); std::_Exit(3);
            }
        };
        strix::RawBaselineMetadata metadata;
        metadata.model_sha256 = raw_model_sha;
        metadata.model_content_identity = strix::model_content_identity_hex(model_residency.binding().model_content_identity);
        metadata.executable_sha256 = raw_executable_sha;
        metadata.gpu_identity = gpu_uuid; metadata.gpu_architecture = properties.gcnArchName;
        metadata.hip_runtime_version = static_cast<std::uint32_t>(runtime_version); metadata.hip_driver_version = static_cast<std::uint32_t>(driver_version);
        metadata.execution_identity = compiled.execution_identity; metadata.kernel_plan_identity = compiled.kernel_plan_identity;
        metadata.runtime_compatibility_identity = identities.compatibility_identity; metadata.vocabulary_size = model.vocab_size;
        metadata.supplied_tokens = options.tokens; metadata.maximum_generated_tokens = options.generate_tokens;
        metadata.mode = options.generate_tokens ? strix::RawBaselineMode::greedy : strix::RawBaselineMode::teacher_forced;
        if (options.generate_tokens) metadata.eos_token = generation_eos;
        metadata.evidence = graph_mode ? strix::RawBaselineEvidence::native_graph : integer_config
            ? strix::RawBaselineEvidence::native_integer_eager : strix::RawBaselineEvidence::native_eager;
        if (buffer_request.activation_element_bytes != 2) throw std::logic_error("RAW BF16 catalog/storage binding changed");
        metadata.activation_storage = strix::RawBaselineActivationStorage::bf16;
        metadata.execution_mode = graph_mode ? strix::RawBaselineExecutionMode::graph_float : integer_config
            ? strix::RawBaselineExecutionMode::eager_integer_lm_head : strix::RawBaselineExecutionMode::eager_float;
        metadata.numerical_identity = integer_config ? integer_config->approved_numerical_identity
            : strix::raw_floating_lm_head_numerical_identity(metadata.activation_storage);
        if (options.q4k_a8g32_gate_up)
            metadata.numerical_identity += ";" + std::string(strix::moe_q4k_a8::runtime_numerical_identity);
        if(options.grouped_exact_q51_down)
            metadata.numerical_identity += ";" + std::string(strix::moe_q4k_a8::grouped_q51_raw_identity);
        if (catalog_config.splitk_hyper_projection)
            metadata.numerical_identity += ";" + std::string(catalog_config.splitk_hyper_projection_fp64
                ? strix::splitk_hc_fp64_numerical_identity : strix::splitk_hc_fp32_numerical_identity);
        strix::RawBaselinePolicy policy;
        policy.maximum_bytes = options.raw_maximum_bytes;
        policy.maximum_steps = options.tokens.size();
        if (options.generate_tokens) policy.maximum_steps = options.generate_tokens - 1 >= options.context_tokens - options.tokens.size()
            ? options.context_tokens : options.tokens.size() + options.generate_tokens - 1;
        policy.state_policy = options.raw_state_policy;
        policy.allow_prefix_restore = options.prefix_cache_roundtrip;
        raw_state_reader = [&](const strix::SessionStateAllocation& allocation, std::uint64_t offset, std::span<std::byte> destination) {
            raw_require_identity();
            const auto address = strix::resolve_session_state(state_layout, memory.state(), allocation.component);
            if (offset > address.bytes || destination.size() > address.bytes - offset || address.address > std::numeric_limits<std::uintptr_t>::max() - offset)
                throw std::out_of_range("RAW copy exceeds actual state allocation");
            const auto* source = reinterpret_cast<const void*>(address.address + offset);
            if (address.storage == strix::SessionStateStorage::device)
                check_raw_copy(hipMemcpy(destination.data(), source, destination.size(), hipMemcpyDeviceToHost), "hipMemcpy(raw state D2H)", raw_drain_or_fail);
            else std::memcpy(destination.data(), source, destination.size());
            raw_require_identity();
        };
        if (!options.validation_final_state_path.empty()) {
            final_state_metadata.emplace();
            auto& final = *final_state_metadata;
            final.model_sha256 = metadata.model_sha256;
            final.model_content_identity = metadata.model_content_identity;
            final.executable_sha256 = metadata.executable_sha256;
            final.numerical_identity = metadata.numerical_identity;
            final.output_mode = options.gpu_greedy_token ? std::string(strix::gpu_greedy_output_identity) : "full-logits-cpu-firstmax";
            final.gpu_identity = metadata.gpu_identity;
            final.gpu_architecture = metadata.gpu_architecture;
            final.hip_runtime_version = metadata.hip_runtime_version;
            final.hip_driver_version = metadata.hip_driver_version;
            final.kernel_plan_identity = compiled.kernel_plan_identity;
            final.launch_plan_identity = launch_plan.plan_identity;
            final.runtime_binding_identity = identities.binding_identity;
            final.prepared_eager = options.prepared_eager;
            final.shared_overlap = options.moe_shared_overlap;
            final.batched_moe_activation = options.batched_moe_activation;
        }
        if (!options.raw_recording_path.empty()) {
            raw_recording = std::make_unique<strix::RawBaselineRecording>(options.raw_recording_path, std::move(metadata), policy, state_layout);
            raw_recording->capture_initial(session, raw_state_reader);
        }
    }
    const auto finalize_raw = [&](std::span<const std::uint32_t> generated = {}, std::string stop = "raw_input_exhausted") {
        if (!raw_recording) return;
        raw_recording->finalize(session, raw_state_reader, generated, std::move(stop), [&] {
            raw_require_identity();
            if (strix::model_content_identity_hex(strix::sha256_file_content(options.model_path)) != raw_model_sha ||
                strix::model_content_identity_hex(strix::sha256_file_content(raw_executable)) != raw_executable_sha)
                throw std::runtime_error("RAW model/executable identity changed during capture");
            raw_require_identity();
        });
    };

    if (options.generate_tokens != 0) {
        if (options.prefix_cache_roundtrip) {
            HipSerialGreedyStepBackend
                prefill_backend(
                    compiled,
                    session,
                    resources,
                    dispatcher,
                    memory.scratch(),
                    lm_io,
                    model.vocab_size,
                    [&](std::uint32_t token, std::uint64_t position, std::span<const float> values) {
                        if (raw_recording) raw_recording->record_token(token, position, values, session, raw_state_reader);
                    }, raw_require_identity, raw_drain_or_fail, execute_native_token, false, options.deferred_moe_status);

            std::vector<float> prefix_logits;
            for (std::size_t index = 0;
                 index < options.tokens.size();
                 ++index) {
                prefix_logits =
                    prefill_backend.execute(
                        options.tokens[index],
                        index);
            }

            GuardedHipSnapshotTransfer snapshot_transfer(raw_require_identity, raw_drain_or_fail);
            auto prefix_cache =
                strix::SerialPrefixCacheEntry::capture(
                    model,
                    options.tokens,
                    session,
                    state_layout,
                    memory.state(),
                    allocator,
                    generations,
                    snapshot_transfer);
            check_raw_copy(hipDeviceSynchronize(), "hipDeviceSynchronize(prefix capture)", raw_drain_or_fail);
            raw_require_identity();
            if (raw_recording) {
                raw_recording->capture_prefix_restore_source(prefix_cache.prefix(), session, raw_state_reader);
            }

            strix::SerialGreedyContinuationRequest
                continuation;
            continuation.initial_logits =
                prefix_logits;
            continuation.start_position =
                session.next_position;
            continuation.vocab_size =
                model.vocab_size;
            continuation.context_capacity =
                options.context_tokens;
            continuation.max_new_tokens =
                options.generate_tokens;
            continuation.eos_token_id =
                generation_eos;

            HipSerialGreedyStepBackend
                baseline_backend(
                    compiled,
                    session,
                    resources,
                    dispatcher,
                    memory.scratch(),
                    lm_io,
                    model.vocab_size, {}, raw_require_identity, raw_drain_or_fail, execute_native_token, true, options.deferred_moe_status);
            const auto baseline =
                strix::execute_serial_greedy_continuation(
                    continuation,
                    baseline_backend);

            const auto mutated_position =
                session.next_position;
            if (mutated_position <= options.tokens.size())
                throw std::runtime_error("prefix-cache roundtrip exercised no mutation before restore");
            const auto restore_start =
                std::chrono::steady_clock::now();
            prefix_cache.restore(
                session,
                model,
                state_layout,
                memory.state(),
                snapshot_transfer,
                options.tokens);
            check_raw_copy(
                hipDeviceSynchronize(),
                "hipDeviceSynchronize(prefix restore)", raw_drain_or_fail);
            const auto restore_stop =
                std::chrono::steady_clock::now();

            if (session.next_position !=
                    options.tokens.size() ||
                session.committed_tokens !=
                    options.tokens.size() ||
                session.health !=
                    strix::SerialSessionHealth::ready) {
                throw std::logic_error(
                    "prefix cache restore did not recover exact session metadata");
            }
            if (raw_recording) raw_recording->record_prefix_restoration(prefix_cache.prefix(), session, raw_state_reader, mutated_position);

            HipSerialGreedyStepBackend
                restored_backend(
                    compiled,
                    session,
                    resources,
                    dispatcher,
                    memory.scratch(),
                    lm_io,
                    model.vocab_size,
                    [&](std::uint32_t token, std::uint64_t position, std::span<const float> values) {
                        if (raw_recording) raw_recording->record_token(token, position, values, session, raw_state_reader);
                    }, raw_require_identity, raw_drain_or_fail, execute_native_token, true, options.deferred_moe_status);
            const auto restored =
                strix::execute_serial_greedy_continuation(
                    continuation,
                    restored_backend);

            const bool equivalent =
                baseline.generated_tokens ==
                    restored.generated_tokens &&
                baseline.stop_reason ==
                    restored.stop_reason &&
                baseline.final_position ==
                    restored.final_position &&
                baseline_backend.logit_hashes() == restored_backend.logit_hashes();

            if (!equivalent) {
                throw std::runtime_error(
                    "prefix cache round-trip changed complete logits or greedy continuation");
            }
            finalize_raw(restored.generated_tokens, std::string(stop_reason_name(restored.stop_reason)));

            double prefill_ms = 0.0;
            for (const auto value :
                 prefill_backend.elapsed_ms()) {
                prefill_ms += value;
            }
            double baseline_ms = 0.0;
            for (const auto value :
                 baseline_backend.elapsed_ms()) {
                baseline_ms += value;
            }
            double restored_ms = 0.0;
            for (const auto value :
                 restored_backend.elapsed_ms()) {
                restored_ms += value;
            }
            const auto restore_ms =
                std::chrono::duration<double, std::milli>(
                    restore_stop -
                    restore_start).count();

            std::cout << "{\n";
            emit_q51_slot_receipt();
            emit_q51_slot_counts();
            std::cout
                << optimization_profile_receipt << eager_submission_receipt << overlap_receipt() << execution_inventory_receipt << gdn_output_mode_receipt() << wave_router_mode_receipt << deferred_moe_mode_receipt
                << strix::decode_detail::qsa_input_dispatch_receipt(qsa_prefetch_stats.get())
                << shared_dispatch_receipt.completed_counts()
                << qsa_gqa_receipt() << gdn_recurrent_layout_receipt() << qsa_selection_receipt() << gdn_islands_receipt() << strix::decode_detail::deferred_moe_completed_tokens_receipt(deferred_moe.get())
                << "  \"model_content_identity\": \""
                << strix::model_content_identity_hex(
                       model_residency.binding().
                           model_content_identity)
                << "\",\n"
                << "  \"execution_identity\": "
                << compiled.execution_identity
                << ",\n"
                << "  \"runtime_compatibility_identity\": "
                << identities.compatibility_identity
                << ",\n"
                << "  \"prefix_identity\": "
                << prefix_cache.prefix().
                       prefix_identity
                << ",\n"
                << "  \"prefix_tokens\": [";

            for (std::size_t index = 0;
                 index < options.tokens.size();
                 ++index) {
                if (index != 0) {
                    std::cout << ',';
                }
                std::cout
                    << options.tokens[index];
            }

            std::cout
                << "],\n"
                << "  \"baseline_generated_tokens\": [";
            for (std::size_t index = 0;
                 index <
                    baseline.generated_tokens.size();
                 ++index) {
                if (index != 0) {
                    std::cout << ',';
                }
                std::cout
                    << baseline.generated_tokens[index];
            }
            std::cout
                << "],\n"
                << "  \"restored_generated_tokens\": [";
            for (std::size_t index = 0;
                 index <
                    restored.generated_tokens.size();
                 ++index) {
                if (index != 0) {
                    std::cout << ',';
                }
                std::cout
                    << restored.generated_tokens[index];
            }

            std::cout
                << "],\n"
                << "  \"continuation_equivalent\": true,\n"
                << "  \"equivalence_scope\": \"tokens_stop_position_and_complete_continuation_logits\",\n"
                << "  \"baseline_continuation_logits_sha256\": ";
            write_logit_hashes(baseline_backend.logit_hashes());
            std::cout << ",\n  \"restored_continuation_logits_sha256\": ";
            write_logit_hashes(restored_backend.logit_hashes());
            std::cout << ",\n"
                << "  \"mutated_position_before_restore\": "
                << mutated_position
                << ",\n"
                << "  \"restored_prefix_position\": "
                << options.tokens.size()
                << ",\n"
                << "  \"stop_reason\": \""
                << stop_reason_name(
                       restored.stop_reason)
                << "\",\n"
                << "  \"prefill_elapsed_ms\": "
                << prefill_ms
                << ",\n"
                << "  \"baseline_continuation_elapsed_ms\": "
                << baseline_ms
                << ",\n"
                << "  \"restore_elapsed_ms\": "
                << restore_ms
                << ",\n"
                << "  \"restored_continuation_elapsed_ms\": "
                << restored_ms
                << "\n"
                << "}\n";
            return 0;
        }

        std::unique_ptr<strix::HipGreedyOutput> greedy_output;
        if (options.gpu_greedy_token) {
            if (model.vocab_size > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("GPU greedy vocabulary exceeds uint32");
            greedy_output = std::make_unique<strix::HipGreedyOutput>(
                static_cast<std::uint32_t>(model.vocab_size), memory.scratch(),
                std::static_pointer_cast<const void>(memory_owner),
                original_work_device);
        }
        std::function<void(std::uint32_t, std::uint64_t, std::span<const float>)> greedy_recording;
        if (raw_recording) greedy_recording = [&](std::uint32_t token, std::uint64_t position,
                                                   std::span<const float> values) {
            raw_recording->record_token(token, position, values, session, raw_state_reader);
        };
        HipSerialGreedyStepBackend backend(
            compiled,
            session,
            resources,
            dispatcher,
            memory.scratch(),
            lm_io,
            model.vocab_size,
            std::move(greedy_recording), raw_require_identity, raw_drain_or_fail,
            execute_native_token, false, options.deferred_moe_status, greedy_output.get(), output_dispatcher.get());

        strix::SerialGreedyRequest request;
        request.prefix_tokens =
            options.tokens;
        request.vocab_size =
            model.vocab_size;
        request.context_capacity =
            options.context_tokens;
        request.max_new_tokens =
            options.generate_tokens;
        request.eos_token_id =
            generation_eos;

        std::optional<strix::CompiledKnownPrefixWindow> known_prefix_plan;
        std::shared_ptr<KnownPromptMemory> known_prefix_memory;
        std::shared_ptr<strix::HipKnownPrefixWindow> known_prefix_owner;
        std::unique_ptr<strix::HipKnownPromptBackend> known_prefix_backend;
        std::optional<strix::KnownPromptGreedyResult> known_prefix_result;
        if (options.known_prefix_window) {
            known_prefix_plan = strix::compile_known_prefix_window(model, compiled, manifest, tensor_access, {true, true, true});
            if (!known_prefix_plan->r2_steps) throw std::invalid_argument("known prompt has no admitted paired operators");
            if (model.vocab_size > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("known-prompt vocabulary exceeds uint32");
            const auto arena_generation = generations.next();
            const auto status_generation = generations.next();
            known_prefix_memory = std::make_shared<KnownPromptMemory>(*known_prefix_plan, arena_generation,
                status_generation, options.context_tokens, static_cast<std::uint32_t>(model.vocab_size),
                std::static_pointer_cast<const void>(experiment_owner), original_work_device);
            known_prefix_owner = std::make_shared<strix::HipKnownPrefixWindow>(*known_prefix_plan, compiled, model,
                catalog_config, resources, known_prefix_memory->scratch, known_prefix_memory->scope,
                std::static_pointer_cast<const void>(known_prefix_memory));
            known_prefix_backend = std::make_unique<strix::HipKnownPromptBackend>(known_prefix_owner, session,
                known_prefix_memory->scratch, model.vocab_size, original_work_device);
        }

        const bool generation_request_timing_enabled =
            !raw_recording && !final_state_metadata;
        const auto generation_request_start = generation_request_timing_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        const auto result = [&]() -> strix::SerialGreedyRequestResult {
            if (!known_prefix_backend) return options.gpu_greedy_token
                ? strix::execute_serial_greedy_token_request(request, backend)
                : strix::execute_serial_greedy_request(request, backend);
            strix::KnownPromptGreedyRequest paired;
            paired.prompt_tokens = options.tokens;
            paired.start_position = session.next_position;
            paired.vocab_size = request.vocab_size; paired.context_capacity = request.context_capacity;
            paired.max_new_tokens = request.max_new_tokens; paired.eos_token_id = request.eos_token_id;
            known_prefix_result = strix::execute_known_prompt_greedy_request(paired, *known_prefix_backend, backend);
            if (!known_prefix_result->generation) throw std::logic_error("known prompt unexpectedly stopped before generation");
            auto total = *known_prefix_result->generation;
            if (known_prefix_result->consumed_prompt_tokens > std::numeric_limits<std::uint64_t>::max() - total.steps_executed)
                throw std::overflow_error("known-prompt total execution count overflow");
            total.steps_executed += known_prefix_result->consumed_prompt_tokens;
            return total;
        }();
        const auto generation_request_stop = generation_request_timing_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        const auto generation_request_elapsed_ms =
            std::chrono::duration<double, std::milli>(
                generation_request_stop - generation_request_start).count();
        finalize_raw(result.generated_tokens, std::string(stop_reason_name(result.stop_reason)));

        double elapsed_total = 0.0;
        for (const auto value :
             backend.elapsed_ms()) {
            elapsed_total += value;
        }

        // Validation-only export is after the request and its per-step timing
        // collection. Never enable this for wall-clock throughput measurements.
        if (final_state_metadata) {
            strix::export_final_generation_state(options.validation_final_state_path,
                {options.validation_final_state_maximum_bytes}, model, state_layout, memory.state(),
                session, request, result, *final_state_metadata,
                std::static_pointer_cast<const void>(memory_owner), raw_state_reader,
                raw_require_identity, raw_drain_or_fail, [&] {
                    raw_require_identity();
                    if (strix::model_content_identity_hex(strix::sha256_file_content(options.model_path)) != raw_model_sha ||
                        strix::model_content_identity_hex(strix::sha256_file_content(raw_executable)) != raw_executable_sha)
                        throw std::runtime_error("final state model/executable identity changed during validation");
                    raw_require_identity();
                });
        }
        std::cout << "{\n" << optimization_profile_receipt << eager_submission_receipt << overlap_receipt() << execution_inventory_receipt << gdn_output_mode_receipt() << wave_router_mode_receipt << deferred_moe_mode_receipt;
        if (known_prefix_result) {
            const auto observed = known_prefix_backend->stats();
            std::array<std::uint64_t, strix::known_window_r2_variant_slots> compiled_variants{};
            for (const auto variant : known_prefix_plan->r2_variant) {
                const auto index = static_cast<std::uint32_t>(variant);
                if (index >= compiled_variants.size()) throw std::logic_error("unknown prompt variant in source receipt");
                if (index) ++compiled_variants[index];
            }
            std::cout << "  \"known_prefix_completed_windows\":" << observed.windows << ",\n"
                << "  \"known_prefix_stats_saturated\":" << (observed.saturated ? "true" : "false") << ",\n"
                << "  \"known_prefix_r2_dispatches\":" << observed.r2_dispatches << ",\n"
                << "  \"known_prefix_row_loop_dispatches\":" << observed.row_loop_dispatches << ",\n"
                << "  \"known_prefix_logical_projection_weight_bytes_avoided\":" << observed.logical_projection_weight_bytes_avoided << ",\n"
                << "  \"known_prefix_projection_launches_saved\":" << observed.projection_launches_saved << ",\n"
                << "  \"known_prefix_r2_variant_dispatches\":[";
            for (std::size_t i = 0; i < observed.r2_variant_dispatches.size(); ++i)
                std::cout << (i ? "," : "") << observed.r2_variant_dispatches[i];
            std::cout << "],\n  \"known_prefix_source_variant_steps\":[";
            for (std::size_t i = 0; i < compiled_variants.size(); ++i) std::cout << (i ? "," : "") << compiled_variants[i];
            std::cout << "],\n  \"known_prefix_window\":true,\n"
                << "  \"known_prefix_window_qualification\":\"not_assessed\",\n"
                << "  \"known_prefix_consumed_prompt_tokens\":" << known_prefix_result->consumed_prompt_tokens << ",\n"
                << "  \"known_prefix_windows_executed\":" << known_prefix_result->windows_executed << ",\n"
                << "  \"known_prefix_serial_prompt_steps\":" << known_prefix_result->serial_prompt_steps << ",\n"
                << "  \"known_prefix_window_execution_identity\":" << known_prefix_plan->window_execution_identity << ",\n"
                << "  \"known_prefix_window_kernel_plan_identity\":" << known_prefix_plan->window_kernel_plan_identity << ",\n"
                << "  \"known_prefix_source_r2_steps\":" << known_prefix_plan->r2_steps << ",\n"
                << "  \"known_prefix_source_projection_launches_saved_per_window\":" << known_prefix_plan->projection_launches_saved << ",\n"
                << "  \"known_prefix_source_logical_weight_bytes_avoided_per_window\":" << known_prefix_plan->logical_projection_weight_bytes_avoided << ",\n";
        }
        if (generation_request_timing_enabled)
            std::cout << "  \"generation_request_elapsed_ms\":" << generation_request_elapsed_ms << ",\n"
                      << "  \"generation_request_output_tokens\":" << result.generated_tokens.size() << ",\n"
                      << "  \"generation_request_output_tokens_per_second\":"
                      << (generation_request_elapsed_ms > 0.0
                              ? static_cast<double>(result.generated_tokens.size()) * 1000.0 /
                                    generation_request_elapsed_ms
                              : 0.0) << ",\n"
                      << "  \"generation_request_timing_scope\":\"full-request-including-prefix-execution-readback-token-selection-and-controller\",\n";
        if (final_state_metadata)
            std::cout << "  \"validation_final_state_timing_invalid\":true,\n"
                      << "  \"validation_final_state_manifest\":\""
                      << strix::decode_receipt::escape_json_string(
                             (std::filesystem::path(options.validation_final_state_path) / "manifest.json").string())
                      << "\",\n";
        if (greedy_output) {
            const auto stats = greedy_output->stats();
            double prefix_state_total = 0.0;
            for (const auto duration : backend.prefix_state_elapsed_ms()) prefix_state_total += duration;
            std::cout << "  \"generation_output_mode\":\"" << strix::gpu_greedy_output_identity << "\",\n"
                << "  \"prefix_state_only_steps\":" << backend.prefix_state_elapsed_ms().size() << ",\n"
                << "  \"prefix_state_only_elapsed_ms_total\":" << prefix_state_total << ",\n"
                << "  \"prefix_state_only_elapsed_ms_per_step\":"
                << (backend.prefix_state_elapsed_ms().empty() ? 0.0 : prefix_state_total /
                    static_cast<double>(backend.prefix_state_elapsed_ms().size())) << ",\n"
                << "  \"generation_output_completed\":" << stats.completed << ",\n"
                << "  \"generation_output_vocabulary_size\":" << model.vocab_size << ",\n"
                << "  \"generation_output_calls\":" << stats.calls << ",\n"
                << "  \"generation_output_failures\":" << stats.failures << ",\n"
                << "  \"generation_output_kernel_submissions\":" << stats.reduction_kernel_submissions << ",\n"
                << "  \"generation_output_poison_h2d_bytes\":" << stats.poison_h2d_bytes << ",\n"
                << "  \"generation_output_stream_completions\":" << stats.stream_completions << ",\n"
                << "  \"generation_output_full_logits_d2h_bytes\":0,\n"
                << "  \"generation_output_result_d2h_bytes\":" << stats.result_d2h_bytes << ",\n"
                << "  \"generation_output_pool_requested_bytes\":" << greedy_output->scratch_bytes() << ",\n"
                << "  \"prefill_lm_heads_omitted\":" << output_dispatcher->omitted_heads() << ",\n"
                << "  \"generation_output_timing_scope\":\"full-output-steps-producer-through-selection-and-readback\",\n";
        }
        emit_q4k_a8_receipt();
        emit_q4k_a8_counts();
        std::cout << strix::decode_detail::qsa_input_dispatch_receipt(qsa_prefetch_stats.get());
        std::cout << shared_dispatch_receipt.completed_counts();
        std::cout << qsa_gqa_receipt() << gdn_recurrent_layout_receipt() << qsa_selection_receipt() << gdn_islands_receipt() << strix::decode_detail::deferred_moe_completed_tokens_receipt(deferred_moe.get());
        std::cout
            << "  \"model_content_identity\": \""
            << strix::model_content_identity_hex(
                   model_residency.binding().
                       model_content_identity)
            << "\",\n"
            << "  \"execution_identity\": "
            << compiled.execution_identity
            << ",\n"
            << "  \"runtime_compatibility_identity\": "
            << identities.compatibility_identity
            << ",\n"
            << "  \"ple_lookup\": \""
            << (options.planned_ple_io
                    ? "planned-pread"
                    : "mapped")
            << "\",\n"
            << "  \"prefix_tokens\": [";

        for (std::size_t index = 0;
             index < options.tokens.size();
             ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << options.tokens[index];
        }

        std::cout
            << "],\n"
            << "  \"generated_tokens\": [";

        for (std::size_t index = 0;
             index <
                result.generated_tokens.size();
             ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout
                << result.generated_tokens[index];
        }

        std::cout
            << "],\n"
            << "  \"stop_reason\": \""
            << stop_reason_name(
                   result.stop_reason)
            << "\",\n"
            << "  \"steps_executed\": "
            << result.steps_executed
            << ",\n"
            << "  \"final_position\": "
            << result.final_position
            << ",\n"
            << (known_prefix_backend ? "  \"known_prefix_serial_tail_and_generation_elapsed_ms_total\": " :
                greedy_output ? "  \"generation_step_elapsed_ms_total\": " : "  \"decode_elapsed_ms_total\": ")
            << elapsed_total
            << ",\n"
            << (known_prefix_backend ? "  \"known_prefix_serial_tail_and_generation_elapsed_ms_per_step\": " :
                greedy_output ? "  \"generation_step_elapsed_ms_per_step\": " : "  \"decode_elapsed_ms_per_step\": ")
            << (backend.elapsed_ms().empty()
                    ? 0.0
                    : elapsed_total /
                          static_cast<double>(
                              backend.elapsed_ms().size()))
            << "\n"
            << "}\n";
        return 0;
    }

    std::vector<float> logits(
        static_cast<std::size_t>(
            model.vocab_size));
    std::vector<double> token_elapsed_ms;
    token_elapsed_ms.reserve(
        options.tokens.size());
    double decode_elapsed_ms_total = 0.0;

    strix::SerialDecodeReplayObservation
        replay_observation;
    if (reference.has_value()) {
        replay_observation.model_content_identity =
            model_residency.binding().
                model_content_identity;
        replay_observation.compiled_execution_identity =
            compiled.execution_identity;
        replay_observation.runtime_compatibility_identity =
            identities.compatibility_identity;
        replay_observation.independent_reference_identity =
            reference->independent_reference_identity;
        replay_observation.input_tokens =
            options.tokens;
        replay_observation.reference_argmax_tokens.reserve(
            reference->steps.size());
        replay_observation.engine_argmax_tokens.reserve(
            reference->steps.size());
    }

    std::cout << '{' << '\n' << optimization_profile_receipt << eager_submission_receipt << execution_inventory_receipt << gdn_output_mode_receipt() << wave_router_mode_receipt << deferred_moe_mode_receipt;
    emit_q4k_a8_receipt();
    std::cout << "  \"model_content_identity\": \""
              << strix::model_content_identity_hex(
                     model_residency.binding().model_content_identity)
              << "\"," << '\n'
              << "  \"execution_identity\": "
              << compiled.execution_identity
              << ',' << '\n'
              << "  \"ranked_moe_router_topk\": " << (options.ranked_moe_router_topk ? "true" : "false") << ",\n"
              << "  \"parallel_moe_router\": " << (options.parallel_moe_router ? "true" : "false") << ",\n"
              << "  \"launch_plan_identity\": "
              << launch_plan.plan_identity
              << ',' << '\n'
              << "  \"runtime_compatibility_identity\": "
              << identities.compatibility_identity
              << ',' << '\n'
              << "  \"context_tokens\": "
              << options.context_tokens
              << ',' << '\n'
              << "  \"residency\": \""
              << (options.residency ==
                          strix::HipModelResidencyMode::
                              registered_file_mapping
                      ? "register"
                      : "copy")
              << "\"," << '\n'
              << "  \"registered_bytes\": "
              << model_residency.registered_bytes()
              << ',' << '\n'
              << "  \"copied_bytes\": "
              << model_residency.copied_bytes()
              << ',' << '\n'
              << "  \"ple_lookup\": \""
              << (options.planned_ple_io
                      ? "planned-pread"
                      : "mapped")
              << "\"," << '\n'
              << "  \"ple_max_rows_per_io\": "
              << options.ple_max_rows_per_io
              << ',' << '\n'
              << "  \"ple_max_inflight_ios\": "
              << options.ple_max_inflight_ios
              << ',' << '\n'
              << "  \"ple_lookup_identity\": "
              << ple_lookup->implementation_identity()
              << ',' << '\n'
              << "  \"tokens\": [" << '\n';

    for (std::size_t input_index = 0;
         input_index <
            options.tokens.size();
         ++input_index) {
        const auto token =
            options.tokens[input_index];

        const auto token_start =
            std::chrono::steady_clock::now();

        try {
        execute_native_token(token);

        if (!deferred_moe) check_hip(
            hipDeviceSynchronize(),
            "hipDeviceSynchronize(serial token)");
        } catch (...) { if (raw_drain_or_fail) raw_drain_or_fail(); throw; }

        const auto token_stop =
            std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(
                token_stop - token_start).count();
        token_elapsed_ms.push_back(
            elapsed_ms);
        decode_elapsed_ms_total +=
            elapsed_ms;

        const auto logits_address =
            strix::resolve_execution_buffer(
                compiled.execution_identity,
                compiled.arena,
                memory.scratch(),
                lm_io.writes[0]);
        if (logits_address.bytes <
            logits.size() *
                sizeof(float)) {
            throw std::logic_error(
                "compiled logits allocation is undersized");
        }

        if (raw_require_identity) raw_require_identity();
        check_raw_copy(
            hipMemcpy(
                logits.data(),
                reinterpret_cast<
                    const void*>(
                        logits_address.address),
                logits.size() *
                    sizeof(float),
                hipMemcpyDeviceToHost),
            "hipMemcpy(logits D2H)", raw_drain_or_fail);
        if (raw_require_identity) raw_require_identity();

        strix::validate_serial_decode_engine_logits(
            logits,
            model.vocab_size);

        const auto ranked =
            top_logits(
                logits,
                5);
        if (ranked.empty()) {
            throw std::logic_error(
                "LM head produced no logits");
        }
        if (raw_recording) raw_recording->record_token(token, input_index, logits, session, raw_state_reader,
            input_index + 1 < options.tokens.size() ? std::optional{options.tokens[input_index + 1]} : std::nullopt);

        if (reference.has_value()) {
            const auto& expected =
                reference->steps[input_index];
            replay_observation.reference_argmax_tokens.push_back(
                expected.expected_argmax_token);
            replay_observation.engine_argmax_tokens.push_back(
                ranked.front().token);

            for (const auto& probe :
                 expected.logits) {
                if (probe.index >=
                    logits.size()) {
                    throw std::out_of_range(
                        "reference logit probe index exceeds model vocabulary");
                }
                const auto error =
                    std::abs(
                        static_cast<double>(
                            logits[probe.index]) -
                        static_cast<double>(
                            probe.value));
                replay_observation.maximum_logit_abs_error =
                    std::max(
                        replay_observation.maximum_logit_abs_error,
                        error);
                ++replay_observation.logit_values_compared;
            }
        }

        std::cout
            << "    {\"input_token\":"
            << token
            << ",\"position\":"
            << (session.next_position - 1ULL)
            << ",\"argmax_token\":"
            << ranked.front().token
            << ",\"argmax_logit\":"
            << ranked.front().logit
            << ",\"elapsed_ms\":"
            << elapsed_ms
            << ",\"top5\":[";

        for (std::size_t rank = 0;
             rank < ranked.size();
             ++rank) {
            if (rank != 0) {
                std::cout << ',';
            }
            std::cout
                << "{\"token\":"
                << ranked[rank].token
                << ",\"logit\":"
                << ranked[rank].logit
                << '}';
        }
        std::cout << "]}";
        if (input_index + 1U !=
            options.tokens.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }

    std::optional<strix::SerialDecodeReplayAssessment>
        replay_assessment;
    finalize_raw();
    if (reference.has_value()) {
        strix::SerialDecodeReplayPolicy policy;
        policy.expected_model_content_identity =
            model_residency.binding().
                model_content_identity;
        policy.expected_compiled_execution_identity =
            compiled.execution_identity;
        policy.expected_runtime_compatibility_identity =
            identities.compatibility_identity;
        policy.policy_identity =
            reference->independent_reference_identity;
        policy.max_logit_abs_error =
            options.max_logit_abs_error;
        policy.minimum_steps =
            static_cast<std::uint64_t>(
                reference->steps.size());
        policy.minimum_logit_values_compared =
            required_reference_logit_values;
        policy.require_exact_argmax = true;

        replay_assessment =
            strix::assess_serial_decode_replay(
                policy,
                replay_observation);
    }

    std::cout << "  ]," << '\n';
    emit_q4k_a8_counts();
    // Dynamic counters belong after the raw token loop, not the JSON prefix.
    std::cout << overlap_receipt();
    std::cout << strix::decode_detail::qsa_input_dispatch_receipt(qsa_prefetch_stats.get());
    std::cout << shared_dispatch_receipt.completed_counts();
    std::cout << qsa_gqa_receipt() << gdn_recurrent_layout_receipt() << qsa_selection_receipt() << gdn_islands_receipt() << strix::decode_detail::deferred_moe_completed_tokens_receipt(deferred_moe.get());
    std::cout
        << "  \"decode_elapsed_ms_total\": "
        << decode_elapsed_ms_total
        << ',' << '\n'
        << "  \"decode_elapsed_ms_per_token\": "
        << (token_elapsed_ms.empty()
                ? 0.0
                : decode_elapsed_ms_total /
                      static_cast<double>(
                          token_elapsed_ms.size()))
        << ',' << '\n'
        << "  \"committed_tokens\": "
        << session.committed_tokens
        << ',' << '\n'
        << "  \"next_position\": "
        << session.next_position;

    if (replay_assessment.has_value()) {
        std::cout
            << ',' << '\n'
            << "  \"replay\": {" << '\n'
            << "    \"reference_id\": "
            << reference->independent_reference_identity
            << ',' << '\n'
            << "    \"logit_values_compared\": "
            << replay_observation.logit_values_compared
            << ',' << '\n'
            << "    \"max_logit_abs_error\": "
            << replay_observation.maximum_logit_abs_error
            << ',' << '\n'
            << "    \"argmax_match\": "
            << (replay_assessment->argmax_match
                    ? "true"
                    : "false")
            << ',' << '\n'
            << "    \"passed\": "
            << (replay_assessment->passes_declared_policy
                    ? "true"
                    : "false")
            << '\n'
            << "  }";
    }

    std::cout
        << '\n'
        << '}' << '\n';

    return replay_assessment.has_value() &&
                   !replay_assessment->passes_declared_policy
               ? 2
               : 0;
} catch (const std::exception& error) {
    std::cerr
        << "strix-hip-decode-token: "
        << error.what() << '\n';
    return 1;
}
