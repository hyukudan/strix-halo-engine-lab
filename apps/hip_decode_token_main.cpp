#include "strix/compiled_execution.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/raw_baseline_recording.hpp"
#include "strix/decode_execution_options.hpp"
#include "strix/decode_integer_head.hpp"
#include "strix/hip_integer_projection_runtime.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "hip_decode_experiment.hpp"
#include "strix/serial_decode_reference.hpp"
#include "strix/serial_decode_replay.hpp"
#include "strix/serial_prefix_cache.hpp"
#include "strix/serial_request.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_state_layout.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
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

std::uint64_t parse_u64(
    std::string_view text,
    const char* name) {
    if (text.empty()) {
        throw std::invalid_argument(
            std::string(name) +
            " must not be empty");
    }
    std::uint64_t value = 0;
    for (const char character : text) {
        if (character < '0' ||
            character > '9') {
            throw std::invalid_argument(
                std::string(name) +
                " must be an unsigned integer");
        }
        const auto digit =
            static_cast<std::uint64_t>(
                character - '0');
        if (value >
            (std::numeric_limits<
                 std::uint64_t>::max() -
             digit) /
                10ULL) {
            throw std::overflow_error(
                std::string(name) +
                " overflows uint64");
        }
        value =
            value * 10ULL +
            digit;
    }
    return value;
}

double parse_double(
    std::string_view text,
    const char* name) {
    if (text.empty()) {
        throw std::invalid_argument(
            std::string(name) +
            " must not be empty");
    }
    std::string copy(text);
    char* end = nullptr;
    errno = 0;
    const auto value =
        std::strtod(
            copy.c_str(),
            &end);
    if (errno != 0 ||
        end == copy.c_str() ||
        *end != '\0' ||
        !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(name) +
            " must be a finite number");
    }
    return value;
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

struct Options {
    std::string model_path;
    std::vector<std::uint32_t> tokens;
    std::uint64_t context_tokens = 128;
    strix::HipModelResidencyMode residency =
        strix::HipModelResidencyMode::
            registered_file_mapping;
    std::uint64_t coalesce_gap_bytes = 4096;
    bool planned_ple_io = false;
    bool parallel_hyper_mix = false;
    bool staged_hyper_projections = false;
    bool fused_staged_hyper_mix = false;
    bool virtual_q8_lm_head = false;
    bool parallel_moe_router = false;
    bool virtual_q8_gdn_input = false;
    std::uint64_t ple_max_rows_per_io = 4;
    std::uint64_t ple_max_inflight_ios = 4;
    std::string reference_path;
    double max_logit_abs_error = 1e-3;
    std::uint64_t generate_tokens = 0;
    std::optional<std::uint32_t> eos_token;
    bool prefix_cache_roundtrip = false;
    std::string raw_recording_path;
    std::uint64_t raw_maximum_bytes = 8ULL * 1024 * 1024 * 1024;
    strix::RawBaselineStatePolicy raw_state_policy = strix::RawBaselineStatePolicy::final;
    bool raw_policy_supplied = false;
    strix::DecodeExecutionOptions execution;
};

Options parse_options(
    int argc,
    char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: strix-hip-decode-token MODEL.gguf [TOKEN_ID ...] "
            "[--reference FILE] [--max-logit-error X] "
            "[--context N] [--residency register|copy] [--coalesce-gap BYTES] "
            "[--planned-ple-io] [--ple-max-rows-per-io N] [--ple-max-inflight-ios N] "
            "[--generate N] [--eos-token TOKEN_ID] [--prefix-cache-roundtrip] "
            "[--parallel-moe-router] [--virtual-q8-lm-head] [--virtual-q8-gdn-input] "
            "[--parallel-hyper-mix [--staged-hyper-projections [--fused-staged-hyper-mix]]]");
    }

    Options options;
    options.model_path =
        argv[1];
    std::set<std::string_view> execution_option_names;

    for (int index = 2;
         index < argc;
         ++index) {
        const std::string_view argument =
            argv[index];
        if (argument == "--parallel-moe-router") {
            if (options.parallel_moe_router) throw std::invalid_argument("duplicate parallel MoE router option");
            options.parallel_moe_router = true;
            continue;
        }
        if (argument == "--parallel-hyper-mix") {
            if (options.parallel_hyper_mix) throw std::invalid_argument("duplicate parallel hyper-mix option");
            options.parallel_hyper_mix = true;
            continue;
        }
        if (argument == "--virtual-q8-gdn-input") {
            if (options.virtual_q8_gdn_input) throw std::invalid_argument("duplicate virtual Q8 GDN input option");
            options.virtual_q8_gdn_input = true;
            continue;
        }
        if (argument == "--virtual-q8-lm-head") {
            if (options.virtual_q8_lm_head) throw std::invalid_argument("duplicate virtual Q8 LM-head option");
            options.virtual_q8_lm_head = true;
            continue;
        }
        if (argument == "--staged-hyper-projections") {
            if (options.staged_hyper_projections) throw std::invalid_argument("duplicate staged hyper projections option");
            options.staged_hyper_projections = true;
            continue;
        }
        if (argument == "--fused-staged-hyper-mix") {
            if (options.fused_staged_hyper_mix) throw std::invalid_argument("duplicate fused staged hyper-mix option");
            options.fused_staged_hyper_mix = true;
            continue;
        }

        if (argument == "--execution-mode" || argument == "--integer-lm-head" ||
            argument == "--graph-workspace-bytes" || argument == "--graph-executable-bytes" ||
            argument == "--graph-capture-peak-bytes" || argument == "--graph-budget-bytes" ||
            argument == "--integer-internal-bytes" || argument == "--integer-retained-bytes") {
            if (!execution_option_names.insert(argument).second)
                throw std::invalid_argument("duplicate decode execution option");
            if (index + 1 >= argc) throw std::invalid_argument("decode execution option requires a value");
            const std::string_view value = argv[++index];
            if (argument == "--execution-mode") options.execution.mode = strix::parse_decode_execution_mode(value);
            else if (argument == "--integer-lm-head") options.execution.integer_head = strix::parse_decode_integer_head(value);
            else {
                const auto bytes = parse_u64(value, "decode experiment memory ceiling");
                if (argument == "--graph-workspace-bytes") options.execution.graph_workspace_bytes = bytes;
                else if (argument == "--graph-executable-bytes") options.execution.graph_executable_bytes = bytes;
                else if (argument == "--graph-capture-peak-bytes") options.execution.graph_capture_peak_bytes = bytes;
                else if (argument == "--graph-budget-bytes") options.execution.graph_budget_bytes = bytes;
                else if (argument == "--integer-internal-bytes") options.execution.integer_internal_bytes = bytes;
                else options.execution.integer_retained_bytes = bytes;
            }
            continue;
        }

        if (argument == "--record-raw") {
            if (index + 1 >= argc || !options.raw_recording_path.empty()) throw std::invalid_argument("--record-raw requires one fresh private directory");
            options.raw_recording_path = argv[++index];
            continue;
        }
        if (argument == "--record-max-bytes") {
            if (index + 1 >= argc) throw std::invalid_argument("--record-max-bytes requires a byte budget");
            options.raw_maximum_bytes = parse_u64(argv[++index], "record-max-bytes");
            options.raw_policy_supplied = true;
            continue;
        }
        if (argument == "--record-state-policy") {
            if (index + 1 >= argc) throw std::invalid_argument("--record-state-policy requires final or each");
            const std::string_view value = argv[++index];
            if (value == "final") options.raw_state_policy = strix::RawBaselineStatePolicy::final;
            else if (value == "each") options.raw_state_policy = strix::RawBaselineStatePolicy::each;
            else throw std::invalid_argument("--record-state-policy requires final or each");
            options.raw_policy_supplied = true;
            continue;
        }

        if (argument == "--reference") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--reference requires a fixture path");
            }
            options.reference_path =
                argv[++index];
            continue;
        }

        if (argument == "--max-logit-error") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--max-logit-error requires a value");
            }
            options.max_logit_abs_error =
                parse_double(
                    argv[++index],
                    "max-logit-error");
            if (options.max_logit_abs_error < 0.0) {
                throw std::invalid_argument(
                    "--max-logit-error must be non-negative");
            }
            continue;
        }

        if (argument == "--context") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--context requires a token count");
            }
            options.context_tokens =
                parse_u64(
                    argv[++index],
                    "context");
            continue;
        }

        if (argument == "--residency") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--residency requires register or copy");
            }
            const std::string_view value =
                argv[++index];
            if (value == "register") {
                options.residency =
                    strix::HipModelResidencyMode::
                        registered_file_mapping;
            } else if (value == "copy") {
                options.residency =
                    strix::HipModelResidencyMode::
                        resident_copy;
            } else {
                throw std::invalid_argument(
                    "--residency must be register or copy");
            }
            continue;
        }

        if (argument == "--coalesce-gap") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--coalesce-gap requires bytes");
            }
            options.coalesce_gap_bytes =
                parse_u64(
                    argv[++index],
                    "coalesce-gap");
            continue;
        }

        if (argument == "--planned-ple-io") {
            options.planned_ple_io = true;
            continue;
        }

        if (argument == "--ple-max-rows-per-io") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--ple-max-rows-per-io requires a value");
            }
            options.ple_max_rows_per_io =
                parse_u64(
                    argv[++index],
                    "ple-max-rows-per-io");
            continue;
        }

        if (argument == "--ple-max-inflight-ios") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--ple-max-inflight-ios requires a value");
            }
            options.ple_max_inflight_ios =
                parse_u64(
                    argv[++index],
                    "ple-max-inflight-ios");
            continue;
        }

        if (argument == "--generate") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--generate requires a token count");
            }
            options.generate_tokens =
                parse_u64(
                    argv[++index],
                    "generate");
            continue;
        }

        if (argument == "--eos-token") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--eos-token requires a token id");
            }
            const auto value =
                parse_u64(
                    argv[++index],
                    "eos-token");
            if (value >
                std::numeric_limits<std::uint32_t>::max()) {
                throw std::out_of_range(
                    "eos-token exceeds uint32");
            }
            options.eos_token =
                static_cast<std::uint32_t>(
                    value);
            continue;
        }

        if (argument == "--prefix-cache-roundtrip") {
            options.prefix_cache_roundtrip = true;
            continue;
        }

        if (argument.starts_with("--")) {
            throw std::invalid_argument(
                "unknown option: " +
                std::string(argument));
        }

        const auto token =
            parse_u64(
                argument,
                "token id");
        if (token >
            std::numeric_limits<
                std::uint32_t>::max()) {
            throw std::out_of_range(
                "token id exceeds uint32");
        }
        options.tokens.push_back(
            static_cast<std::uint32_t>(
                token));
    }

    if (options.tokens.empty() &&
        options.reference_path.empty()) {
        throw std::invalid_argument(
            "at least one token id or --reference is required");
    }
    if (options.generate_tokens != 0 &&
        !options.reference_path.empty()) {
        throw std::invalid_argument(
            "--generate and --reference are separate execution modes");
    }
    if (options.generate_tokens != 0 &&
        options.tokens.empty()) {
        throw std::invalid_argument(
            "--generate requires at least one prefix token id");
    }
    if (options.prefix_cache_roundtrip)
        strix::validate_decode_prefix_roundtrip(options.tokens.size(), options.context_tokens, options.generate_tokens);
    if (options.context_tokens == 0) {
        throw std::invalid_argument(
            "context must be non-zero");
    }
    if (!options.raw_recording_path.empty()) {
        strix::validate_raw_baseline_output_directory(options.raw_recording_path);
    }
    else if (options.raw_policy_supplied) throw std::invalid_argument("recording policy requires --record-raw");
    options.execution.validate(options.residency == strix::HipModelResidencyMode::resident_copy,
                               options.prefix_cache_roundtrip);
    if (options.execution.mode == strix::DecodeExecutionMode::graph && options.planned_ple_io)
        throw std::invalid_argument("graph mode uses resident device PLE gather, not planned host IO");
    if (options.ple_max_inflight_ios == 0) {
        throw std::invalid_argument(
            "ple-max-inflight-ios must be non-zero");
    }
    if (options.staged_hyper_projections && !options.parallel_hyper_mix)
        throw std::invalid_argument("--staged-hyper-projections requires --parallel-hyper-mix");
    if (options.fused_staged_hyper_mix && !options.staged_hyper_projections)
        throw std::invalid_argument("--fused-staged-hyper-mix requires --staged-hyper-projections");
    if (options.virtual_q8_lm_head && options.execution.integer_head != strix::DecodeIntegerHead::none)
        throw std::invalid_argument("integer and virtual Q8 LM heads are mutually exclusive");
    return options;
}

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
        std::function<void(std::uint32_t)> execute_override = {}, bool track_logit_hashes = false)
        : compiled_(&compiled),
          session_(&session),
          resources_(&resources),
          dispatcher_(&dispatcher),
          scratch_(&scratch),
          lm_io_(&lm_io),
          vocab_size_(vocab_size), recording_(std::move(recording)), recording_binding_(std::move(recording_binding)), recording_drain_(std::move(recording_drain)), execute_override_(std::move(execute_override)), track_logit_hashes_(track_logit_hashes) {}

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

        check_hip(
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

private:
    const strix::CompiledSerialExecution* compiled_ = nullptr;
    strix::SerialSessionContext* session_ = nullptr;
    const strix::ExecutionRuntimeResources* resources_ = nullptr;
    strix::EagerExecutionDispatcher* dispatcher_ = nullptr;
    const strix::ExecutionMemoryBinding* scratch_ = nullptr;
    const strix::ExecutionStepIo* lm_io_ = nullptr;
    std::uint64_t vocab_size_ = 0;
    std::vector<double> elapsed_ms_;
    std::vector<std::string> logit_hashes_;
    std::function<void(std::uint32_t, std::uint64_t, std::span<const float>)> recording_;
    std::function<void()> recording_binding_;
    std::function<void()> recording_drain_;
    std::function<void(std::uint32_t)> execute_override_;
    bool track_logit_hashes_ = false;
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
            "  --record-raw ABS_NEW_PRIVATE_DIRECTORY --record-max-bytes BYTES\n"
            "  --record-state-policy final|each --prefix-cache-roundtrip\n"
            "  --execution-mode eager|graph (default eager)\n"
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

    const auto info =
        strix::gguf::read_file(
            options.model_path);
    const auto manifest =
        strix::build_model_manifest(
            info);
    const auto tensor_index =
        strix::build_model_tensor_index(
            manifest);
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
    catalog_config.virtual_q8_lm_head = options.virtual_q8_lm_head;
    catalog_config.parallel_moe_router = options.parallel_moe_router;
    binding_request.parallel_moe_router = options.parallel_moe_router;
    catalog_config.virtual_q8_gdn_input = options.virtual_q8_gdn_input;
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

    const auto compiled =
        strix::compile_serial_execution(
            model,
            kernel_registry,
            binding_request,
            buffer_request,
            256);

    const auto tensor_access = strix::compile_execution_tensor_access_plan(
        compiled.execution, manifest, tensor_index,
        strix::build_serial_execution_tensor_requirements(compiled.execution));

    strix::SerialSessionStateLayoutRequest
        state_request;
    state_request.context_tokens =
        options.context_tokens;
    const auto state_layout =
        strix::build_serial_session_state_layout(
            model,
            state_request);

    strix::AllocationGenerationSource
        generations;
    auto mapped_owner = std::make_shared<strix::MappedModelFile>(
        strix::MappedModelFile::open_read_only(options.model_path, "model", generations));
    auto& mapped = *mapped_owner;

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
    auto launcher_catalog = std::make_shared<strix::HipLauncherCatalog>(model, catalog_config, integer_bundle);
    strix::ExecutionLauncherRegistry launchers;
    launcher_catalog->register_for_compiled_execution(compiled, launchers);
    if (launcher_catalog->audit(compiled, launchers).missing_steps)
        throw std::logic_error("HIP launcher catalog is incomplete for actual compiled execution");
    const auto launch_plan = strix::compile_launch_plan(compiled, launchers);
    strix::DirectLaunchDispatcher dispatcher(launch_plan);
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

    std::uint64_t input_access_epoch = 1;
    const std::function<void(std::uint32_t)> execute_native_token = [&](std::uint32_t token) {
        original_work_device->require_original();
        if (arena_pin) {
            if (input_access_epoch == UINT64_MAX) throw std::overflow_error("integer input epochs exhausted");
            arena_pin->advance_access(++input_access_epoch);
        }
        if (graph) static_cast<void>(graph->execute(session, token));
        else static_cast<void>(strix::execute_serial_token(compiled, session, resources, token, dispatcher));
        check_hip(hipDeviceSynchronize(), "native token original-device completion");
        original_work_device->require_original();
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
    if (!options.raw_recording_path.empty()) {
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
        raw_recording = std::make_unique<strix::RawBaselineRecording>(options.raw_recording_path, std::move(metadata), policy, state_layout);
        raw_recording->capture_initial(session, raw_state_reader);
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
                    }, raw_require_identity, raw_drain_or_fail, execute_native_token);

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
                    model.vocab_size, {}, raw_require_identity, raw_drain_or_fail, execute_native_token, true);
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
                    }, raw_require_identity, raw_drain_or_fail, execute_native_token, true);
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

            std::cout
                << "{\n"
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

        HipSerialGreedyStepBackend backend(
            compiled,
            session,
            resources,
            dispatcher,
            memory.scratch(),
            lm_io,
            model.vocab_size,
            [&](std::uint32_t token, std::uint64_t position, std::span<const float> values) {
                if (raw_recording) raw_recording->record_token(token, position, values, session, raw_state_reader);
            }, raw_require_identity, raw_drain_or_fail, execute_native_token);

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

        const auto result =
            strix::execute_serial_greedy_request(
                request,
                backend);
        finalize_raw(result.generated_tokens, std::string(stop_reason_name(result.stop_reason)));

        double elapsed_total = 0.0;
        for (const auto value :
             backend.elapsed_ms()) {
            elapsed_total += value;
        }

        std::cout
            << "{\n"
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
            << "  \"decode_elapsed_ms_total\": "
            << elapsed_total
            << ",\n"
            << "  \"decode_elapsed_ms_per_step\": "
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

    std::cout << '{' << '\n'
              << "  \"model_content_identity\": \""
              << strix::model_content_identity_hex(
                     model_residency.binding().model_content_identity)
              << "\"," << '\n'
              << "  \"execution_identity\": "
              << compiled.execution_identity
              << ',' << '\n'
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

        check_hip(
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

    std::cout
        << "  ]," << '\n'
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
