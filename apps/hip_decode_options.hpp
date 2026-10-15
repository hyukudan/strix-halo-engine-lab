#pragma once
#include "strix/decode_execution_options.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/raw_baseline_recording.hpp"
#include "strix/model_contract.hpp"
#include "strix/wave_router_contract.hpp"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

namespace strix::decode_cli {
inline std::uint64_t parse_u64(
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

inline double parse_double(
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

enum class OptimizationProfile { automatic, manual };
inline OptimizationProfile parse_optimization_profile(std::string_view value) {
    if (value == "auto") return OptimizationProfile::automatic;
    if (value == "manual") return OptimizationProfile::manual;
    throw std::invalid_argument("--optimization-profile requires auto or manual");
}
struct OptimizationProfileResolution {
    std::string requested = "auto";
    std::string selected = "manual";
    std::string reason = "not-resolved";
    bool automatic_applied = false;
};
struct Options {
    bool execution_inventory = false;
    bool host_dispatch_telemetry = false;
    OptimizationProfile optimization_profile = OptimizationProfile::automatic;
    bool profile_explicit = false;
    bool any_low_level_flag = false;
    bool context_explicit = false;
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
    bool typed_hyper_projection = false;
    bool prefetch_typed_hyper_rows = false;
    bool prefetch_q8_hyper_output = false;
    bool lane_chain_hyper_output = false;
    bool virtual_q8_lm_head = false;
    bool parallel_moe_router = false;
    bool virtual_q8_gdn_input = false;
    bool ranked_moe_router_topk = false;
    bool wave_moe_router_topk = false;
    bool q4k_a8g32_gate_up = false;
    bool grouped_exact_q51_down = false;
    bool deferred_moe_status = false;
    bool prefetch_q8_gdn_input = false;
    bool prefetch_q8_qsa_input = false;
    bool prefetch_q8_shared_expert = false;
    bool virtual_q8_gdn_output = false;
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

inline bool is_low_level_optimization_option(std::string_view argument) {
    for (const std::string_view option : {
        "--parallel-hyper-mix", "--staged-hyper-projections", "--fused-staged-hyper-mix",
        "--typed-fused-hyper-projections", "--prefetch-typed-hyper-rows",
        "--prefetch-q8-hyper-output", "--lane-chain-hyper-output", "--virtual-q8-lm-head",
        "--parallel-moe-router", "--ranked-moe-router-topk", "--wave-moe-router-topk",
        "--virtual-q8-gdn-input", "--prefetch-q8-gdn-input", "--virtual-q8-gdn-output",
        "--q4k-a8g32-gate-up", "--grouped-exact-q51-down", "--deferred-moe-status",
        "--prefetch-q8-qsa-input", "--prefetch-q8-shared-expert", "--planned-ple-io",
        "--ple-max-rows-per-io", "--ple-max-inflight-ios", "--coalesce-gap"})
        if (argument == option) return true;
    return false;
}

// Context can change only after immutable model/profile admission. Roundtrip
// validates the resolved prefix capacity; generation may stop at the existing
// context_capacity boundary before reaching the requested token count.
inline void validate_resolved_prefix_roundtrip(const Options& options) {
    if (!options.prefix_cache_roundtrip) return;
    validate_decode_prefix_roundtrip(options.tokens.size(), options.context_tokens,
                                    options.generate_tokens);
}

inline Options parse_options(
    int argc,
    char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: strix-hip-decode-token MODEL.gguf [TOKEN_ID ...] "
            "[--optimization-profile auto|manual] [--reference FILE] [--max-logit-error X] "
            "[--context N] [--residency register|copy] [--coalesce-gap BYTES] "
            "[--planned-ple-io] [--ple-max-rows-per-io N] [--ple-max-inflight-ios N] "
            "[--generate N] [--eos-token TOKEN_ID] [--prefix-cache-roundtrip] "
            "[--parallel-moe-router] [--ranked-moe-router-topk | --wave-moe-router-topk] "
            "[--prefetch-q8-qsa-input] [--prefetch-q8-shared-expert] [--virtual-q8-lm-head] [--virtual-q8-gdn-input [--prefetch-q8-gdn-input]] [--q4k-a8g32-gate-up [--grouped-exact-q51-down]] [--deferred-moe-status] "
            "[--parallel-hyper-mix [--staged-hyper-projections [--fused-staged-hyper-mix [--typed-fused-hyper-projections [--prefetch-typed-hyper-rows] [--prefetch-q8-hyper-output [--lane-chain-hyper-output]]]]]]");
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
        options.any_low_level_flag = options.any_low_level_flag ||
            is_low_level_optimization_option(argument);
        if (argument == "--execution-inventory") {
            if (options.execution_inventory)
                throw std::invalid_argument("duplicate execution inventory option");
            options.execution_inventory = true;
            continue;
        }
        if (argument == "--host-dispatch-telemetry") {
            if (options.host_dispatch_telemetry)
                throw std::invalid_argument("duplicate host dispatch telemetry option");
            options.host_dispatch_telemetry = true;
            continue;
        }
        if (argument == "--optimization-profile") {
            if (!execution_option_names.insert(argument).second)
                throw std::invalid_argument("duplicate optimization profile option");
            if (index + 1 >= argc)
                throw std::invalid_argument("--optimization-profile requires auto or manual");
            options.optimization_profile = parse_optimization_profile(argv[++index]);
            options.profile_explicit = true;
            continue;
        }
        if (argument == "--grouped-exact-q51-down") {
#ifndef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
            throw std::invalid_argument("--grouped-exact-q51-down requires the experimental runtime build capability");
#else
            if(options.grouped_exact_q51_down) throw std::invalid_argument("duplicate grouped exact Q5_1 down option");
            options.grouped_exact_q51_down=true;
            continue;
#endif
        }
        if (argument == "--q4k-a8g32-gate-up") {
#ifndef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
            throw std::invalid_argument("--q4k-a8g32-gate-up requires the experimental runtime build capability");
#else
            if (options.q4k_a8g32_gate_up) throw std::invalid_argument("duplicate Q4K A8G32 gate/up option");
            options.q4k_a8g32_gate_up = true;
            continue;
#endif
        }
        if (argument == "--wave-moe-router-topk") {
            if (options.wave_moe_router_topk) throw std::invalid_argument("duplicate wave MoE router top-k option");
            options.wave_moe_router_topk = true;
            continue;
        }
        if (argument == "--deferred-moe-status") {
            if (options.deferred_moe_status) throw std::invalid_argument("duplicate deferred MoE status option");
            options.deferred_moe_status = true; continue;
        }
        if (argument == "--ranked-moe-router-topk") {
            if (options.ranked_moe_router_topk) throw std::invalid_argument("duplicate ranked MoE router top-k option");
            options.ranked_moe_router_topk = true;
            continue;
        }
        if (argument == "--parallel-moe-router") {
            if (options.parallel_moe_router) throw std::invalid_argument("duplicate parallel MoE router option");
            options.parallel_moe_router = true;
            continue;
        }
        if(argument=="--virtual-q8-gdn-output") {
            if(options.virtual_q8_gdn_output) throw std::invalid_argument("duplicate virtual Q8 GDN output option");
            options.virtual_q8_gdn_output=true;continue;
        }
        if(argument=="--lane-chain-hyper-output"){
            if(options.lane_chain_hyper_output)throw std::invalid_argument("duplicate lane-chain HC output option");
            options.lane_chain_hyper_output=true;continue;
        }
        if (argument == "--prefetch-q8-hyper-output") {
            if (options.prefetch_q8_hyper_output) throw std::invalid_argument("duplicate Q8 hyper output prefetch option");
            options.prefetch_q8_hyper_output = true;
            continue;
        }
        if (argument == "--prefetch-typed-hyper-rows") {
            if (options.prefetch_typed_hyper_rows) throw std::invalid_argument("duplicate typed hyper row prefetch option");
            options.prefetch_typed_hyper_rows = true;
            continue;
        }
        if (argument == "--typed-fused-hyper-projections") {
            if (options.typed_hyper_projection) throw std::invalid_argument("duplicate typed hyper projection option");
            options.typed_hyper_projection = true;
            continue;
        }
        if (argument == "--parallel-hyper-mix") {
            if (options.parallel_hyper_mix) throw std::invalid_argument("duplicate parallel hyper-mix option");
            options.parallel_hyper_mix = true;
            continue;
        }
        if(argument=="--prefetch-q8-qsa-input"){if(options.prefetch_q8_qsa_input)throw std::invalid_argument("duplicate prefetch Q8 QSA input option");options.prefetch_q8_qsa_input=true;continue;}
        if(argument=="--prefetch-q8-shared-expert"){if(options.prefetch_q8_shared_expert)throw std::invalid_argument("duplicate prefetch Q8 shared expert option");options.prefetch_q8_shared_expert=true;continue;}
        if (argument == "--prefetch-q8-gdn-input") {
            if (options.prefetch_q8_gdn_input) throw std::invalid_argument("duplicate prefetch Q8 GDN input option");
            options.prefetch_q8_gdn_input = true;
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
            options.context_explicit = true;
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
    if (options.context_tokens == 0) {
        throw std::invalid_argument(
            "context must be non-zero");
    }
    if (!options.raw_recording_path.empty()) {
        strix::validate_raw_baseline_output_directory(options.raw_recording_path);
    }
    else if (options.raw_policy_supplied) throw std::invalid_argument("recording policy requires --record-raw");
    if (options.deferred_moe_status && (options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none))
        throw std::invalid_argument("--deferred-moe-status requires eager C1 without an integer-head stream");
    options.execution.validate(options.residency == strix::HipModelResidencyMode::resident_copy,
                               options.prefix_cache_roundtrip);
    if (options.execution.mode == strix::DecodeExecutionMode::graph && options.planned_ple_io)
        throw std::invalid_argument("graph mode uses resident device PLE gather, not planned host IO");
    if (options.ple_max_inflight_ios == 0) {
        throw std::invalid_argument(
            "ple-max-inflight-ios must be non-zero");
    }
    if(options.grouped_exact_q51_down && !options.q4k_a8g32_gate_up)
        throw std::invalid_argument("--grouped-exact-q51-down requires --q4k-a8g32-gate-up");
    if (options.wave_moe_router_topk && (!options.parallel_moe_router || options.ranked_moe_router_topk))
        throw std::invalid_argument("--wave-moe-router-topk requires --parallel-moe-router and excludes --ranked-moe-router-topk");
    if (options.ranked_moe_router_topk && !options.parallel_moe_router)
        throw std::invalid_argument("--ranked-moe-router-topk requires --parallel-moe-router");
    if (options.prefetch_q8_gdn_input && !options.virtual_q8_gdn_input)
        throw std::invalid_argument("--prefetch-q8-gdn-input requires --virtual-q8-gdn-input");
    if(options.lane_chain_hyper_output && !options.prefetch_q8_hyper_output)
        throw std::invalid_argument("--lane-chain-hyper-output requires --prefetch-q8-hyper-output");
    if (options.prefetch_q8_hyper_output && !options.typed_hyper_projection)
        throw std::invalid_argument("Q8 hyper output prefetch requires typed projection mode");
    if (options.prefetch_typed_hyper_rows && !options.typed_hyper_projection)
        throw std::invalid_argument("typed hyper row prefetch requires typed projection mode");
    if (options.typed_hyper_projection && !(options.parallel_hyper_mix && options.staged_hyper_projections && options.fused_staged_hyper_mix))
        throw std::invalid_argument("typed hyper projections require parallel, staged and fused options");
    if (options.staged_hyper_projections && !options.parallel_hyper_mix)
        throw std::invalid_argument("--staged-hyper-projections requires --parallel-hyper-mix");
    if (options.fused_staged_hyper_mix && !options.staged_hyper_projections)
        throw std::invalid_argument("--fused-staged-hyper-mix requires --staged-hyper-projections");
    if (options.virtual_q8_lm_head && options.execution.integer_head != strix::DecodeIntegerHead::none)
        throw std::invalid_argument("integer and virtual Q8 LM heads are mutually exclusive");
    return options;
}

inline constexpr std::string_view validated_model_content_identity =
    "3726d6f0d9ba30b9d655ba303595118941baeca5e6945fc8785f938e0f5455e4";

inline OptimizationProfileResolution resolve_optimization_profile(
    Options& options, const ModelContract& model, const gguf::FileInfo& info,
    const ModelManifest& manifest, std::string_view content_identity,
    bool has_q4k_a8_runtime) {
    OptimizationProfileResolution result;
    const auto fallback = [&](std::string reason) {
        result.reason = std::move(reason);
        return result;
    };
    if (options.optimization_profile == OptimizationProfile::manual) {
        result.requested = "manual";
        return fallback("explicit-manual");
    }
    if (options.any_low_level_flag && !options.profile_explicit)
        return fallback("explicit-low-level-options");
    if (options.execution.mode != DecodeExecutionMode::eager)
        return fallback("execution-mode-not-qualified");
    if (options.execution.integer_head != DecodeIntegerHead::none)
        return fallback("integer-head-explicit");
    if (options.residency != HipModelResidencyMode::registered_file_mapping)
        return fallback("residency-not-qualified");
    if (!options.reference_path.empty())
        return fallback("explicit-replay-reference");
    if (!has_q4k_a8_runtime)
        return fallback("a8-runtime-not-built");
    if (content_identity != validated_model_content_identity)
        return fallback("checkpoint-content-not-qualified");
    if (options.context_explicit && options.context_tokens != 8192)
        return fallback("context-not-qualified");
    if (options.planned_ple_io)
        return fallback("planned-ple-io-explicit");
    const auto canonical = ModelContract::qwen38_flash_next();
    if (!model.validate().empty() || model.hidden_size != canonical.hidden_size ||
        model.hidden_layers != canonical.hidden_layers || model.vocab_size != canonical.vocab_size ||
        model.hc_count != canonical.hc_count || model.hc_lowrank != canonical.hc_lowrank ||
        model.num_experts != canonical.num_experts ||
        model.num_experts_per_token != canonical.num_experts_per_token ||
        model.moe_intermediate_size != canonical.moe_intermediate_size ||
        model.shared_expert_intermediate_size != canonical.shared_expert_intermediate_size ||
        model.full_attention_interval != canonical.full_attention_interval)
        return fallback("model-contract-not-qualified");
    try {
        validate_wave_router_model_inputs(info);
        validate_wave_router_manifest(manifest);
        if (manifest.layout_identity == 0 ||
            model_manifest_layout_identity(manifest) != manifest.layout_identity)
            return fallback("manifest-layout-not-qualified");
        validate_wave_router_selection_contract(model.num_experts, model.num_experts_per_token,
                                               true, false);
    } catch (const std::invalid_argument&) {
        return fallback("router-contract-not-qualified");
    }
    // The measured grouped A8 profile is a separate numerical identity. Its
    // allowlist is this exact content identity, not all same-shaped checkpoints.
    options.parallel_hyper_mix = true;
    options.staged_hyper_projections = true;
    options.fused_staged_hyper_mix = true;
    options.virtual_q8_lm_head = true;
    options.parallel_moe_router = true;
    options.virtual_q8_gdn_input = true;
    options.prefetch_q8_gdn_input = true;
    options.typed_hyper_projection = true;
    options.prefetch_typed_hyper_rows = true;
    options.q4k_a8g32_gate_up = true;
    options.grouped_exact_q51_down = true;
    options.prefetch_q8_hyper_output = true;
    options.lane_chain_hyper_output = true;
    options.virtual_q8_gdn_output = true;
    options.prefetch_q8_qsa_input = true;
    options.prefetch_q8_shared_expert = true;
    options.deferred_moe_status = true;
    // This policy candidate requires fresh automatic-profile qualification.
    // Explicit ranked routing keeps its requested algorithm. Otherwise the
    // exact wave replacement is the qualified faster selection.
    if (!options.ranked_moe_router_topk) options.wave_moe_router_topk = true;
    options.context_tokens = 8192;
    result.automatic_applied = true;
    result.selected = options.ranked_moe_router_topk ? "canonical-a8-ranked-qsa-hc-shared-deferred-v3" :
                                                    "canonical-a8-wave-qsa-hc-shared-deferred-v3";
    result.reason = options.ranked_moe_router_topk ? "explicit-ranked-router" : "qualified-checkpoint";
    return result;
}

} // namespace strix::decode_cli
