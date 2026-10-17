#pragma once
#include "strix/decode_execution_options.hpp"
#include "strix/decode_numerical_profile.hpp"
#include "strix/gdn_recurrent_layout.hpp"
#include "strix/gdn_recurrent_storage.hpp"
#include "strix/known_window_output_policy.hpp"
#include "strix/session_snapshot.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/raw_baseline_recording.hpp"
#include "strix/final_generation_state_export.hpp"
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
// Local, externally pinned MTP assets and independent replay evidence are
// mandatory. None of these fields is an execution/qualification override.
struct MtpCliOptions {
    bool enabled=false,supplied=false,metadata_only=false;
    std::string sidecar,sidecar_logical_name,sidecar_sha256,target_sha256,target_content,golden,golden_sha256;
    std::string repository,revision,source_path,converter_sha256,norm_converter_sha256,converter_revision;
    std::string conversion,normalized_rounding,projection_rounding,sidecar_residency;
    std::uint64_t sidecar_bytes=0,header_bytes=0,sidecar_retained_bytes=0,sidecar_setup_bytes=0;
    std::uint64_t golden_max_bytes=0,replay_host_bytes=0,runtime_bytes=0,target_capture_bytes=0;
    std::uint64_t policy_identity=0;
    std::uint32_t candidate_depth=0,window_rows=0;
    std::uint64_t window_arena_bytes=0,window_capture_bytes=0,window_checkpoint_bytes=0,window_total_bytes=0,window_history_intervals=0;
    double max_hidden_error=0.0,max_logit_error=0.0;
    bool window_expert_union=false;
    std::uint64_t window_expert_union_bytes=0;
    bool window_expert_union_bytes_explicit=false;
};
// This is execution evidence: a configured owner alone is not an attempted
// native window (EOS, remaining budget/context and history can choose serial).
inline std::string_view mtp_target_verification_name(std::uint64_t attempted_windows) noexcept {
    return attempted_windows ? "closed-known-window-attempted-with-serial-fallback" : "serial-target-fallback";
}
inline bool mtp_sha256_text(std::string_view value) noexcept {
    if(value.size()!=64U)return false;
    bool nonzero=false;
    for(char ch:value){if(!((ch>='0'&&ch<='9')||(ch>='a'&&ch<='f')||(ch>='A'&&ch<='F')))return false;nonzero=nonzero||ch!='0';}
    return nonzero;
}
struct Options {
    MtpCliOptions mtp;
    bool moe_shared_overlap = false;
    bool moe_token_cost_policy = false;
    std::uint64_t moe_overlap_arena_budget_bytes = 64ULL * 1024ULL * 1024ULL;
    bool moe_overlap_budget_explicit = false;
    bool batched_moe_activation = false;
    bool prepared_eager = false;
    bool gpu_greedy_token = false;
    bool execution_inventory = false;
    bool host_dispatch_telemetry = false;
    bool execution_telemetry_trace = false;
    OptimizationProfile optimization_profile = OptimizationProfile::automatic;
    strix::DecodeNumericalProfile numerical_profile = strix::DecodeNumericalProfile::exact_v5;
    bool numerical_profile_explicit = false;
    bool profile_explicit = false;
    bool any_low_level_flag = false;
    bool context_explicit = false;
    std::string model_path;
    std::vector<std::uint32_t> tokens;
    std::uint64_t context_tokens = 128;
    strix::HipModelResidencyMode residency =
        strix::HipModelResidencyMode::
            registered_file_mapping;
    strix::HipModelCopyTier model_copy_tier = strix::HipModelCopyTier::none;
    bool model_copy_tier_explicit = false;
    std::optional<std::uint64_t> model_copy_payload_bytes;
    std::optional<std::uint64_t> model_retained_budget, model_setup_peak_budget;
    std::uint64_t coalesce_gap_bytes = 4096;
    bool planned_ple_io = false;
    bool parallel_hyper_mix = false;
    bool staged_hyper_projections = false;
    bool fused_staged_hyper_mix = false;
    bool typed_hyper_projection = false;
    bool prefetch_typed_hyper_rows = false;
    bool prefetch_q8_hyper_output = false;
    bool lane_chain_hyper_output = false;
    bool prefetch_hc_projection_next_tile = false;
    bool virtual_q8_lm_head = false;
    bool prefetch_q8_lm_head = false;
    bool wave8_q8_lm_head = false;
    bool wave8_q8_gdn_input = false;
    bool wave8_q8_gdn_output = false;
    bool wave8_q8_qsa_output = false;
    bool parallel_moe_router = false;
    bool staged_moe_router_logits = false;
    bool virtual_q8_gdn_input = false;
    bool ranked_moe_router_topk = false;
    bool wave_moe_router_topk = false;
    bool q4k_a8g32_gate_up = false;
    bool grouped_exact_q51_down = false;
    bool q51_slot_parallel_down = false;
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
    std::string validation_final_state_path;
    std::uint64_t validation_final_state_maximum_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    bool validation_final_state_budget_explicit = false;
    std::uint64_t raw_maximum_bytes = 8ULL * 1024 * 1024 * 1024;
    strix::RawBaselineStatePolicy raw_state_policy = strix::RawBaselineStatePolicy::final;
    bool raw_policy_supplied = false;
    strix::DecodeExecutionOptions execution;
    bool gdn_islands = false;
    bool known_prefix_window = false;
    strix::HipGdnRecurrentLayout gdn_recurrent_layout = strix::HipGdnRecurrentLayout::baseline;
    bool gdn_recurrent_layout_explicit = false;
    bool qsa_hierarchical_selection = false;
    std::uint32_t qsa_gqa_heads_per_block = 1;
    bool qsa_gqa_explicit = false;
    bool prompt_lookup = false;
    std::uint32_t prompt_lookup_depth = 2;
    bool prompt_lookup_depth_explicit = false;
    std::vector<std::string> serving_prefix_files;
    std::uint32_t serving_concurrency = 1;
    std::uint32_t serving_tensor_cohort_rows = 1;
    bool serving_tensor_cohort_explicit = false;
    std::uint64_t serving_memory_budget = 0, serving_request_budget = 0, serving_headroom = 0;
    bool serving_concurrency_explicit = false, serving_memory_explicit = false;
    bool serving_request_explicit = false, serving_headroom_explicit = false;
    strix::GdnRecurrentStorage gdn_recurrent_storage = strix::GdnRecurrentStorage::key_value;
    bool gdn_recurrent_storage_explicit = false;
    std::uint32_t known_prefix_rows = 2;
    bool known_prefix_rows_explicit = false;
    std::uint32_t ple_cpu_helper_threads = 0;
    bool ple_cpu_helper_threads_explicit = false;
    std::vector<std::uint32_t> logprob_token_ids;
    bool teacher_forced_score = false;
    std::string teacher_logits_path;
    bool known_prefix_expert_union = false;
    // Router + combine/inject rows entries in the known-prefix or MTP window.
    bool known_window_router_rows = false;
    std::string offline_tuning_cache_path;
    std::string offline_compile_provenance_path;
    strix::KnownWindowOutputPolicy known_prefix_output_policy = strix::KnownWindowOutputPolicy::all_rows;
    bool known_prefix_output_policy_explicit = false;
    bool prompt_lookup_adaptive = false;
    strix::SessionSnapshotPolicy prompt_lookup_snapshot = strix::SessionSnapshotPolicy::full_components;
    std::uint64_t prompt_lookup_snapshot_bytes = 0;
    bool prompt_lookup_snapshot_explicit = false, prompt_lookup_snapshot_bytes_explicit = false;
    bool device_queue_trace = false;
    std::uint32_t known_prefix_prefill_rows = 0;
    bool known_prefix_prefill_explicit = false;
    std::uint64_t known_prefill_workspace_budget = 0, known_prefill_metadata_budget = 0;
    std::uint64_t known_prefill_uma_budget = 0, known_prefill_host_headroom = 0;
    bool known_prefill_workspace_explicit = false, known_prefill_metadata_explicit = false;
    bool known_prefill_uma_explicit = false, known_prefill_headroom_explicit = false;
    std::uint64_t ple_cache_rows = 0, ple_cache_budget_bytes = 0;
    bool ple_cache_rows_explicit = false, ple_cache_budget_explicit = false;
    std::uint64_t serving_cache_entries = 0, serving_cache_snapshot_bytes = 0;
    std::uint64_t serving_cache_capture_peak_bytes = 0, serving_cache_max_prefix_tokens = 0;
    bool serving_cache_entries_explicit = false, serving_cache_snapshot_explicit = false;
    bool serving_cache_peak_explicit = false, serving_cache_tokens_explicit = false;
    std::uint32_t serving_prefill_rows = 0;
    std::uint64_t serving_prefill_workspace_budget = 0, serving_prefill_metadata_budget = 0;
    bool serving_prefill_explicit = false, serving_prefill_workspace_explicit = false, serving_prefill_metadata_explicit = false;
};

// The wrapper can omit a terminal stateless head without allocating a GPU
// selector. Only the explicit known-prefix state-only route needs that seam
// independently of token-output mode; all other CPU/RAW routes stay unchanged.
[[nodiscard]] inline bool needs_generation_output_dispatcher(const Options& options) noexcept {
    return options.gpu_greedy_token || (options.known_prefix_window &&
        options.known_prefix_output_policy == strix::KnownWindowOutputPolicy::state_only);
}

inline void validate_serving_prefix_cache_limits(const Options& options) {
    const bool any_serving_cache_limit = options.serving_cache_entries_explicit || options.serving_cache_snapshot_explicit ||
        options.serving_cache_peak_explicit || options.serving_cache_tokens_explicit ||
        options.serving_cache_entries || options.serving_cache_snapshot_bytes ||
        options.serving_cache_capture_peak_bytes || options.serving_cache_max_prefix_tokens;
    if (any_serving_cache_limit && (options.serving_prefix_files.empty() ||
        !options.serving_cache_entries_explicit || !options.serving_cache_snapshot_explicit ||
        !options.serving_cache_peak_explicit || !options.serving_cache_tokens_explicit ||
        !options.serving_cache_entries || !options.serving_cache_snapshot_bytes ||
        !options.serving_cache_capture_peak_bytes || !options.serving_cache_max_prefix_tokens ||
        options.serving_cache_entries > std::numeric_limits<std::size_t>::max() ||
        options.serving_cache_max_prefix_tokens > options.context_tokens))
        throw std::invalid_argument("serving prefix cache requires serving files and all four positive explicit bounded limits");
}

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

// Selective backing does not select arithmetic or turn off automatic organization.
inline void validate_model_copy_cli(const Options& o) {
    switch (o.model_copy_tier) {
    case HipModelCopyTier::none:
        if (o.model_copy_payload_bytes)
            throw std::invalid_argument("model copy payload requires a selective model copy tier");
        return;
    case HipModelCopyTier::normalization:
    case HipModelCopyTier::hot_nonexpert: break;
    case HipModelCopyTier::experts:
        if (o.coalesce_gap_bytes)
            throw std::invalid_argument("expert model copy requires --coalesce-gap 0");
        break;
    default: throw std::invalid_argument("unknown model copy tier");
    }
    if (!o.model_copy_payload_bytes || !*o.model_copy_payload_bytes ||
        !o.model_retained_budget || !*o.model_retained_budget ||
        !o.model_setup_peak_budget || !*o.model_setup_peak_budget)
        throw std::invalid_argument("selective model copy requires positive exact payload, retained and setup-peak budgets");
    if (o.residency != HipModelResidencyMode::registered_file_mapping)
        throw std::invalid_argument("selective model copy requires register residency");
    if (o.execution.mode != DecodeExecutionMode::eager)
        throw std::invalid_argument("selective model copy requires eager execution");
    if (o.mtp.enabled || o.mtp.supplied)
        throw std::invalid_argument("selective model copy excludes MTP options");
    if (!o.serving_prefix_files.empty() || o.serving_concurrency != 1 || o.serving_tensor_cohort_rows != 1 ||
        o.serving_prefill_rows || o.serving_concurrency_explicit || o.serving_tensor_cohort_explicit ||
        o.serving_memory_explicit || o.serving_request_explicit || o.serving_headroom_explicit ||
        o.serving_cache_entries_explicit || o.serving_cache_snapshot_explicit || o.serving_cache_peak_explicit ||
        o.serving_cache_tokens_explicit || o.serving_prefill_explicit || o.serving_prefill_workspace_explicit ||
        o.serving_prefill_metadata_explicit)
        throw std::invalid_argument("selective model copy excludes serving options");
    if (o.execution.integer_head != DecodeIntegerHead::none)
        throw std::invalid_argument("selective model copy excludes integer head");
    if (!o.offline_tuning_cache_path.empty() || !o.offline_compile_provenance_path.empty())
        throw std::invalid_argument("selective model copy excludes offline tuning and compile provenance");
    if (o.known_prefix_window || o.known_prefix_rows_explicit || o.known_prefix_output_policy_explicit ||
        o.known_prefix_expert_union || o.known_prefix_prefill_rows || o.known_prefix_prefill_explicit || o.known_prefill_workspace_explicit ||
        o.known_prefill_metadata_explicit || o.known_prefill_uma_explicit || o.known_prefill_headroom_explicit)
        throw std::invalid_argument("selective model copy excludes known-prefix window and prefill");
    if (o.prompt_lookup || o.prompt_lookup_depth_explicit || o.prompt_lookup_adaptive ||
        o.prompt_lookup_snapshot_explicit || o.prompt_lookup_snapshot_bytes_explicit)
        throw std::invalid_argument("selective model copy excludes prompt lookup");
    if (o.prefix_cache_roundtrip)
        throw std::invalid_argument("selective model copy excludes prefix cache roundtrip");
    if (o.gpu_greedy_token)
        throw std::invalid_argument("selective model copy excludes GPU greedy token output");
    if (o.planned_ple_io || o.ple_cpu_helper_threads_explicit || o.ple_cache_rows_explicit || o.ple_cache_budget_explicit)
        throw std::invalid_argument("selective model copy excludes planned PLE IO");
    // Explicit expert backing can compose with the two screened fast recipes.
    // Admission does not qualify their combination; compare each recipe against
    // its own unchanged-backing baseline before any performance adoption.
    const bool expert_recipe = o.model_copy_tier == HipModelCopyTier::experts &&
        (o.numerical_profile == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1 ||
         o.numerical_profile == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1);
    if (o.numerical_profile != DecodeNumericalProfile::exact_v5 && !expert_recipe)
        throw std::invalid_argument("selective model copy requires exact-v5 or an admitted explicit expert recipe");
}

// An experimental layout composes only with an explicit complete manual
// organization; it never changes automatic profile eligibility or defaults.
inline bool explicit_transposed_gdn_islands(const Options& options) {
    return options.gdn_islands && options.gdn_recurrent_layout == HipGdnRecurrentLayout::transposed &&
        options.gdn_recurrent_layout_explicit && options.profile_explicit &&
        options.optimization_profile == OptimizationProfile::manual &&
        options.gdn_recurrent_storage == GdnRecurrentStorage::key_value;
}
inline bool transposed_gdn_island_numerical_recipe(const Options& options) {
    return explicit_transposed_gdn_islands(options) && options.numerical_profile_explicit &&
        (options.numerical_profile == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1 ||
         options.numerical_profile == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1);
}

// Arithmetic overlays never silently replace the selected submission profile.
inline void validate_numerical_profile_cli(const Options& options) {
    if (options.numerical_profile == DecodeNumericalProfile::exact_v5) return;
    if (options.execution.mode != DecodeExecutionMode::eager ||
        options.execution.integer_head != DecodeIntegerHead::none ||
        options.mtp.enabled || options.mtp.supplied || options.known_prefix_window ||
        options.known_prefix_rows_explicit || options.known_prefix_prefill_rows ||
        options.known_prefix_expert_union || options.prompt_lookup ||
        !options.serving_prefix_files.empty() || options.serving_concurrency != 1 ||
        options.serving_tensor_cohort_rows != 1 || options.serving_prefill_rows ||
        options.prefix_cache_roundtrip || options.gpu_greedy_token ||
        options.moe_token_cost_policy || (options.gdn_recurrent_layout != HipGdnRecurrentLayout::baseline &&
            !transposed_gdn_island_numerical_recipe(options)) ||
        options.gdn_recurrent_storage != GdnRecurrentStorage::key_value ||
        options.qsa_hierarchical_selection || options.qsa_gqa_heads_per_block != 1 ||
        options.prefetch_hc_projection_next_tile || options.ranked_moe_router_topk ||
        !options.offline_tuning_cache_path.empty() || !options.offline_compile_provenance_path.empty())
        throw std::invalid_argument("numerical profile requires ordinary eager floating C1 execution without alternate controllers or layouts");
}

// Context can change only after immutable model/profile admission. Roundtrip
// validates the resolved prefix capacity; generation may stop at the existing
// context_capacity boundary before reaching the requested token count.
inline void validate_resolved_prefix_roundtrip(const Options& options) {
    if (!options.prefix_cache_roundtrip) return;
    validate_decode_prefix_roundtrip(options.tokens.size(), options.context_tokens,
                                    options.generate_tokens);
}

// New organizations overlay the resolved base; the base's qualification does
// not qualify the overlay. Manual/fallback modes must supply prerequisites.
inline void validate_resolved_q8_projection_organization(const Options& options) {
    if (options.wave8_q8_gdn_output && !options.virtual_q8_gdn_output)
        throw std::invalid_argument("wave8 Q8 GDN output requires virtual Q8 GDN output");
    if (options.prefetch_q8_lm_head && !options.virtual_q8_lm_head)
        throw std::invalid_argument("prefetch Q8 LM head requires resolved virtual Q8 LM head");
    if (options.wave8_q8_lm_head && !options.prefetch_q8_lm_head)
        throw std::invalid_argument("wave8 Q8 LM head requires explicit prefetch Q8 LM head");
    if (options.wave8_q8_gdn_input && (!options.virtual_q8_gdn_input || !options.prefetch_q8_gdn_input))
        throw std::invalid_argument("wave8 Q8 GDN input requires resolved virtual and prefetch Q8 input");
}

inline void validate_resolved_hc_projection_next_tile(const Options& options) {
    if (!options.prefetch_hc_projection_next_tile) return;
    if (!options.parallel_hyper_mix || !options.staged_hyper_projections || !options.fused_staged_hyper_mix ||
        !options.typed_hyper_projection || !options.prefetch_typed_hyper_rows ||
        options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none)
        throw std::invalid_argument("HC next-tile prefetch requires resolved parallel/staged/fused/typed/row-prefetch eager execution");
}

inline void validate_resolved_q51_slot_parallel_down(const Options& options) {
    if (options.q51_slot_parallel_down &&
        (!options.grouped_exact_q51_down || !options.q4k_a8g32_gate_up ||
         options.execution.mode != strix::DecodeExecutionMode::eager ||
         options.execution.integer_head != strix::DecodeIntegerHead::none))
        throw std::invalid_argument("slot-parallel Q5_1 down requires resolved grouped eager A8 execution");
}

// Explicit per-backend helper cap overlays the resolved PLE choice; never
// selects the planned backend or disables another requested optimization.
inline void validate_resolved_device_queue_trace(const Options& options) {
    if (!options.device_queue_trace) return;
    if (!options.host_dispatch_telemetry || !options.deferred_moe_status ||
        options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none || !options.serving_prefix_files.empty() ||
        options.known_prefix_window || options.prompt_lookup || options.prefix_cache_roundtrip)
        throw std::invalid_argument("device queue trace requires host telemetry and eager D1 C1 without graph, integer, serving, known-prefix, lookup or cache roundtrip");
}

inline void validate_resolved_ple_cache(const Options& options) {
    if (!(options.ple_cache_rows_explicit || options.ple_cache_budget_explicit ||
          options.ple_cache_rows || options.ple_cache_budget_bytes)) return;
    if (!options.ple_cache_rows_explicit || !options.ple_cache_budget_explicit ||
        !options.ple_cache_rows || !options.ple_cache_budget_bytes)
        throw std::invalid_argument("PLE cache requires both positive explicit row and byte limits");
    if (!options.planned_ple_io)
        throw std::invalid_argument("PLE row cache requires resolved planned PLE I/O");
}

inline void validate_resolved_ple_cpu_helpers(const Options& options) {
    if (options.ple_cpu_helper_threads > 8U ||
        (options.ple_cpu_helper_threads_explicit && options.ple_cpu_helper_threads == 0U))
        throw std::invalid_argument("PLE CPU helper threads must be 1 through 8 when explicit");
    if (options.ple_cpu_helper_threads && !options.planned_ple_io)
        throw std::invalid_argument("PLE CPU helper threads require resolved planned PLE I/O");
}

inline void validate_resolved_requested_logprobs(const Options& options, std::uint64_t vocabulary) {
    if (options.logprob_token_ids.empty()) return;
    if (!options.generate_tokens || options.gpu_greedy_token || options.known_prefix_window || options.prompt_lookup ||
        !options.serving_prefix_files.empty() || options.prefix_cache_roundtrip ||
        !options.raw_recording_path.empty() || !options.reference_path.empty())
        throw std::invalid_argument("requested logprobs require serial full-output generation without scalar, window, lookup, serving, cache, RAW or reference modes");
    if (!vocabulary || vocabulary > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("requested logprobs require a uint32 vocabulary");
    if (options.logprob_token_ids.size() > 64U || options.context_tokens > 262144U ||
        options.context_tokens > 1048576U / options.logprob_token_ids.size())
        throw std::invalid_argument("requested logprob receipt exceeds bounded row/value capacity");
    for (const auto id : options.logprob_token_ids)
        if (id >= vocabulary) throw std::invalid_argument("requested logprob token exceeds model vocabulary");
}

inline void validate_teacher_forced_score(const Options& options) {
    if (!options.teacher_logits_path.empty() && !options.teacher_forced_score)
        throw std::invalid_argument("teacher logits file requires teacher-forced score");
    if (!options.teacher_forced_score) return;
    if (options.generate_tokens != 1 || options.tokens.size() < 2 || options.tokens.size() > 262144U ||
        options.execution.mode != DecodeExecutionMode::eager || options.execution.integer_head != DecodeIntegerHead::none ||
        !options.logprob_token_ids.empty() || options.gpu_greedy_token || options.mtp.enabled || options.mtp.supplied ||
        options.known_prefix_window || options.known_prefix_prefill_rows || options.known_prefix_expert_union ||
        options.prompt_lookup || !options.serving_prefix_files.empty() || options.prefix_cache_roundtrip ||
        !options.raw_recording_path.empty() || !options.validation_final_state_path.empty() ||
        !options.reference_path.empty() || options.moe_token_cost_policy)
        throw std::invalid_argument("teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode");
}

inline void validate_resolved_teacher_forced_score(const Options& options, std::uint64_t vocabulary) {
    validate_teacher_forced_score(options);
    if (!options.teacher_forced_score) return;
    if (!vocabulary || vocabulary > std::numeric_limits<std::uint32_t>::max() ||
        options.tokens.size() > options.context_tokens)
        throw std::invalid_argument("teacher-forced score vocabulary or context is invalid");
    for (const auto token : options.tokens)
        if (token >= vocabulary) throw std::invalid_argument("teacher input exceeds model vocabulary");
}

inline void validate_known_prefill_options(const Options& options) {
    const bool any = options.known_prefix_prefill_explicit || options.known_prefix_prefill_rows ||
        options.known_prefill_workspace_explicit || options.known_prefill_metadata_explicit ||
        options.known_prefill_uma_explicit || options.known_prefill_headroom_explicit;
    if (!any) return;
    if (!options.known_prefix_window || !options.known_prefix_prefill_explicit ||
        options.known_prefix_prefill_rows < 16U || options.known_prefix_prefill_rows > 2048U ||
        options.known_prefix_rows_explicit || options.prompt_lookup ||
        !options.known_prefill_workspace_explicit || !options.known_prefill_metadata_explicit ||
        !options.known_prefill_uma_explicit || !options.known_prefill_headroom_explicit ||
        !options.known_prefill_workspace_budget || options.known_prefill_metadata_budget < 2U || !options.known_prefill_uma_budget)
        throw std::invalid_argument("known prefill requires explicit T16..2048, window, all budgets/headroom and no small-row or lookup selector");
    if (options.known_prefix_output_policy == strix::KnownWindowOutputPolicy::state_only && options.generate_tokens)
        throw std::invalid_argument("tiled state-only prefill cannot generate without a terminal output frontier");
}

inline void validate_resolved_known_prefix_window(const Options& options) {
    validate_known_prefill_options(options);
    strix::validate_known_window_output_policy(options.known_prefix_output_policy);
    if (!options.known_prefix_window && (options.known_prefix_output_policy_explicit ||
        options.known_prefix_output_policy != strix::KnownWindowOutputPolicy::all_rows))
        throw std::invalid_argument("known-prefix output policy requires --known-prefix-window");
    if (options.known_prefix_rows != 2 && options.known_prefix_rows != 4 && options.known_prefix_rows != 8)
        throw std::invalid_argument("known-prefix rows must be 2, 4 or 8");
    if (!options.known_prefix_window) {
        if (options.known_prefix_expert_union)
            throw std::invalid_argument("known-prefix expert union requires --known-prefix-window");
        if (options.known_window_router_rows && !options.mtp.window_rows)
            throw std::invalid_argument("known-window router rows require --known-prefix-window or an MTP window");
        if (options.known_prefix_rows_explicit || options.known_prefix_rows != 2)
            throw std::invalid_argument("known-prefix rows require --known-prefix-window");
        return;
    }
    if (options.prompt_lookup) throw std::invalid_argument("known prefix and prompt lookup are separate request routes");
    if (options.known_prefix_expert_union && (!options.q4k_a8g32_gate_up || !options.grouped_exact_q51_down ||
        !options.q51_slot_parallel_down || !options.deferred_moe_status))
        throw std::invalid_argument("known-prefix expert union requires resolved A8/grouped Q5_1/slot-parallel/D1");
    if ((!options.generate_tokens && !(options.known_prefix_prefill_rows &&
          options.known_prefix_output_policy == strix::KnownWindowOutputPolicy::state_only)) ||
        options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none || options.gpu_greedy_token ||
        options.prepared_eager || options.moe_shared_overlap || options.gdn_islands || options.prefix_cache_roundtrip ||
        !options.reference_path.empty() || !options.raw_recording_path.empty() ||
        options.gdn_recurrent_layout != strix::HipGdnRecurrentLayout::baseline ||
        options.gdn_recurrent_storage != strix::GdnRecurrentStorage::key_value ||
        options.qsa_hierarchical_selection || options.qsa_gqa_heads_per_block != 1)
        throw std::invalid_argument("known-prefix window requires eager full-head generation without RAW/reference, alternate submission, cache, islands or candidate layouts");
    if (!options.deferred_moe_status || !options.virtual_q8_gdn_input || !options.prefetch_q8_gdn_input ||
        !options.virtual_q8_gdn_output || !options.wave8_q8_gdn_input || !options.wave8_q8_gdn_output)
        throw std::invalid_argument("known-prefix window requires resolved D1 and explicit paired wave8 GDN input/output");
}

inline void validate_resolved_serving(const Options& options) {
    if ((options.ple_cache_rows_explicit || options.ple_cache_budget_explicit) &&
        (!options.ple_cache_rows_explicit || !options.ple_cache_budget_explicit))
        throw std::invalid_argument("PLE cache requires both positive explicit row and byte limits");
    if (!options.offline_compile_provenance_path.empty() && options.offline_tuning_cache_path.empty())
        throw std::invalid_argument("offline compile provenance requires --offline-tuning-cache");
    validate_known_prefill_options(options);
    validate_serving_prefix_cache_limits(options);
    const bool any_prefill = options.serving_prefill_explicit || options.serving_prefill_workspace_explicit ||
        options.serving_prefill_metadata_explicit || options.serving_prefill_rows || options.serving_prefill_workspace_budget || options.serving_prefill_metadata_budget;
    if (any_prefill && (options.serving_prefix_files.empty() || !options.serving_prefill_explicit ||
        !options.serving_prefill_workspace_explicit || !options.serving_prefill_metadata_explicit ||
        !options.serving_prefill_workspace_budget || !options.serving_prefill_metadata_budget ||
        (options.serving_prefill_rows!=2U && options.serving_prefill_rows!=4U && options.serving_prefill_rows!=8U &&
            (options.serving_prefill_rows<16U || options.serving_prefill_rows>2048U)) || options.serving_tensor_cohort_rows!=1U))
        throw std::invalid_argument("serving prefill requires rows2/4/8 or16..2048, explicit workspace/metadata budgets and independent serving workers");
    if (any_prefill && (options.gdn_recurrent_layout != strix::HipGdnRecurrentLayout::baseline ||
        options.gdn_recurrent_storage != strix::GdnRecurrentStorage::key_value || options.qsa_hierarchical_selection ||
        options.qsa_gqa_heads_per_block != 1 || !options.wave8_q8_gdn_input || !options.wave8_q8_gdn_output ||
        !options.virtual_q8_gdn_input || !options.prefetch_q8_gdn_input || !options.virtual_q8_gdn_output))
        throw std::invalid_argument("serving prefill requires explicit paired wave8 GDN input/output and baseline recurrent/QSA layouts");
    if (options.serving_tensor_cohort_rows != 1 && options.serving_tensor_cohort_rows != 2 && options.serving_tensor_cohort_rows != 4)
        throw std::invalid_argument("serving tensor cohort rows must be1,2or4");
    if (options.serving_tensor_cohort_rows > options.serving_concurrency)
        throw std::invalid_argument("serving tensor cohort exceeds admitted serving concurrency");
    if (options.serving_prefix_files.empty()) {
        if (options.serving_tensor_cohort_explicit || options.serving_tensor_cohort_rows != 1)
            throw std::invalid_argument("tensor cohort requires --serve-prefix-u32le");
        return;
    }
    if (!options.generate_tokens || !options.tokens.empty() || !options.deferred_moe_status ||
        options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none || options.moe_shared_overlap ||
        options.gdn_islands || options.prefix_cache_roundtrip || options.known_prefix_window || options.prompt_lookup ||
        !options.reference_path.empty() || !options.raw_recording_path.empty() || !options.validation_final_state_path.empty() ||
        options.execution_inventory)
        throw std::invalid_argument("serving requires eager D1 generation from prefix files, without positional tokens, graph/integer, overlap/islands, cache, paired/prompt routes or single-session validation outputs");
    if ((options.serving_concurrency != 1 && options.serving_concurrency != 2 && options.serving_concurrency != 4) ||
        options.serving_prefix_files.size() > 64 || !options.serving_memory_budget || !options.serving_request_budget ||
        !options.serving_memory_explicit || !options.serving_request_explicit || !options.serving_headroom_explicit ||
        options.serving_headroom >= options.serving_memory_budget)
        throw std::invalid_argument("serving requires concurrency 1/2/4, at most64 requests and explicit total/request/headroom reservation bytes");
}

inline void validate_moe_token_cost_options(const Options& options) {
    if (!options.moe_token_cost_policy) return;
    if (!options.generate_tokens || !options.deferred_moe_status || !options.moe_shared_overlap ||
        options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none ||
        !options.serving_prefix_files.empty() || options.prompt_lookup || options.known_prefix_window ||
        options.prefix_cache_roundtrip || !options.raw_recording_path.empty() || !options.reference_path.empty())
        throw std::invalid_argument("observed MoE token policy requires eager generation, D1 and explicit overlap without serving, lookup, known-window, RAW, reference, roundtrip or integer modes");
}
inline void validate_prompt_lookup_snapshot_options(const Options& options) {
    strix::validate_session_snapshot_policy(options.prompt_lookup_snapshot);
    const bool selected = options.prompt_lookup_snapshot_explicit || options.prompt_lookup_snapshot_bytes_explicit ||
        options.prompt_lookup_snapshot != strix::SessionSnapshotPolicy::full_components || options.prompt_lookup_snapshot_bytes;
    if (!selected) return;
    if (!options.prompt_lookup || options.serving_prefix_files.size() || options.known_prefix_window)
        throw std::invalid_argument("snapshot policy is an explicit prompt-lookup-only option");
    if (options.prompt_lookup_snapshot == strix::SessionSnapshotPolicy::occupied_kv_v1) {
        if (!options.prompt_lookup_snapshot_explicit || !options.prompt_lookup_snapshot_bytes_explicit || !options.prompt_lookup_snapshot_bytes)
            throw std::invalid_argument("occupied KV snapshot requires explicit positive payload budget");
    } else if (options.prompt_lookup_snapshot_bytes_explicit || options.prompt_lookup_snapshot_bytes)
        throw std::invalid_argument("snapshot byte limit is reserved for explicit occupied-KV policy");
}
inline void validate_resolved_prompt_lookup(const Options& options) {
    validate_prompt_lookup_snapshot_options(options);
    if (options.prompt_lookup_adaptive && (!options.prompt_lookup || options.prompt_lookup_depth_explicit))
        throw std::invalid_argument("adaptive prompt lookup requires --prompt-lookup without an explicit fixed depth");
    if (!options.prompt_lookup) return;
    if (options.known_prefix_window || options.prompt_lookup_depth > 2 ||
        !options.generate_tokens || options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none || options.gpu_greedy_token ||
        options.prepared_eager || options.moe_shared_overlap || options.gdn_islands || options.prefix_cache_roundtrip ||
        !options.reference_path.empty() || !options.raw_recording_path.empty() ||
        options.gdn_recurrent_layout != strix::HipGdnRecurrentLayout::baseline ||
        options.gdn_recurrent_storage != strix::GdnRecurrentStorage::key_value ||
        options.qsa_hierarchical_selection || options.qsa_gqa_heads_per_block != 1)
        throw std::invalid_argument("prompt lookup requires eager generation with its own device selectors, without alternate submission/cache/RAW/reference/candidate layouts");
    if (!options.deferred_moe_status || !options.virtual_q8_gdn_input || !options.prefetch_q8_gdn_input ||
        !options.virtual_q8_gdn_output || !options.wave8_q8_gdn_input || !options.wave8_q8_gdn_output)
        throw std::invalid_argument("prompt lookup requires resolved D1 and paired wave8 GDN input/output");
}

inline void validate_resolved_qsa_gqa(const Options& options) {
    if (options.qsa_gqa_heads_per_block != 1 && options.qsa_gqa_heads_per_block != 2 && options.qsa_gqa_heads_per_block != 4)
        throw std::invalid_argument("QSA heads per block must be 1, 2 or 4");
    if (options.qsa_gqa_heads_per_block != 1 &&
        (!options.deferred_moe_status || options.execution.mode != strix::DecodeExecutionMode::eager ||
         options.execution.integer_head != strix::DecodeIntegerHead::none || options.prefix_cache_roundtrip ||
         options.context_tokens > 262144))
        throw std::invalid_argument("QSA grouped attention requires eager D1 without integer head or roundtrip and context<=262144");
}

inline void validate_resolved_qsa_hierarchical_selection(const Options& options) {
    if (!options.qsa_hierarchical_selection) return;
    if (!options.deferred_moe_status || options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none || options.prefix_cache_roundtrip ||
        options.context_tokens > 131072)
        throw std::invalid_argument("hierarchical QSA requires eager D1 without integer head or roundtrip and context<=131072");
}

inline void validate_resolved_gdn_recurrent_storage(const Options& options) {
    strix::validate_gdn_recurrent_storage(options.gdn_recurrent_storage);
    if (options.gdn_recurrent_storage != strix::GdnRecurrentStorage::key_value &&
        (options.execution.mode != strix::DecodeExecutionMode::eager ||
         options.execution.integer_head != strix::DecodeIntegerHead::none || options.gdn_islands ||
         options.known_prefix_window || options.prompt_lookup))
        throw std::invalid_argument("GDN value-key storage requires eager C1 without graph/integer, islands or R2 routes");
}

inline void validate_resolved_gdn_recurrent_layout(const Options& options) {
    (void)strix::gdn_recurrent_layout_marker(options.gdn_recurrent_layout);
    if (options.gdn_recurrent_layout != strix::HipGdnRecurrentLayout::baseline &&
        (options.execution.mode != strix::DecodeExecutionMode::eager ||
         options.execution.integer_head != strix::DecodeIntegerHead::none ||
         (options.gdn_islands && !gdn_island_recurrent_layout_admitted(options.gdn_recurrent_layout))))
        throw std::invalid_argument("GDN recurrent LDS layout requires eager floating execution with an admitted island layout");
    if (options.gdn_islands && options.gdn_recurrent_layout == HipGdnRecurrentLayout::transposed &&
        (!explicit_transposed_gdn_islands(options) || options.mtp.enabled || options.mtp.supplied ||
         options.known_prefix_window || options.known_prefix_rows_explicit || options.known_prefix_prefill_rows ||
         options.known_prefix_expert_union || options.prompt_lookup || !options.serving_prefix_files.empty() ||
         options.serving_concurrency_explicit || options.serving_tensor_cohort_explicit ||
         options.serving_memory_explicit || options.serving_request_explicit || options.serving_headroom_explicit ||
         options.serving_cache_entries_explicit || options.serving_cache_snapshot_explicit ||
         options.serving_cache_peak_explicit || options.serving_cache_tokens_explicit ||
         options.serving_prefill_rows || options.serving_prefill_explicit ||
         options.serving_prefill_workspace_explicit || options.serving_prefill_metadata_explicit))
        throw std::invalid_argument("transposed GDN islands require explicit manual ordinary eager C1 execution");
}

inline void validate_resolved_gdn_islands(const Options& options) {
    if (!options.gdn_islands) return;
    validate_resolved_gdn_recurrent_layout(options);
    if (options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none || options.prefix_cache_roundtrip ||
        !gdn_island_recurrent_layout_admitted(options.gdn_recurrent_layout) ||
        options.gdn_recurrent_storage != strix::GdnRecurrentStorage::key_value ||
        (options.wave8_q8_gdn_input != options.wave8_q8_gdn_output))
        throw std::invalid_argument("GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    if (!options.deferred_moe_status || !options.virtual_q8_gdn_input ||
        !options.prefetch_q8_gdn_input || !options.virtual_q8_gdn_output)
        throw std::invalid_argument("GDN islands require resolved D1 and virtual/prefetch input with virtual output");
}

inline void validate_mtp_cli(const Options& o,bool resolved=false) {
    const auto& m=o.mtp;
    if(!m.enabled){if(m.supplied)throw std::invalid_argument("MTP parameters require --mtp");return;}
    if(o.execution.mode!=DecodeExecutionMode::eager || o.execution.integer_head!=DecodeIntegerHead::none ||
       o.prepared_eager || o.moe_shared_overlap || o.moe_token_cost_policy || o.gdn_islands || o.gpu_greedy_token ||
       o.prefix_cache_roundtrip || o.prompt_lookup || o.known_prefix_window || !o.serving_prefix_files.empty() ||
       o.device_queue_trace || !o.reference_path.empty() || !o.raw_recording_path.empty() ||
       !o.validation_final_state_path.empty() || !o.logprob_token_ids.empty())
        throw std::invalid_argument("MTP requires its explicit raw eager target route without graph/prepared/overlap/islands/other controllers/recording");
    if(o.tokens.size()<3U || o.tokens.size()>4097U || !o.generate_tokens || o.generate_tokens>UINT32_MAX ||
       ((resolved || o.context_explicit) && (o.tokens.size()>o.context_tokens ||
        o.generate_tokens>o.context_tokens-o.tokens.size()+1U)))
        throw std::invalid_argument("MTP requires 3..4097 known prompt tokens, positive uint32 output budget and bounded target context");
    if(m.metadata_only && (!m.golden.empty() || !m.golden_sha256.empty()))
        throw std::invalid_argument("MTP metadata diagnostic does not accept unused golden options");
    if(m.sidecar.empty() || m.sidecar_logical_name.empty() || (!m.metadata_only && m.golden.empty()) || m.repository.empty() || m.revision.empty() || m.source_path.empty() || m.converter_revision.empty() ||
       !mtp_sha256_text(m.sidecar_sha256) || !mtp_sha256_text(m.target_sha256) || !mtp_sha256_text(m.target_content) ||
       (!m.metadata_only && !mtp_sha256_text(m.golden_sha256)) || !mtp_sha256_text(m.converter_sha256) || !mtp_sha256_text(m.norm_converter_sha256))
        throw std::invalid_argument("MTP requires explicit local assets, full SHA256 pins and source/conversion provenance");
    if(m.conversion!="effective-norm-embedding-first-v1" ||
       (m.normalized_rounding!="fp32" && m.normalized_rounding!="bf16-rne") ||
       (m.projection_rounding!="fp32" && m.projection_rounding!="bf16-rne") ||
       (m.sidecar_residency!="register" && m.sidecar_residency!="copy"))
        throw std::invalid_argument("MTP requires explicit admitted conversion, both cast boundaries and sidecar residency");
    if(!m.sidecar_bytes || !m.header_bytes || m.header_bytes>32ULL*1024ULL*1024ULL || !m.sidecar_retained_bytes || !m.sidecar_setup_bytes ||
       !m.golden_max_bytes || !m.replay_host_bytes || !m.runtime_bytes || !m.target_capture_bytes || !m.policy_identity ||
       m.candidate_depth<1U || m.candidate_depth>8U || !std::isfinite(m.max_hidden_error) || m.max_hidden_error<0.0 ||
       !std::isfinite(m.max_logit_error) || m.max_logit_error<0.0)
        throw std::invalid_argument("MTP requires positive explicit asset/replay/runtime/capture budgets, policy identity and depth1..8");
    const bool any_window=m.window_rows || m.window_arena_bytes || m.window_capture_bytes || m.window_checkpoint_bytes || m.window_total_bytes || m.window_history_intervals;
    if(any_window && ((m.window_rows!=2U && m.window_rows!=4U && m.window_rows!=8U) || !m.window_arena_bytes ||
       !m.window_capture_bytes || !m.window_checkpoint_bytes || !m.window_total_bytes || !m.window_history_intervals))
        throw std::invalid_argument("MTP closed target windows require explicit rows2/4/8 and all five positive window budgets");
    if((m.window_expert_union || m.window_expert_union_bytes_explicit || m.window_expert_union_bytes) &&
       (!m.window_expert_union || !m.window_expert_union_bytes_explicit || !m.window_expert_union_bytes ||
        !m.window_rows || m.metadata_only || m.window_expert_union_bytes>m.window_arena_bytes ||
        m.window_expert_union_bytes>m.window_total_bytes))
        throw std::invalid_argument("MTP window expert union requires an actual target window and a positive explicit budget within arena and total window budgets");
    if(m.window_rows && m.candidate_depth<m.window_rows-1U)
        throw std::invalid_argument("MTP target window requires candidate depth at least rows minus one");
    if(m.window_rows && (o.gdn_recurrent_layout!=HipGdnRecurrentLayout::baseline || o.gdn_recurrent_storage!=GdnRecurrentStorage::key_value ||
       o.qsa_hierarchical_selection || o.qsa_gqa_heads_per_block!=1U))
        throw std::invalid_argument("MTP closed target windows require baseline recurrent and QSA organizations");
}
inline void validate_resolved_mtp_cli(const Options& o) {
    validate_mtp_cli(o,true);
    if(o.mtp.enabled && o.mtp.window_rows && (!o.deferred_moe_status || !o.virtual_q8_gdn_input || !o.prefetch_q8_gdn_input ||
       !o.virtual_q8_gdn_output || !o.wave8_q8_gdn_input || !o.wave8_q8_gdn_output))
        throw std::invalid_argument("MTP target windows require resolved D1 and explicit paired wave8 GDN configuration");
    if(o.mtp.enabled && o.mtp.window_expert_union && (!o.q4k_a8g32_gate_up || !o.grouped_exact_q51_down ||
       !o.q51_slot_parallel_down || !o.deferred_moe_status))
        throw std::invalid_argument("MTP window expert union requires resolved A8, grouped Q5_1, slot-parallel down and D1");
}

inline Options parse_options(
    int argc,
    char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: strix-hip-decode-token MODEL.gguf [TOKEN_ID ...] "
            "[--mtp --mtp-sidecar FILE --mtp-sidecar-sha256 HEX --mtp-golden FILE --mtp-golden-sha256 HEX [--mtp-window-expert-union --mtp-window-expert-union-bytes N] (explicit source/conversion/cast/budget options required)] "
            "[--optimization-profile auto|manual] [--execution-inventory] [--host-dispatch-telemetry] [--execution-telemetry-trace] [--device-queue-trace] "
            "[--numerical-profile exact-v5|hc-splitk-f32-v1|q51-a8-down-v1|hc-splitk-f32+q51-a8-down-v1|q51-q80-a8-down-v1|hc-splitk-f32+q51-q80-a8-down-v1|hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1|hc-deferred-norm-f32+q51-q80-a8-down-v1|hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1] "
            "[--teacher-forced-score --generate 1 [--teacher-logits-file PATH]] "
            "[--gpu-greedy-token] [--logprob-token ID ...] [--offline-tuning-cache FILE [--offline-compile-provenance FILE]] [--gdn-recurrent-layout baseline|stride32|stride33|transposed] [--gdn-recurrent-storage key-value|value-key] [--known-prefix-window --known-prefix-rows 2|4|8 --known-prefix-expert-union] [--known-window-router-rows] [--known-prefix-prefill-rows 16..2048 --known-prefix-workspace-budget-bytes N --known-prefix-host-metadata-budget-bytes N --known-prefix-uma-budget-bytes N --known-prefix-host-headroom-bytes N] [--prompt-lookup [--prompt-lookup-depth 0|1|2 | --prompt-lookup-adaptive] [--prompt-lookup-snapshot occupied-kv-v1 --prompt-lookup-snapshot-bytes N]] [--serve-prefix-u32le FILE ... --serve-concurrency 1|2|4 [--serve-tensor-cohort-rows 2|4 | --serve-prefill-rows 2|4|8|16..2048 --serve-prefill-workspace-budget-bytes N --serve-prefill-host-metadata-budget-bytes N] --serve-memory-budget-bytes N --serve-request-budget-bytes N --serve-headroom-bytes N] [--gdn-islands] [--qsa-hierarchical-selection] [--qsa-gqa-heads-per-block 1|2|4] [--prepared-eager] [--moe-shared-overlap [--moe-token-cost-policy]] [--moe-overlap-arena-budget-bytes N] [--batched-moe-activation] [--validation-final-state DIR] [--validation-final-state-max-bytes N] [--reference FILE] [--max-logit-error X] "
            "[--context N] [--residency register|copy] [--coalesce-gap BYTES] [--model-copy-tier none|normalization|hot_nonexpert|experts --model-copy-payload-bytes N --model-retained-budget N --model-setup-peak-budget N] "
            "[--planned-ple-io] [--ple-cpu-helper-threads 1..8] [--ple-cache-rows N --ple-cache-budget-bytes N] [--ple-max-rows-per-io N] [--ple-max-inflight-ios N] "
            "[--generate N] [--eos-token TOKEN_ID] [--prefix-cache-roundtrip] "
            "[--parallel-moe-router] [--ranked-moe-router-topk | --wave-moe-router-topk] "
            "[--prefetch-q8-qsa-input] [--prefetch-q8-shared-expert] [--prefetch-q8-lm-head [--wave8-q8-lm-head]] [--wave8-q8-gdn-input] [--wave8-q8-gdn-output] [--wave8-q8-qsa-output] [--virtual-q8-lm-head] [--virtual-q8-gdn-input [--prefetch-q8-gdn-input]] [--q4k-a8g32-gate-up [--grouped-exact-q51-down]] [--deferred-moe-status] "
            "[--parallel-hyper-mix [--staged-hyper-projections [--fused-staged-hyper-mix [--typed-fused-hyper-projections [--prefetch-typed-hyper-rows] [--prefetch-q8-hyper-output [--lane-chain-hyper-output]]]]]]");
    }

    Options options;
    options.model_path =
        argv[1];
    std::set<std::string_view> execution_option_names;
    std::set<std::string> mtp_option_names;

    for (int index = 2;
         index < argc;
         ++index) {
        const std::string_view argument =
            argv[index];
        options.any_low_level_flag = options.any_low_level_flag ||
            is_low_level_optimization_option(argument);
        if (argument == "--model-copy-tier" || argument.starts_with("--model-copy-tier=") ||
            argument == "--model-copy-payload-bytes" || argument.starts_with("--model-copy-payload-bytes=") ||
            argument == "--model-retained-budget" || argument.starts_with("--model-retained-budget=") ||
            argument == "--model-setup-peak-budget" || argument.starts_with("--model-setup-peak-budget=")) {
            const auto equal = argument.find('=');
            const auto key = argument.substr(0, equal);
            const auto value = equal != std::string_view::npos ? argument.substr(equal + 1) :
                (index + 1 < argc ? std::string_view(argv[++index]) : std::string_view{});
            if (value.empty() || value.starts_with("--"))
                throw std::invalid_argument("model copy option requires a value");
            if (key == "--model-copy-tier") {
                if (options.model_copy_tier_explicit) throw std::invalid_argument("duplicate model copy tier option");
                options.model_copy_tier_explicit = true;
                if (value == "none") options.model_copy_tier = strix::HipModelCopyTier::none;
                else if (value == "normalization") options.model_copy_tier = strix::HipModelCopyTier::normalization;
                else if (value == "hot_nonexpert") options.model_copy_tier = strix::HipModelCopyTier::hot_nonexpert;
                else if (value == "experts") options.model_copy_tier = strix::HipModelCopyTier::experts;
                else throw std::invalid_argument("model copy tier must be none, normalization, hot_nonexpert or experts");
            } else {
                auto* target = key == "--model-copy-payload-bytes" ? &options.model_copy_payload_bytes :
                    key == "--model-retained-budget" ? &options.model_retained_budget : &options.model_setup_peak_budget;
                if (*target) throw std::invalid_argument("duplicate model copy budget option");
                const auto bytes = parse_u64(value, "model copy budget");
                if (!bytes) throw std::invalid_argument("model copy budgets must be positive");
                *target = bytes;
            }
            continue;
        }
        if (argument == "--teacher-logits-file" || argument.starts_with("--teacher-logits-file=")) {
            if (!options.teacher_logits_path.empty())
                throw std::invalid_argument("duplicate teacher logits file option");
            const auto value = argument == "--teacher-logits-file" ?
                (index + 1 < argc ? std::string_view(argv[++index]) : std::string_view{}) :
                argument.substr(std::string_view("--teacher-logits-file=").size());
            if (value.empty() || value.starts_with("--"))
                throw std::invalid_argument("teacher logits file requires a nonempty path");
            options.teacher_logits_path = value;
            continue;
        }
        if (argument == "--teacher-forced-score") {
            if (options.teacher_forced_score)
                throw std::invalid_argument("duplicate teacher-forced score option");
            options.teacher_forced_score = true;
            continue;
        }
        if (argument == "--numerical-profile" || argument.starts_with("--numerical-profile=")) {
            if (options.numerical_profile_explicit)
                throw std::invalid_argument("duplicate numerical profile option");
            std::string_view value;
            if (argument == "--numerical-profile") {
                if (index + 1 >= argc)
                    throw std::invalid_argument("--numerical-profile requires a supported numerical recipe");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--numerical-profile=").size());
            const auto parsed = strix::parse_decode_numerical_profile(value);
            if (!parsed)
                throw std::invalid_argument("--numerical-profile requires a supported numerical recipe");
            options.numerical_profile = *parsed;
            options.numerical_profile_explicit = true;
            continue;
        }
        if(argument=="--mtp") {
            if(options.mtp.enabled)throw std::invalid_argument("duplicate MTP route option");
            options.mtp.enabled=true;continue;
        }
        if(argument=="--mtp-window-expert-union") {
            if(!mtp_option_names.insert(std::string(argument)).second)throw std::invalid_argument("duplicate MTP option");
            options.mtp.supplied=true;options.mtp.window_expert_union=true;continue;
        }
        if(argument.starts_with("--mtp-window-expert-union="))
            throw std::invalid_argument("MTP window expert union is a flag without a value");
        if(argument.starts_with("--mtp-")) {
            const auto equal=argument.find('=');const auto key=argument.substr(0,equal);
            if(!mtp_option_names.insert(std::string(key)).second)throw std::invalid_argument("duplicate MTP option");
            std::string value;
            if(equal!=std::string_view::npos)value=argument.substr(equal+1U);
            else {if(index+1>=argc)throw std::invalid_argument("MTP option requires a value");value=argv[++index];}
            if(value.empty())throw std::invalid_argument("MTP option value is empty");
            auto& m=options.mtp;m.supplied=true;
            if(key=="--mtp-sidecar")m.sidecar=value;
            else if(key=="--mtp-sidecar-logical-name")m.sidecar_logical_name=value;
            else if(key=="--mtp-sidecar-sha256")m.sidecar_sha256=value;
            else if(key=="--mtp-target-sha256")m.target_sha256=value;
            else if(key=="--mtp-target-content")m.target_content=value;
            else if(key=="--mtp-golden")m.golden=value;
            else if(key=="--mtp-golden-sha256")m.golden_sha256=value;
            else if(key=="--mtp-source-repository")m.repository=value;
            else if(key=="--mtp-source-revision")m.revision=value;
            else if(key=="--mtp-source-path")m.source_path=value;
            else if(key=="--mtp-converter-sha256")m.converter_sha256=value;
            else if(key=="--mtp-norm-converter-sha256")m.norm_converter_sha256=value;
            else if(key=="--mtp-converter-revision")m.converter_revision=value;
            else if(key=="--mtp-conversion")m.conversion=value;
            else if(key=="--mtp-normalized-rounding")m.normalized_rounding=value;
            else if(key=="--mtp-projection-rounding")m.projection_rounding=value;
            else if(key=="--mtp-sidecar-residency")m.sidecar_residency=value;
            else if(key=="--mtp-max-hidden-error")m.max_hidden_error=parse_double(value,"MTP hidden policy");
            else if(key=="--mtp-max-logit-error")m.max_logit_error=parse_double(value,"MTP logit policy");
            else {
                const auto n=parse_u64(value,"MTP option");
                if(key=="--mtp-metadata-only"){
                    if(n!=1U)throw std::invalid_argument("MTP metadata-only requires value1");
                    m.metadata_only=true;
                }
                else if(key=="--mtp-sidecar-bytes")m.sidecar_bytes=n;
                else if(key=="--mtp-header-bytes")m.header_bytes=n;
                else if(key=="--mtp-sidecar-retained-bytes")m.sidecar_retained_bytes=n;
                else if(key=="--mtp-sidecar-setup-bytes")m.sidecar_setup_bytes=n;
                else if(key=="--mtp-golden-max-bytes")m.golden_max_bytes=n;
                else if(key=="--mtp-replay-host-bytes")m.replay_host_bytes=n;
                else if(key=="--mtp-runtime-bytes")m.runtime_bytes=n;
                else if(key=="--mtp-target-capture-bytes")m.target_capture_bytes=n;
                else if(key=="--mtp-policy-id")m.policy_identity=n;
                else if(key=="--mtp-window-arena-bytes")m.window_arena_bytes=n;
                else if(key=="--mtp-window-capture-bytes")m.window_capture_bytes=n;
                else if(key=="--mtp-window-checkpoint-bytes")m.window_checkpoint_bytes=n;
                else if(key=="--mtp-window-total-bytes")m.window_total_bytes=n;
                else if(key=="--mtp-window-history-intervals")m.window_history_intervals=n;
                else if(key=="--mtp-window-expert-union-bytes"){m.window_expert_union_bytes=n;m.window_expert_union_bytes_explicit=true;}
                else if(key=="--mtp-candidate-depth" || key=="--mtp-target-window-rows") {
                    if(n>UINT32_MAX)throw std::invalid_argument("MTP depth/rows exceeds uint32");
                    if(key=="--mtp-candidate-depth")m.candidate_depth=static_cast<std::uint32_t>(n);else m.window_rows=static_cast<std::uint32_t>(n);
                } else throw std::invalid_argument("unknown MTP option: "+std::string(key));
            }
            continue;
        }
        if (argument == "--prefetch-hc-projection-next-tile") {
            if (options.prefetch_hc_projection_next_tile)
                throw std::invalid_argument("duplicate HC next-tile prefetch option");
            options.prefetch_hc_projection_next_tile = true;
            continue;
        }
        if (argument == "--prefetch-q8-lm-head" || argument == "--wave8-q8-lm-head" ||
            argument == "--wave8-q8-gdn-input" || argument == "--wave8-q8-gdn-output" ||
            argument == "--wave8-q8-qsa-output") {
            auto& enabled = argument == "--prefetch-q8-lm-head" ? options.prefetch_q8_lm_head
                : argument == "--wave8-q8-lm-head" ? options.wave8_q8_lm_head
                : argument == "--wave8-q8-gdn-input" ? options.wave8_q8_gdn_input
                : argument == "--wave8-q8-gdn-output" ? options.wave8_q8_gdn_output
                : options.wave8_q8_qsa_output;
            if (enabled) throw std::invalid_argument("duplicate Q8 projection organization option");
            enabled = true;
            continue;
        }
        if (argument == "--serve-prefix-u32le") {
            if (index + 1 >= argc || options.serving_prefix_files.size() >= 64)
                throw std::invalid_argument("serving prefix requires a file and at most64 requests");
            options.serving_prefix_files.emplace_back(argv[++index]);
            continue;
        }
        if (argument == "--serve-prefix-cache-entries" || argument == "--serve-prefix-cache-snapshot-bytes" ||
            argument == "--serve-prefix-cache-capture-peak-bytes" || argument == "--serve-prefix-cache-max-tokens") {
            if (index + 1 >= argc) throw std::invalid_argument("serving prefix cache option requires a value");
            const auto value = parse_u64(argv[++index], "serving prefix cache");
            bool* supplied = argument == "--serve-prefix-cache-entries" ? &options.serving_cache_entries_explicit :
                argument == "--serve-prefix-cache-snapshot-bytes" ? &options.serving_cache_snapshot_explicit :
                argument == "--serve-prefix-cache-capture-peak-bytes" ? &options.serving_cache_peak_explicit :
                &options.serving_cache_tokens_explicit;
            if (*supplied) throw std::invalid_argument("duplicate serving prefix cache option");
            if (!value) throw std::invalid_argument("serving prefix cache limits must be positive");
            *supplied = true;
            if (argument == "--serve-prefix-cache-entries") options.serving_cache_entries = value;
            else if (argument == "--serve-prefix-cache-snapshot-bytes") options.serving_cache_snapshot_bytes = value;
            else if (argument == "--serve-prefix-cache-capture-peak-bytes") options.serving_cache_capture_peak_bytes = value;
            else options.serving_cache_max_prefix_tokens = value;
            continue;
        }
        if (argument == "--serve-prefill-rows" || argument.starts_with("--serve-prefill-rows=") ||
            argument == "--serve-prefill-workspace-budget-bytes" || argument == "--serve-prefill-host-metadata-budget-bytes") {
            std::string_view value;
            const bool row_option=argument=="--serve-prefill-rows" || argument.starts_with("--serve-prefill-rows=");
            bool* supplied=row_option?&options.serving_prefill_explicit:
                argument=="--serve-prefill-workspace-budget-bytes"?&options.serving_prefill_workspace_explicit:&options.serving_prefill_metadata_explicit;
            if(*supplied) throw std::invalid_argument("duplicate serving prefill option");
            if(argument.starts_with("--serve-prefill-rows=")) value=argument.substr(std::string_view("--serve-prefill-rows=").size());
            else {if(index+1>=argc) throw std::invalid_argument("serving prefill option requires a value");value=argv[++index];}
            const auto number=parse_u64(value,"serving prefill");
            if(!number) throw std::invalid_argument("serving prefill values must be positive");
            *supplied=true;
            if(row_option) {
                if(number!=2U && number!=4U && number!=8U && (number<16U || number>2048U))
                    throw std::invalid_argument("serving prefill rows must be2/4/8 or16..2048");
                options.serving_prefill_rows=static_cast<std::uint32_t>(number);
            } else if(argument=="--serve-prefill-workspace-budget-bytes") options.serving_prefill_workspace_budget=number;
            else options.serving_prefill_metadata_budget=number;
            continue;
        }
        if (argument == "--serve-tensor-cohort-rows" || argument.starts_with("--serve-tensor-cohort-rows=")) {
            if (options.serving_tensor_cohort_explicit) throw std::invalid_argument("duplicate serving tensor cohort selector");
            options.serving_tensor_cohort_explicit = true;
            std::string_view value;
            if (argument == "--serve-tensor-cohort-rows") {
                if (index + 1 >= argc) throw std::invalid_argument("serving tensor cohort requires rows2or4");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--serve-tensor-cohort-rows=").size());
            const auto rows = parse_u64(value, "serving tensor cohort rows");
            if (rows != 2 && rows != 4) throw std::invalid_argument("serving tensor cohort rows must be2or4");
            options.serving_tensor_cohort_rows = static_cast<std::uint32_t>(rows);
            continue;
        }
        if (argument == "--serve-concurrency" || argument == "--serve-memory-budget-bytes" ||
            argument == "--serve-request-budget-bytes" || argument == "--serve-headroom-bytes") {
            if (index + 1 >= argc) throw std::invalid_argument("serving option requires a value");
            const auto value = parse_u64(argv[++index], "serving option");
            bool* explicit_value = argument == "--serve-concurrency" ? &options.serving_concurrency_explicit :
                argument == "--serve-memory-budget-bytes" ? &options.serving_memory_explicit :
                argument == "--serve-request-budget-bytes" ? &options.serving_request_explicit : &options.serving_headroom_explicit;
            if (*explicit_value) throw std::invalid_argument("duplicate serving option");
            *explicit_value = true;
            if (argument == "--serve-concurrency") {
                if (value != 1 && value != 2 && value != 4) throw std::invalid_argument("serving concurrency must be1,2or4");
                options.serving_concurrency = static_cast<std::uint32_t>(value);
            } else if (argument == "--serve-memory-budget-bytes") options.serving_memory_budget = value;
            else if (argument == "--serve-request-budget-bytes") options.serving_request_budget = value;
            else options.serving_headroom = value;
            continue;
        }
        if (argument == "--prompt-lookup") {
            if (options.prompt_lookup) throw std::invalid_argument("duplicate prompt lookup option");
            options.prompt_lookup = true;
            continue;
        }
        if (argument == "--device-queue-trace") {
            if (options.device_queue_trace) throw std::invalid_argument("duplicate device queue trace option");
            options.device_queue_trace = true;
            continue;
        }
        if (argument == "--prompt-lookup-snapshot" || argument.starts_with("--prompt-lookup-snapshot=")) {
            if (options.prompt_lookup_snapshot_explicit) throw std::invalid_argument("duplicate prompt lookup snapshot policy");
            options.prompt_lookup_snapshot_explicit = true;
            std::string_view value;
            if (argument == "--prompt-lookup-snapshot") {
                if (index + 1 >= argc) throw std::invalid_argument("snapshot policy requires a value");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--prompt-lookup-snapshot=").size());
            if (value == "full-components") options.prompt_lookup_snapshot = strix::SessionSnapshotPolicy::full_components;
            else if (value == "occupied-kv-v1") options.prompt_lookup_snapshot = strix::SessionSnapshotPolicy::occupied_kv_v1;
            else throw std::invalid_argument("snapshot policy must be full-components or occupied-kv-v1");
            continue;
        }
        if (argument == "--prompt-lookup-snapshot-bytes") {
            if (options.prompt_lookup_snapshot_bytes_explicit || index + 1 >= argc)
                throw std::invalid_argument("snapshot payload budget requires exactly one positive value");
            options.prompt_lookup_snapshot_bytes_explicit = true;
            options.prompt_lookup_snapshot_bytes = parse_u64(argv[++index], "snapshot payload budget");
            if (!options.prompt_lookup_snapshot_bytes) throw std::invalid_argument("snapshot payload budget must be positive");
            continue;
        }
        if (argument == "--prompt-lookup-adaptive") {
            if (options.prompt_lookup_adaptive) throw std::invalid_argument("duplicate prompt lookup adaptive option");
            options.prompt_lookup_adaptive = true;
            continue;
        }
        if (argument == "--prompt-lookup-depth") {
            if (options.prompt_lookup_depth_explicit || index + 1 >= argc)
                throw std::invalid_argument("prompt lookup depth requires one value zero, one or two");
            const auto value = parse_u64(argv[++index], "prompt lookup depth");
            if (value > 2) throw std::invalid_argument("prompt lookup depth must be zero, one or two");
            options.prompt_lookup_depth = static_cast<std::uint32_t>(value);
            options.prompt_lookup_depth_explicit = true;
            continue;
        }
        if (argument == "--offline-compile-provenance") {
            if (!options.offline_compile_provenance_path.empty()) throw std::invalid_argument("duplicate offline compile provenance option");
            if (index + 1 >= argc || !*argv[index+1]) throw std::invalid_argument("offline compile provenance requires a file");
            options.offline_compile_provenance_path = argv[++index];
            continue;
        }
        if (argument == "--offline-tuning-cache") {
            if (!options.offline_tuning_cache_path.empty()) throw std::invalid_argument("duplicate offline tuning cache option");
            if (index + 1 >= argc || !*argv[index + 1]) throw std::invalid_argument("offline tuning cache requires a file");
            options.offline_tuning_cache_path = argv[++index];
            continue;
        }
        if (argument == "--logprob-token") {
            if (index + 1 >= argc) throw std::invalid_argument("logprob token requires an ID");
            if (options.logprob_token_ids.size() == 64U) throw std::invalid_argument("requested logprob ID list exceeds 64 entries");
            const auto id = parse_u64(argv[++index], "logprob token");
            if (id > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("logprob token exceeds uint32");
            options.logprob_token_ids.push_back(static_cast<std::uint32_t>(id));
            continue;
        }
        if (argument == "--known-prefix-prefill-rows" || argument.starts_with("--known-prefix-prefill-rows=")) {
            if (options.known_prefix_prefill_explicit) throw std::invalid_argument("duplicate known prefill rows option");
            std::string_view value;
            if (argument == "--known-prefix-prefill-rows") {
                if (index + 1 >= argc) throw std::invalid_argument("known prefill rows require a value");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--known-prefix-prefill-rows=").size());
            const auto rows = parse_u64(value,"known prefill rows");
            if (rows < 16U || rows > 2048U) throw std::invalid_argument("known prefill rows must be 16 through 2048");
            options.known_prefix_prefill_rows = static_cast<std::uint32_t>(rows);
            options.known_prefix_prefill_explicit = true;
            continue;
        }
        if (argument == "--known-prefix-workspace-budget-bytes" || argument == "--known-prefix-host-metadata-budget-bytes" ||
            argument == "--known-prefix-uma-budget-bytes" || argument == "--known-prefix-host-headroom-bytes") {
            if (index + 1 >= argc) throw std::invalid_argument("known prefill budget requires a value");
            bool* supplied = argument == "--known-prefix-workspace-budget-bytes" ? &options.known_prefill_workspace_explicit :
                argument == "--known-prefix-host-metadata-budget-bytes" ? &options.known_prefill_metadata_explicit :
                argument == "--known-prefix-uma-budget-bytes" ? &options.known_prefill_uma_explicit : &options.known_prefill_headroom_explicit;
            if (*supplied) throw std::invalid_argument("duplicate known prefill budget");
            const auto value = parse_u64(argv[++index],"known prefill budget");
            if (!value && argument != "--known-prefix-host-headroom-bytes") throw std::invalid_argument("known prefill budgets must be positive");
            *supplied = true;
            if (argument == "--known-prefix-workspace-budget-bytes") options.known_prefill_workspace_budget = value;
            else if (argument == "--known-prefix-host-metadata-budget-bytes") options.known_prefill_metadata_budget = value;
            else if (argument == "--known-prefix-uma-budget-bytes") options.known_prefill_uma_budget = value;
            else options.known_prefill_host_headroom = value;
            continue;
        }
        if (argument == "--known-prefix-output-policy" || argument.starts_with("--known-prefix-output-policy=")) {
            if (options.known_prefix_output_policy_explicit) throw std::invalid_argument("duplicate known-prefix output policy");
            std::string_view value;
            if (argument == "--known-prefix-output-policy") {
                if (index + 1 >= argc) throw std::invalid_argument("known-prefix output policy requires a value");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--known-prefix-output-policy=").size());
            if (value == "all-rows") options.known_prefix_output_policy = strix::KnownWindowOutputPolicy::all_rows;
            else if (value == "terminal-row") options.known_prefix_output_policy = strix::KnownWindowOutputPolicy::terminal_row;
            else if (value == "state-only") options.known_prefix_output_policy = strix::KnownWindowOutputPolicy::state_only;
            else throw std::invalid_argument("known-prefix output policy must be all-rows, terminal-row or state-only");
            options.known_prefix_output_policy_explicit = true;
            continue;
        }
        if (argument == "--known-prefix-rows" || argument.starts_with("--known-prefix-rows=")) {
            if (options.known_prefix_rows_explicit) throw std::invalid_argument("duplicate known-prefix rows option");
            std::string_view value;
            if (argument == "--known-prefix-rows") {
                if (index + 1 >= argc) throw std::invalid_argument("known-prefix rows require a value");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--known-prefix-rows=").size());
            const auto rows = parse_u64(value, "known-prefix rows");
            if (rows != 2 && rows != 4 && rows != 8) throw std::invalid_argument("known-prefix rows must be 2, 4 or 8");
            options.known_prefix_rows = static_cast<std::uint32_t>(rows);
            options.known_prefix_rows_explicit = true;
            continue;
        }
        if (argument == "--known-window-router-rows") {
            if (options.known_window_router_rows) throw std::invalid_argument("duplicate known-window router rows option");
            options.known_window_router_rows = true;
            continue;
        }
        if (argument == "--known-prefix-expert-union") {
            if (options.known_prefix_expert_union) throw std::invalid_argument("duplicate known-prefix expert union option");
            options.known_prefix_expert_union = true;
            continue;
        }
        if (argument == "--known-prefix-window") {
            if (options.known_prefix_window) throw std::invalid_argument("duplicate known-prefix window option");
            options.known_prefix_window = true;
            continue;
        }
        if (argument == "--gdn-recurrent-storage" || argument.starts_with("--gdn-recurrent-storage=")) {
            if (options.gdn_recurrent_storage_explicit) throw std::invalid_argument("duplicate GDN recurrent storage option");
            std::string_view value;
            if (argument == "--gdn-recurrent-storage") {
                if (index + 1 >= argc) throw std::invalid_argument("GDN recurrent storage requires a value");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--gdn-recurrent-storage=").size());
            if (value == "key-value") options.gdn_recurrent_storage = strix::GdnRecurrentStorage::key_value;
            else if (value == "value-key") options.gdn_recurrent_storage = strix::GdnRecurrentStorage::value_key;
            else throw std::invalid_argument("GDN recurrent storage must be key-value or value-key");
            options.gdn_recurrent_storage_explicit = true;
            continue;
        }
        if (argument == "--gdn-recurrent-layout" || argument.starts_with("--gdn-recurrent-layout=")) {
            if (options.gdn_recurrent_layout_explicit) throw std::invalid_argument("duplicate GDN recurrent layout option");
            std::string_view value;
            if (argument == "--gdn-recurrent-layout") {
                if (index + 1 >= argc) throw std::invalid_argument("GDN recurrent layout requires a value");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--gdn-recurrent-layout=").size());
            if (value == "baseline") options.gdn_recurrent_layout = strix::HipGdnRecurrentLayout::baseline;
            else if (value == "stride32") options.gdn_recurrent_layout = strix::HipGdnRecurrentLayout::stride32;
            else if (value == "stride33") options.gdn_recurrent_layout = strix::HipGdnRecurrentLayout::stride33;
            else if (value == "transposed") options.gdn_recurrent_layout = strix::HipGdnRecurrentLayout::transposed;
            else throw std::invalid_argument("GDN recurrent layout must be baseline, stride32, stride33 or transposed");
            options.gdn_recurrent_layout_explicit = true;
            continue;
        }
        if (argument == "--qsa-gqa-heads-per-block") {
            if (options.qsa_gqa_explicit || index + 1 >= argc)
                throw std::invalid_argument("QSA heads per block requires one value");
            const std::string_view value = argv[++index];
            if (value == "1") options.qsa_gqa_heads_per_block = 1;
            else if (value == "2") options.qsa_gqa_heads_per_block = 2;
            else if (value == "4") options.qsa_gqa_heads_per_block = 4;
            else throw std::invalid_argument("QSA heads per block must be 1, 2 or 4");
            options.qsa_gqa_explicit = true;
            continue;
        }
        if (argument == "--qsa-hierarchical-selection") {
#ifndef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
            throw std::invalid_argument("--qsa-hierarchical-selection requires the experimental runtime build capability");
#else
            if (options.qsa_hierarchical_selection) throw std::invalid_argument("duplicate hierarchical QSA option");
            options.qsa_hierarchical_selection = true;
            continue;
#endif
        }
        if (argument == "--gdn-islands") {
#ifndef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
            throw std::invalid_argument("--gdn-islands requires the experimental runtime build capability");
#else
            if (options.gdn_islands) throw std::invalid_argument("duplicate GDN islands option");
            options.gdn_islands = true;
            continue;
#endif
        }
        if (argument == "--batched-moe-activation") {
#ifndef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
            throw std::invalid_argument("--batched-moe-activation requires the experimental runtime build capability");
#else
            if (options.batched_moe_activation)
                throw std::invalid_argument("duplicate batched MoE activation option");
            options.batched_moe_activation = true;
            continue;
#endif
        }
        if (argument == "--moe-token-cost-policy") {
            if (options.moe_token_cost_policy) throw std::invalid_argument("duplicate observed MoE token policy");
            options.moe_token_cost_policy = true;
            continue;
        }
        if (argument == "--moe-shared-overlap") {
            if (options.moe_shared_overlap)
                throw std::invalid_argument("duplicate MoE shared overlap option");
            options.moe_shared_overlap = true;
            continue;
        }
        if (argument == "--moe-overlap-arena-budget-bytes") {
            if (options.moe_overlap_budget_explicit || index + 1 >= argc)
                throw std::invalid_argument("MoE overlap arena budget requires one byte value");
            options.moe_overlap_budget_explicit = true;
            options.moe_overlap_arena_budget_bytes = parse_u64(argv[++index], "MoE overlap arena budget");
            continue;
        }
        if (argument == "--prepared-eager") {
            if (options.prepared_eager)
                throw std::invalid_argument("duplicate prepared eager option");
            options.prepared_eager = true;
            continue;
        }
        if (argument == "--gpu-greedy-token") {
            if (options.gpu_greedy_token)
                throw std::invalid_argument("duplicate GPU greedy token option");
            options.gpu_greedy_token = true;
            continue;
        }
        if (argument == "--execution-inventory") {
            if (options.execution_inventory)
                throw std::invalid_argument("duplicate execution inventory option");
            options.execution_inventory = true;
            continue;
        }
        if (argument == "--execution-telemetry-trace") {
            if (options.execution_telemetry_trace) throw std::invalid_argument("duplicate execution telemetry trace option");
            options.execution_telemetry_trace = true;
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
        if (argument == "--q51-slot-parallel-down") {
            if (options.q51_slot_parallel_down) throw std::invalid_argument("duplicate slot-parallel Q5_1 down option");
            options.q51_slot_parallel_down = true;
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
        if (argument == "--staged-moe-router-logits") {
            if (options.staged_moe_router_logits) throw std::invalid_argument("duplicate staged MoE router logits option");
            options.staged_moe_router_logits = true;
            continue;
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

        if (argument == "--validation-final-state") {
            if (index + 1 >= argc || !options.validation_final_state_path.empty())
                throw std::invalid_argument("--validation-final-state requires one fresh private directory");
            options.validation_final_state_path = argv[++index];
            if (options.validation_final_state_path.empty())
                throw std::invalid_argument("--validation-final-state directory is empty");
            continue;
        }
        if (argument == "--validation-final-state-max-bytes") {
            if (index + 1 >= argc || options.validation_final_state_budget_explicit)
                throw std::invalid_argument("--validation-final-state-max-bytes requires one byte budget");
            options.validation_final_state_budget_explicit = true;
            options.validation_final_state_maximum_bytes = parse_u64(argv[++index], "validation-final-state-max-bytes");
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

        if (argument == "--ple-cache-rows" || argument == "--ple-cache-budget-bytes") {
            if (index + 1 >= argc) throw std::invalid_argument("PLE cache option requires a value");
            bool& explicit_value = argument == "--ple-cache-rows" ? options.ple_cache_rows_explicit : options.ple_cache_budget_explicit;
            if (explicit_value) throw std::invalid_argument("duplicate PLE cache limit");
            const auto value = parse_u64(argv[++index], "PLE cache limit");
            if (!value) throw std::invalid_argument("PLE cache limits must be positive");
            explicit_value = true;
            if (argument == "--ple-cache-rows") options.ple_cache_rows = value;
            else options.ple_cache_budget_bytes = value;
            continue;
        }
        if (argument == "--ple-cpu-helper-threads" || argument.starts_with("--ple-cpu-helper-threads=")) {
            if (options.ple_cpu_helper_threads_explicit) throw std::invalid_argument("duplicate PLE CPU helper threads option");
            std::string_view value;
            if (argument == "--ple-cpu-helper-threads") {
                if (index + 1 >= argc) throw std::invalid_argument("PLE CPU helper threads require a value");
                value = argv[++index];
            } else value = argument.substr(std::string_view("--ple-cpu-helper-threads=").size());
            const auto count = parse_u64(value, "PLE CPU helper threads");
            if (count < 1U || count > 8U) throw std::invalid_argument("PLE CPU helper threads must be 1 through 8 when explicit");
            options.ple_cpu_helper_threads = static_cast<std::uint32_t>(count);
            options.ple_cpu_helper_threads_explicit = true;
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

    validate_model_copy_cli(options);
    if (options.tokens.empty() && options.serving_prefix_files.empty() &&
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
        options.tokens.empty() && options.serving_prefix_files.empty()) {
        throw std::invalid_argument(
            "--generate requires at least one prefix token id or serving prefix file");
    }
    if (options.validation_final_state_budget_explicit && options.validation_final_state_path.empty())
        throw std::invalid_argument("final state export budget requires --validation-final-state");
    if (!options.validation_final_state_path.empty()) {
        if (!options.validation_final_state_maximum_bytes || (!options.generate_tokens &&
            !(options.known_prefix_window && options.known_prefix_prefill_rows &&
              options.known_prefix_output_policy == strix::KnownWindowOutputPolicy::state_only)) ||
            !options.raw_recording_path.empty() || !options.reference_path.empty() ||
            options.prefix_cache_roundtrip || options.execution.mode != strix::DecodeExecutionMode::eager ||
            options.execution.integer_head != strix::DecodeIntegerHead::none)
            throw std::invalid_argument("final state validation requires eager generation without RAW, reference, roundtrip or integer head");
        strix::validate_final_generation_state_destination(options.validation_final_state_path);
    }
    if ((options.prefetch_q8_lm_head || options.wave8_q8_lm_head || options.wave8_q8_gdn_input ||
         options.wave8_q8_gdn_output || options.wave8_q8_qsa_output) &&
        (options.execution.mode != strix::DecodeExecutionMode::eager ||
         options.execution.integer_head != strix::DecodeIntegerHead::none))
        throw std::invalid_argument("Q8 projection organizations require eager execution without integer head");
    if (options.q51_slot_parallel_down &&
        (options.execution.mode != strix::DecodeExecutionMode::eager ||
         options.execution.integer_head != strix::DecodeIntegerHead::none))
        throw std::invalid_argument("slot-parallel Q5_1 down requires eager execution without integer head");
    if (options.prefetch_hc_projection_next_tile &&
        (options.execution.mode != strix::DecodeExecutionMode::eager ||
         options.execution.integer_head != strix::DecodeIntegerHead::none))
        throw std::invalid_argument("HC next-tile prefetch requires eager execution without integer head");
    if (options.wave8_q8_lm_head && !options.prefetch_q8_lm_head)
        throw std::invalid_argument("wave8 Q8 LM head requires explicit prefetch Q8 LM head");
    if (options.batched_moe_activation && options.execution.mode != strix::DecodeExecutionMode::eager)
        throw std::invalid_argument("batched MoE activation requires eager mode");
    if (options.moe_shared_overlap && (options.prefix_cache_roundtrip ||
        options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none))
        throw std::invalid_argument("MoE shared overlap requires eager mode without integer head or prefix roundtrip");
    if ((options.moe_overlap_budget_explicit && !options.moe_shared_overlap) ||
        (options.moe_shared_overlap && !options.moe_overlap_arena_budget_bytes))
        throw std::invalid_argument("MoE overlap arena budget requires enabled overlap and nonzero bytes");
    if (options.gdn_islands && (options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none || options.prefix_cache_roundtrip ||
        (options.wave8_q8_gdn_input != options.wave8_q8_gdn_output)))
        throw std::invalid_argument("GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    if (options.prepared_eager && (options.prefix_cache_roundtrip ||
        options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none))
        throw std::invalid_argument("prepared eager execution excludes graph, integer head and prefix roundtrip");
    if (options.gpu_greedy_token && (!options.generate_tokens ||
        !options.reference_path.empty() || !options.raw_recording_path.empty() ||
        options.prefix_cache_roundtrip || options.execution.mode != strix::DecodeExecutionMode::eager ||
        options.execution.integer_head != strix::DecodeIntegerHead::none))
        throw std::invalid_argument("GPU greedy token output requires eager generation without RAW, reference, roundtrip or integer head");
    if ((options.ple_cache_rows_explicit || options.ple_cache_budget_explicit) &&
        (!options.ple_cache_rows_explicit || !options.ple_cache_budget_explicit))
        throw std::invalid_argument("PLE cache requires both positive explicit row and byte limits");
    if (!options.offline_compile_provenance_path.empty() && options.offline_tuning_cache_path.empty())
        throw std::invalid_argument("offline compile provenance requires --offline-tuning-cache");
    validate_moe_token_cost_options(options);
    validate_prompt_lookup_snapshot_options(options);
    validate_known_prefill_options(options);
    validate_serving_prefix_cache_limits(options);
    if (options.serving_prefix_files.empty() && (options.serving_tensor_cohort_explicit || options.serving_concurrency_explicit || options.serving_memory_explicit ||
        options.serving_request_explicit || options.serving_headroom_explicit))
        throw std::invalid_argument("serving controls require --serve-prefix-u32le");
    if (options.known_prefix_output_policy_explicit && !options.known_prefix_window)
        throw std::invalid_argument("known-prefix output policy requires --known-prefix-window");
    if (options.known_prefix_expert_union && !options.known_prefix_window)
        throw std::invalid_argument("known-prefix expert union requires --known-prefix-window");
    if (options.known_prefix_rows_explicit && !options.known_prefix_window)
        throw std::invalid_argument("known-prefix rows require --known-prefix-window");
    if (options.prompt_lookup_adaptive && (!options.prompt_lookup || options.prompt_lookup_depth_explicit))
        throw std::invalid_argument("adaptive prompt lookup requires --prompt-lookup without an explicit fixed depth");
    if (options.prompt_lookup_depth_explicit && !options.prompt_lookup)
        throw std::invalid_argument("prompt lookup depth requires --prompt-lookup");
    if (options.device_queue_trace && !options.host_dispatch_telemetry)
        throw std::invalid_argument("device queue trace requires --host-dispatch-telemetry");
    if (options.execution_telemetry_trace && !options.host_dispatch_telemetry)
        throw std::invalid_argument("execution telemetry trace requires host dispatch telemetry");
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
    if (options.staged_moe_router_logits && !options.parallel_moe_router)
        throw std::invalid_argument("staged router logits requires parallel projection");
    if (options.staged_moe_router_logits && (options.mtp.enabled || options.mtp.supplied))
        throw std::invalid_argument("staged router logits excludes MTP");
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
    validate_teacher_forced_score(options);
    validate_numerical_profile_cli(options);
    validate_mtp_cli(options);
    return options;
}

inline constexpr std::string_view validated_model_content_identity =
    "3726d6f0d9ba30b9d655ba303595118941baeca5e6945fc8785f938e0f5455e4";

inline OptimizationProfileResolution resolve_optimization_profile(
    Options& options, const ModelContract& model, const gguf::FileInfo& info,
    const ModelManifest& manifest, std::string_view content_identity,
    bool has_q4k_a8_runtime, bool permit_generation_defaults = true) {
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
    // Concurrent submission is an ordinary CPU-output generation default only. Keep the existing
    // v3 policy for other controllers/intents or additional numerical choices;
    // explicit options are never disabled to make a request eligible. The
    // caller excludes diagnostic target-input export, which is not in Options.
    const bool concurrency_generation = permit_generation_defaults && options.generate_tokens && !options.tokens.empty() &&
        !options.mtp.enabled && options.serving_prefix_files.empty() && !options.prompt_lookup && !options.known_prefix_window &&
        !options.prefix_cache_roundtrip && options.raw_recording_path.empty() && !options.gpu_greedy_token &&
        options.logprob_token_ids.empty() && !options.device_queue_trace && !options.moe_token_cost_policy && !options.ranked_moe_router_topk &&
        options.offline_tuning_cache_path.empty() && options.offline_compile_provenance_path.empty() &&
        options.gdn_recurrent_layout == HipGdnRecurrentLayout::baseline &&
        options.gdn_recurrent_storage == GdnRecurrentStorage::key_value &&
        !options.qsa_hierarchical_selection && options.qsa_gqa_heads_per_block == 1 &&
        !options.prefetch_hc_projection_next_tile && !options.prefetch_q8_lm_head && !options.wave8_q8_lm_head &&
        !options.wave8_q8_qsa_output;
    if (concurrency_generation) {
        options.q51_slot_parallel_down = true;
        options.batched_moe_activation = true;
        options.wave8_q8_gdn_input = true;
        options.wave8_q8_gdn_output = true;
        options.gdn_islands = true;
        options.prepared_eager = true;
        options.moe_shared_overlap = true;
        options.prefetch_q8_lm_head = true;
        options.wave8_q8_lm_head = true;
        options.wave8_q8_qsa_output = true;
        result.selected = "canonical-a8-wave-qsa-hc-shared-output-v5";
        result.reason = "qualified-ordinary-cpu-generation";
    }
    return result;
}

inline void apply_model_copy_profile(const Options& options, OptimizationProfileResolution& resolution) {
    validate_model_copy_cli(options);
    if (options.model_copy_tier != HipModelCopyTier::none)
        resolution.selected += "+model-copy-tier:" + std::string(hip_model_copy_tier_name(options.model_copy_tier));
}

// Apply only after the organization selector. Explicit diagnostic/manual
// callers may supply the same complete organization; missing prerequisites
// fail rather than being enabled or disabled by an arithmetic choice.
inline void apply_numerical_profile(
    const Options& options, const ModelContract& model, const ModelManifest& manifest,
    std::string_view content_identity, bool has_q4k_a8_runtime,
    OptimizationProfileResolution& resolution) {
    if (options.numerical_profile == DecodeNumericalProfile::exact_v5) return;
    validate_model_copy_cli(options);
    validate_numerical_profile_cli(options);
    if (numerical_profile_name(options.numerical_profile).empty())
        throw std::invalid_argument("unknown numerical profile");
    const auto canonical = ModelContract::qwen38_flash_next();
    if (!has_q4k_a8_runtime || content_identity != validated_model_content_identity ||
        !model.validate().empty() || model.hidden_size != canonical.hidden_size ||
        model.hidden_layers != canonical.hidden_layers || model.vocab_size != canonical.vocab_size ||
        model.hc_count != canonical.hc_count || model.hc_lowrank != canonical.hc_lowrank ||
        model.moe_intermediate_size != canonical.moe_intermediate_size ||
        model.num_experts != canonical.num_experts || model.num_experts_per_token != canonical.num_experts_per_token ||
        options.context_tokens != 8192 || !manifest.layout_identity ||
        model_manifest_layout_identity(manifest) != manifest.layout_identity)
        throw std::invalid_argument("numerical profile requires the admitted canonical content, model and context");
    if (options.residency != HipModelResidencyMode::registered_file_mapping ||
        !options.parallel_hyper_mix || !options.staged_hyper_projections || !options.fused_staged_hyper_mix ||
        !options.typed_hyper_projection || !options.prefetch_typed_hyper_rows ||
        !options.prefetch_q8_hyper_output || !options.lane_chain_hyper_output ||
        !options.q4k_a8g32_gate_up || !options.grouped_exact_q51_down || !options.q51_slot_parallel_down ||
        !options.batched_moe_activation || !options.deferred_moe_status || !options.wave_moe_router_topk ||
        !options.virtual_q8_gdn_input || !options.prefetch_q8_gdn_input || !options.virtual_q8_gdn_output ||
        !options.wave8_q8_gdn_input || !options.wave8_q8_gdn_output || !options.prefetch_q8_qsa_input ||
        !options.prefetch_q8_shared_expert || !options.virtual_q8_lm_head || !options.prefetch_q8_lm_head ||
        !options.wave8_q8_lm_head || !options.wave8_q8_qsa_output ||
        !options.prepared_eager || !options.gdn_islands || !options.moe_shared_overlap)
        throw std::invalid_argument("numerical profile requires the resolved full v5 organization without implicit flag changes");
    resolution.selected += "+" + std::string(numerical_profile_name(options.numerical_profile));
}

} // namespace strix::decode_cli
