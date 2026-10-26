#pragma once
#include "../apps/hip_decode_options.hpp"
// Frozen organization logic from 9b1d1c3e. Changes require historical evidence.
namespace strix::decode_cli {
inline OptimizationProfileResolution resolve_historical_9b1d1c3e(
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
    // STRIX_GSQ_Q20_A8=1: the operator admits a GSQ-RCO file (or a GSQ/production composite) to the same
    // bundle; its non-Q8 tensors run the GSQ kernels (own numerical identities and CPU oracles).
    const char* gsq_admit = std::getenv("STRIX_GSQ_Q20_A8");
    if (content_identity != validated_model_content_identity && !(gsq_admit && std::string_view(gsq_admit) == "1"))
        return fallback("checkpoint-content-not-qualified");
    if (options.context_explicit && options.context_tokens != 8192 && !options.unqualified_context)
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
    if (!options.unqualified_context || !options.context_explicit) options.context_tokens = 8192;
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
        // GDN islands capture the incumbent projections; the dense A8 profile runs without them.
        options.gdn_islands = !numerical_profile_dense_a8(options.numerical_profile);
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

}
