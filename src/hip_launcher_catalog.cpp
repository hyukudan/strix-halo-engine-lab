#include "strix/hip_launcher_catalog.hpp"

#include "strix/runtime_launchers.hpp"
#include "strix/hip_integer_projection_runtime.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/lane_chain_hyper_output.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
void register_hip_graph_position_launcher(ExecutionLauncherRegistry&);
namespace {

std::uint32_t checked_u32(
    std::uint64_t value,
    const char* name) {
    if (value == 0 ||
        value >
            std::numeric_limits<
                std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(name) +
            " does not fit uint32");
    }
    return static_cast<std::uint32_t>(
        value);
}



HipGdnLauncherConfig make_gdn_config(const ModelContract& model,
    HipActivationStorage activation, bool graph, bool virtual_q8, bool prefetch_q8, bool virtual_output,
    bool wave8, bool wave8_output, HipGdnRecurrentLayout recurrent_layout = HipGdnRecurrentLayout::baseline,
    GdnRecurrentStorage recurrent_storage = GdnRecurrentStorage::key_value, bool dense_a8 = false, bool dense_f16 = false) {
    HipGdnLauncherConfig config;
    config.model = model;
    config.activation = activation;
    config.enable_device_graph_inputs = graph;
    config.virtual_q8_input_projections = virtual_q8;
    config.prefetch_q8_input_projections = prefetch_q8;
    config.wave8_q8_input_projections = wave8;
    config.virtual_q8_output_projection = virtual_output;
    config.wave8_q8_output_projection = wave8_output;
    config.recurrent_layout = recurrent_layout;
    config.recurrent_storage = recurrent_storage;
    config.dense_a8_projections = dense_a8;
    config.dense_f16_prefill = dense_f16;
    return config;
}

HipIndexedAttentionLauncherConfig
indexed_attention_config(
    const ModelContract& model,
    HipActivationStorage activation, bool graph = false, bool prefetch = false,
    std::vector<std::uint32_t> layers = {}, std::shared_ptr<qsa_input_prefetch::RuntimeDispatchStats> stats = {},
    bool wave8_output = false, bool hierarchical_selection = false,
    std::shared_ptr<QsaHierarchicalSelectionStats> selection_stats = {},
    std::uint32_t gqa_heads = 1, std::shared_ptr<QsaHierarchicalSelectionStats> gqa_stats = {}, bool dense_a8 = false,
    bool dense_f16 = false) {
    HipIndexedAttentionLauncherConfig config;
    config.model = model;
    config.activation = activation;
    config.enable_device_graph_inputs = graph;
    config.prefetch_q8_input = prefetch;
    config.prefetch_eligible_layers=std::move(layers);
    config.prefetch_stats=std::move(stats);
    config.wave8_q8_output = wave8_output;
    config.qsa_hierarchical_selection = hierarchical_selection;
    config.selection_stats = std::move(selection_stats);
    config.qsa_gqa_heads_per_block = gqa_heads;
    config.gqa_stats = std::move(gqa_stats);
    config.dense_a8_projections = dense_a8;
    config.dense_f16_prefill = dense_f16;
    return config;
}

HipPleLauncherConfig
ple_config(
    const ModelContract& model,
    HipActivationStorage activation, bool graph = false) {
    HipPleLauncherConfig config;
    config.model = model;
    config.activation =
        activation;
    config.enable_resident_graph = graph;
    return config;
}

HipElementwiseLauncherConfig
elementwise_config(
    const ModelContract& model,
    HipActivationStorage activation) {
    HipElementwiseLauncherConfig config;
    config.hidden_size =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    config.hc_count =
        checked_u32(
            model.hc_count,
            "hc_count");
    config.activation =
        activation;
    return config;
}

HipEmbeddingLauncherConfig
embedding_config(
    const ModelContract& model,
    HipActivationStorage activation, bool graph = false) {
    HipEmbeddingLauncherConfig config;
    config.hidden_size =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    config.output =
        activation;
    config.enable_device_graph_inputs = graph;
    return config;
}

HipHyperMixLauncherConfig
hyper_mix_config(
    const ModelContract& model,
    HipActivationStorage activation, bool parallel, bool staged, bool fused, bool typed, bool prefetch_rows, bool prefetch_output, bool lane_chain,
    bool splitk, bool splitk_fp64, bool next_tile,
    std::shared_ptr<HipHyperMixArithmeticStats> arithmetic_stats = {}, bool deferred_norm = false, bool fast_prefill = false) {
    HipHyperMixLauncherConfig config;
    config.hidden_size =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    config.hc_count =
        checked_u32(
            model.hc_count,
            "hc_count");
    config.lowrank =
        checked_u32(
            model.hc_lowrank,
            "hc_lowrank");
    config.rms_epsilon =
        static_cast<float>(
            model.rms_norm_epsilon);
    config.activation =
        activation;
    config.parallel = parallel;
    config.staged_projections = staged;
    config.fused_injection = fused;
    config.typed_projection = typed;
    config.prefetch_rows = prefetch_rows;
    config.prefetch_q8_output = prefetch_output;
    config.lane_chain_output = lane_chain;
    config.splitk_projection = splitk;
    config.splitk_deferred_norm = deferred_norm;
    config.splitk_projection_fp64 = splitk_fp64;
    config.arithmetic_stats = std::move(arithmetic_stats);
    config.prefetch_hc_projection_next_tile = next_tile;
    config.fast_prefill_wmma = fast_prefill;
    return config;
}


HipLmHeadLauncherConfig
lm_head_config(
    const ModelContract& model,
    HipActivationStorage activation, bool virtual_wave, bool prefetch, bool wave8) {
    HipLmHeadLauncherConfig config;
    config.hidden_size =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    config.vocab_size =
        checked_u32(
            model.vocab_size,
            "vocab_size");
    config.input =
        activation;
    config.virtual_q8_wave = virtual_wave;
    config.prefetch_q8_wave = prefetch;
    config.wave8_q8_wave = wave8;
    return config;
}


HipSharedExpertLauncherConfig
shared_expert_config(
    const ModelContract& model,
    HipActivationStorage activation, bool prefetch=false, std::vector<std::uint32_t> layers={}, std::shared_ptr<shared_expert_prefetch::RuntimeDispatchStats> stats={},
    bool dense_a8=false, bool dense_f16=false) {
    HipSharedExpertLauncherConfig config;
    config.hidden_size =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    config.intermediate_size =
        checked_u32(
            model.shared_expert_intermediate_size,
            "shared_expert_intermediate_size");
    config.activation =
        activation;
    config.prefetch_q8=prefetch;config.prefetch_eligible_layers=std::move(layers);config.prefetch_stats=std::move(stats);
    config.dense_a8_projections = dense_a8;
    config.dense_f16_prefill = dense_f16;
    return config;
}


HipMoeExpertsLauncherConfig
moe_experts_config(
    const ModelContract& model,
    HipActivationStorage activation, bool q4k_a8 = false,
    std::shared_ptr<moe_q4k_a8::RuntimeDispatchStats> stats = {}, bool grouped_q51=false, bool deferred_status=false, bool batched_activation=false, bool slot_parallel=false,
    bool a8_down_q51=false, bool a8_down_q80=false, bool q5k_a8_gateup=false) {
    HipMoeExpertsLauncherConfig config;
    config.hidden_size =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    config.intermediate_size =
        checked_u32(
            model.moe_intermediate_size,
            "moe_intermediate_size");
    config.expert_count =
        checked_u32(
            model.num_experts,
            "num_experts");
    config.top_k =
        checked_u32(
            model.num_experts_per_token,
            "num_experts_per_token");
    config.activation =
        activation;
    config.q4k_a8g32_gate_up = q4k_a8;
    config.grouped_exact_q51_down = grouped_q51;
    config.q51_slot_parallel_down = slot_parallel;
    config.a8_down_q51 = a8_down_q51;
    config.a8_down_q80 = a8_down_q80;
    config.q5k_a8g32_gate_up = q5k_a8_gateup;
    config.deferred_moe_status = deferred_status;
    config.batched_slot_activation = batched_activation;
    config.q4k_a8_stats = std::move(stats);
    return config;
}

HipMoeRouteLauncherConfig
moe_route_config(
    const ModelContract& model,
    HipActivationStorage activation, bool parallel = false, bool ranked = false, bool wave = false,
    std::shared_ptr<const WaveRouterManifestAdmission> admission = {}, bool staged = false) {
    HipMoeRouteLauncherConfig config;
    config.hidden_size =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    config.expert_count =
        checked_u32(
            model.num_experts,
            "num_experts");
    config.top_k =
        checked_u32(
            model.num_experts_per_token,
            "num_experts_per_token");
    config.activation =
        activation;
    config.normalize_selected =
        model.norm_topk_probability;
    config.parallel_projection = parallel;
    config.staged_logits = staged;
    config.ranked_topk = ranked;
    config.wave_topk = wave;
    config.wave_manifest_admission = std::move(admission);
    return config;
}

// Metadata-only admission. Physical ranges and generations are checked by
// the HC bundle before publishing its first launcher or enqueueing any work.
void validate_deferred_hc_catalog_model(const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources* resources) {
    if (!resources || !resources->model_manifest || !resources->tensor_access ||
        compiled.logical_rows != 1U || compiled.request_concurrency != 1U || compiled.device_graph_inputs ||
        compiled.bindings.bindings.size() != compiled.execution.steps.size() ||
        !resources->model_manifest->layout_identity ||
        resources->tensor_access->manifest_layout_identity != resources->model_manifest->layout_identity)
        throw std::invalid_argument("HC deferred normalization requires complete canonical R1 C1 metadata");
    std::uint32_t count = 0;
    for (const auto& step : compiled.execution.steps) {
        if (step.op != SerialOp::attn_hyper_mix && step.op != SerialOp::mlp_hyper_mix &&
            step.op != SerialOp::final_hyper_mix) continue;
        if (step.id >= compiled.bindings.bindings.size())
            throw std::invalid_argument("HC deferred mixer binding is absent");
        const auto& binding = compiled.bindings.bindings[step.id];
        if (binding.host_or_runtime_step || !binding.kernel_id ||
            binding.kernel_id->find(".hc-deferred-norm-v1") == std::string::npos ||
            !binding.kernel_id->ends_with(lane_chain_hyper_output_suffix))
            throw std::invalid_argument("HC deferred normalization rejects every non-lane-chain mixer fallback");
        const auto tensor = [&](std::uint32_t slot) -> const ManifestTensor& {
            const auto* access = find_execution_tensor_access(*resources->tensor_access,step.id,slot);
            if (!access || access->op != step.op || access->tensor_indices.size() != 1U ||
                access->tensor_indices.front() >= resources->model_manifest->tensors.size())
                throw std::invalid_argument("HC deferred canonical tensor role is absent");
            return resources->model_manifest->tensors[access->tensor_indices.front()];
        };
        const auto& norm = tensor(0U);const auto& up = tensor(2U);
        if (norm.ggml_type != 0U || norm.dimensions != std::vector<std::uint64_t>{10240U} ||
            norm.byte_size != 40960U || !lane_chain_hyper_output_eligible(up,2560U,4U,320U,0U))
            throw std::invalid_argument("HC deferred normalization requires F32 wide norm and eligible Q8 up for every mixer");
        ++count;
    }
    if (count != 97U)
        throw std::invalid_argument("HC deferred normalization requires all 97 canonical mixers without fallback");
}

void validate_numerical_catalog_config(const ModelContract& model, const HipLauncherCatalogConfig& config) {
    if (numerical_profile_name(config.numerical_profile).empty())
        throw std::invalid_argument("HIP catalog numerical profile is invalid");
    if (config.a8_down_q51 != numerical_profile_q51_a8_down(config.numerical_profile) ||
        config.a8_down_q80 != numerical_profile_q80_a8_down(config.numerical_profile))
        throw std::invalid_argument("HIP catalog A8 down recipe differs from typed numerical profile");
    if (config.q5k_a8g32_gate_up != numerical_profile_q5k_a8_gateup(config.numerical_profile) ||
        config.splitk_deferred_norm != numerical_profile_hc_deferred_norm(config.numerical_profile))
        throw std::invalid_argument("HIP catalog Q5K or HC deferred recipe differs from typed numerical profile");
    if (config.dense_a8 != numerical_profile_dense_a8(config.numerical_profile))
        throw std::invalid_argument("HIP catalog dense A8 recipe differs from typed numerical profile");
    if ((config.fast_prefill_components != 0U) != numerical_profile_fast_prefill(config.numerical_profile) ||
        (config.fast_prefill_components & ~kFastPrefillKnown) != 0U)
        throw std::invalid_argument("HIP catalog fast prefill components differ from typed numerical profile");
    if (config.dense_a8 && (!config.wave8_q8_gdn_input || !config.wave8_q8_gdn_output || !config.wave8_q8_qsa_output ||
                            !config.prefetch_q8_qsa_input || !config.prefetch_q8_shared_expert))
        throw std::invalid_argument("HIP catalog dense A8 requires the wave8/prefetch Q8 organizations it replaces");
    if (config.splitk_deferred_norm && (!config.lane_chain_hyper_output || config.prefetch_hc_projection_next_tile))
        throw std::invalid_argument("HC deferred normalization requires all mixers on the lane-chain path");
    if (config.numerical_profile == DecodeNumericalProfile::exact_v5) return;
    if (config.splitk_hyper_projection != numerical_profile_hc_splitk(config.numerical_profile) ||
        config.splitk_hyper_projection_fp64 || config.enable_device_graph_inputs ||
        config.activation != HipActivationStorage::bf16 || !config.deferred_moe_status ||
        model.hidden_size != 2560 || model.hc_count != 4 || model.hc_lowrank != 320 ||
        model.moe_intermediate_size != 640 || model.num_experts != 512 ||
        model.num_experts_per_token != 10 || model.hidden_layers != 48)
        throw std::invalid_argument("HIP catalog numerical profile requires matching canonical eager BF16 arithmetic");
}

}  // namespace


void register_complete_hip_kernel_catalog(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    HipLauncherCatalogConfig config,
    kernels::KernelRegistry& registry,
    const integer_projection::RuntimeConfig* alternate_lm_head) {
    validate_numerical_catalog_config(model, config);
    if (request.a8_down_q51 != config.a8_down_q51 || request.a8_down_q80 != config.a8_down_q80 ||
        request.q5k_a8g32_gate_up != config.q5k_a8g32_gate_up || request.dense_a8 != config.dense_a8 ||
        (config.numerical_profile != DecodeNumericalProfile::exact_v5 &&
         (alternate_lm_head || request.device_graph_inputs || request.require_graph_safe ||
          request.logical_rows != 1 || request.request_concurrency != 1)))
        throw std::invalid_argument("HIP catalog numerical profile request/config or C1 execution mismatch");
    validate_gdn_recurrent_storage(config.gdn_recurrent_storage);
    if (config.gdn_recurrent_storage != GdnRecurrentStorage::key_value && config.enable_device_graph_inputs)
        throw std::invalid_argument("GDN recurrent storage candidate requires eager execution");
    (void)gdn_recurrent_layout_marker(config.gdn_recurrent_layout);
    if (request.gdn_recurrent_storage != config.gdn_recurrent_storage)
        throw std::invalid_argument("GDN recurrent storage request/config mismatch");
    if (config.gdn_recurrent_storage != GdnRecurrentStorage::key_value &&
        (alternate_lm_head || request.require_graph_safe || request.device_graph_inputs ||
         request.logical_rows != 1 || request.request_concurrency != 1))
        throw std::invalid_argument("GDN recurrent storage candidate requires eager floating C1 execution");
    if (request.gdn_recurrent_layout != config.gdn_recurrent_layout)
        throw std::invalid_argument("GDN recurrent layout request/config mismatch");
    if (config.gdn_recurrent_layout != HipGdnRecurrentLayout::baseline &&
        (alternate_lm_head || config.enable_device_graph_inputs || request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("GDN recurrent LDS layouts support eager floating execution only");
    (void)qsa_gqa_organization_marker(config.qsa_gqa_heads_per_block);
    if (request.qsa_gqa_heads_per_block != config.qsa_gqa_heads_per_block)
        throw std::invalid_argument("QSA GQA catalog request/config mismatch");
    if (config.qsa_gqa_heads_per_block != 1 &&
        (alternate_lm_head || config.enable_device_graph_inputs || !config.deferred_moe_status ||
         !request.deferred_moe_status || request.device_graph_inputs || request.require_graph_safe ||
         request.logical_rows != 1 || request.request_concurrency != 1))
        throw std::invalid_argument("QSA GQA catalog requires eager C1 deferred completion without integer head");
    if (request.qsa_hierarchical_selection != config.qsa_hierarchical_selection)
        throw std::invalid_argument("hierarchical QSA catalog request/config mismatch");
    if (config.qsa_hierarchical_selection &&
        (alternate_lm_head || config.enable_device_graph_inputs || !config.deferred_moe_status ||
         !request.deferred_moe_status || request.device_graph_inputs || request.require_graph_safe ||
         request.logical_rows != 1 || request.request_concurrency != 1))
        throw std::invalid_argument("hierarchical QSA catalog requires eager C1 deferred completion without integer head");
    if (alternate_lm_head && config.virtual_q8_lm_head)
        throw std::invalid_argument("integer and virtual Q8 LM heads are mutually exclusive");
    if (alternate_lm_head && config.enable_device_graph_inputs)
        throw std::invalid_argument("blocking integer LM-head cannot enter a captured graph");
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (config.prefetch_q8_lm_head != request.prefetch_q8_lm_head ||
        config.wave8_q8_lm_head != request.wave8_q8_lm_head ||
        config.wave8_q8_gdn_input != request.wave8_q8_gdn_input ||
        config.wave8_q8_gdn_output != request.wave8_q8_gdn_output ||
        config.wave8_q8_qsa_output != request.wave8_q8_qsa_output)
        throw std::invalid_argument("HIP catalog Q8 organization differs from binding request");
    if (config.wave8_q8_gdn_output && !config.virtual_q8_gdn_output)
        throw std::invalid_argument("HIP catalog wave8 GDN output requires virtual Q8 output");
    if ((config.prefetch_q8_lm_head || config.wave8_q8_lm_head || config.wave8_q8_gdn_input ||
         config.wave8_q8_gdn_output || config.wave8_q8_qsa_output) &&
        (config.enable_device_graph_inputs || request.device_graph_inputs || request.require_graph_safe || alternate_lm_head))
        throw std::invalid_argument("HIP catalog Q8 organizations support eager floating execution only");
    if (config.staged_moe_router_logits != request.staged_moe_router_logits ||
        (config.staged_moe_router_logits && !config.parallel_moe_router))
        throw std::invalid_argument("staged router catalog mode differs from binding request");
    if (config.wave_moe_router_topk != request.wave_moe_router_topk ||
        (config.wave_moe_router_topk && (!config.parallel_moe_router || config.ranked_moe_router_topk)))
        throw std::invalid_argument("HIP catalog wave router mode differs from binding request");
    if(config.prefetch_q8_qsa_input!=request.prefetch_q8_qsa_input || config.qsa_input_prefetch_layers!=request.qsa_input_prefetch_layers)throw std::invalid_argument("HIP catalog QSA input prefetch differs from binding request");
    qsa_input_prefetch::validate_layers(model,config.prefetch_q8_qsa_input,config.qsa_input_prefetch_layers);
    if(config.prefetch_q8_shared_expert!=request.prefetch_q8_shared_expert || config.shared_expert_prefetch_layers!=request.shared_expert_prefetch_layers)throw std::invalid_argument("HIP catalog shared prefetch request mismatch");
    shared_expert_prefetch::validate_layers(model.hidden_layers,config.prefetch_q8_shared_expert,config.shared_expert_prefetch_layers);
    if (config.ranked_moe_router_topk != request.ranked_moe_router_topk ||
        (config.ranked_moe_router_topk && !config.parallel_moe_router))
        throw std::invalid_argument("HIP catalog ranked router mode differs from binding request");
    if(config.lane_chain_hyper_output!=request.lane_chain_hyper_output)
        throw std::invalid_argument("HIP catalog lane-chain HC output differs from binding request");
    if (config.prefetch_hc_projection_next_tile != request.prefetch_hc_projection_next_tile)
        throw std::invalid_argument("HIP catalog HC next-tile prefetch differs from binding request");
    if (config.prefetch_hc_projection_next_tile &&
        (!config.parallel_hyper_mix || !config.staged_hyper_projections || !config.fused_staged_hyper_mix ||
         !config.typed_hyper_projection || !config.prefetch_typed_hyper_rows ||
         config.splitk_hyper_projection || config.splitk_hyper_projection_fp64 ||
         config.activation != HipActivationStorage::bf16 || model.hidden_size != 2560 ||
         model.hc_count != 4 || model.hc_lowrank != 320 || model.hidden_layers != 48 ||
         model.bytes_bf16 != 2 || model.bytes_fp32 != 4 ||
         config.enable_device_graph_inputs || request.device_graph_inputs || request.require_graph_safe ||
         request.logical_rows != 1 || request.request_concurrency != 1))
        throw std::invalid_argument("HIP catalog HC next-tile prefetch requires canonical typed staged eager C1 execution");
    if (config.deferred_moe_status != request.deferred_moe_status ||
        (config.deferred_moe_status && (config.enable_device_graph_inputs || request.require_graph_safe)))
        throw std::invalid_argument("HIP catalog deferred MoE mode differs from eager request");
    if (config.batched_moe_activation != request.batched_moe_activation ||
        (config.batched_moe_activation && (!config.q4k_a8g32_gate_up || config.enable_device_graph_inputs || request.require_graph_safe)))
        throw std::invalid_argument("HIP catalog batched activation requires matching eager A8 mode");
    if (config.q51_slot_parallel_down != request.q51_slot_parallel_down ||
        (config.q51_slot_parallel_down && (!config.grouped_exact_q51_down || !config.q4k_a8g32_gate_up ||
         !request.grouped_exact_q51_down || !request.q4k_a8g32_gate_up ||
         config.enable_device_graph_inputs || request.device_graph_inputs || request.require_graph_safe || alternate_lm_head ||
         request.logical_rows != 1 || request.request_concurrency != 1)))
        throw std::invalid_argument("HIP catalog slot-parallel Q5_1 requires matching grouped eager C1 A8 mode");
    if(config.grouped_exact_q51_down != request.grouped_exact_q51_down)
        throw std::invalid_argument("HIP catalog grouped exact Q5_1 mode differs from binding request");
    if (config.q4k_a8g32_gate_up != request.q4k_a8g32_gate_up)
        throw std::invalid_argument("HIP catalog Q4K A8 mode differs from binding request");
    if (config.parallel_moe_router != request.parallel_moe_router)
        throw std::invalid_argument("HIP catalog router mode differs from binding request");
    if (config.enable_device_graph_inputs != request.device_graph_inputs)
        throw std::invalid_argument("HIP catalog and graph topology modes differ");
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "complete HIP kernel catalog currently targets C1 one-row decode");
    }
    if (config.activation !=
        HipActivationStorage::bf16) {
        throw std::invalid_argument(
            "complete HIP kernel catalog currently targets canonical BF16 activation mode");
    }
    if (request.minimum_qualification !=
        kernels::QualificationStatus::experimental) {
        throw std::invalid_argument(
            "complete HIP kernel catalog remains experimental until indexed-attention tie semantics and native target qualification are closed");
    }

    const auto qualification =
        kernels::QualificationStatus::experimental;

    const auto elementwise =
        elementwise_config(
            model,
            config.activation);
    register_hip_elementwise_kernel_variants(
        model,
        request,
        elementwise,
        registry,
        qualification);

    const auto ple =
        ple_config(
            model,
            config.activation, config.enable_device_graph_inputs);
    register_hip_ple_kernel_variant(
        model,
        request,
        ple,
        registry,
        qualification);

    const auto gdn = make_gdn_config(model, config.activation,
        config.enable_device_graph_inputs, config.virtual_q8_gdn_input, config.prefetch_q8_gdn_input, config.virtual_q8_gdn_output, config.wave8_q8_gdn_input, config.wave8_q8_gdn_output, config.gdn_recurrent_layout, config.gdn_recurrent_storage, config.dense_a8);
    register_hip_gdn_kernel_variant(
        model,
        request,
        gdn,
        registry,
        qualification);

    const auto indexed =
        indexed_attention_config(
            model,
            config.activation, config.enable_device_graph_inputs, config.prefetch_q8_qsa_input, config.qsa_input_prefetch_layers, config.qsa_input_prefetch_stats, config.wave8_q8_qsa_output, config.qsa_hierarchical_selection, config.qsa_selection_stats, config.qsa_gqa_heads_per_block, config.qsa_gqa_stats, config.dense_a8);
    if(config.prefetch_q8_qsa_input){auto base=indexed;base.prefetch_q8_input=false;base.prefetch_eligible_layers.clear();auto base_request=request;base_request.prefetch_q8_qsa_input=false;base_request.qsa_input_prefetch_layers.clear();register_hip_indexed_attention_kernel_variant(model,base_request,base,registry,qualification);}
    register_hip_indexed_attention_kernel_variant(
        model,
        request,
        indexed,
        registry,
        qualification);

    const auto embedding =
        embedding_config(
            model,
            config.activation, config.enable_device_graph_inputs);
    register_hip_embedding_kernel_variant(
        model,
        request,
        embedding,
        registry,
        qualification);

    auto hyper_mix =
        hyper_mix_config(
            model,
            config.activation, config.parallel_hyper_mix, config.staged_hyper_projections, config.fused_staged_hyper_mix, config.typed_hyper_projection, config.prefetch_typed_hyper_rows, config.prefetch_q8_hyper_output, config.lane_chain_hyper_output,
            config.splitk_hyper_projection, config.splitk_hyper_projection_fp64, config.prefetch_hc_projection_next_tile, config.arithmetic_stats, config.splitk_deferred_norm);
    register_hip_hyper_mix_kernel_variant(
        model,
        request,
        hyper_mix,
        registry,
        qualification);
    if(config.lane_chain_hyper_output && !config.splitk_deferred_norm){hyper_mix.lane_chain_output=false;register_hip_hyper_mix_kernel_variant(model,request,hyper_mix,registry,qualification);}

    const auto moe_route =
        moe_route_config(
            model,
            config.activation, config.parallel_moe_router, config.ranked_moe_router_topk, config.wave_moe_router_topk, config.wave_router_manifest_admission, config.staged_moe_router_logits);
    register_hip_moe_route_kernel_variant(
        model,
        request,
        moe_route,
        registry,
        qualification);

    const auto moe_experts =
        moe_experts_config(
            model,
            config.activation, config.q4k_a8g32_gate_up, config.q4k_a8_stats, config.grouped_exact_q51_down, config.deferred_moe_status, config.batched_moe_activation, config.q51_slot_parallel_down, config.a8_down_q51, config.a8_down_q80, config.q5k_a8g32_gate_up);
    register_hip_moe_experts_kernel_variant(
        model,
        request,
        moe_experts,
        registry,
        qualification);

    if (alternate_lm_head) {
        integer_projection::register_integer_projection_runtime_variant(model, request, *alternate_lm_head, registry);
    } else {
    const auto lm_head =
        lm_head_config(
            model,
            config.activation, config.virtual_q8_lm_head, config.prefetch_q8_lm_head, config.wave8_q8_lm_head);
    register_hip_lm_head_kernel_variant(
        model,
        request,
        lm_head,
        registry,
        qualification);
    }
    const auto shared =
        shared_expert_config(
            model,
            config.activation,config.prefetch_q8_shared_expert,config.shared_expert_prefetch_layers,{},config.dense_a8);
    if(config.prefetch_q8_shared_expert){auto base=shared;base.prefetch_q8=false;base.prefetch_eligible_layers.clear();auto base_request=request;base_request.prefetch_q8_shared_expert=false;base_request.shared_expert_prefetch_layers.clear();register_hip_shared_expert_kernel_variant(model,base_request,base,registry,qualification);}
    register_hip_shared_expert_kernel_variant(
        model,
        request,
        shared,
        registry,
        qualification);
}

HipLauncherCatalog::HipLauncherCatalog(
    const ModelContract& model,
    HipLauncherCatalogConfig config,
    std::shared_ptr<integer_projection::HipIntegerProjectionRuntimeBundle> alternate_lm_head)
    : device_graph_inputs_(config.enable_device_graph_inputs), deferred_moe_status_(config.deferred_moe_status), splitk_deferred_norm_(config.splitk_deferred_norm), alternate_lm_head_(std::move(alternate_lm_head)), elementwise_(
          elementwise_config(
              model,
              config.activation)),
      ple_(
          ple_config(
              model,
              config.activation, config.enable_device_graph_inputs)),
      gdn_config_(
          make_gdn_config(model, config.activation,
              config.enable_device_graph_inputs, config.virtual_q8_gdn_input, config.prefetch_q8_gdn_input, config.virtual_q8_gdn_output, config.wave8_q8_gdn_input, config.wave8_q8_gdn_output, config.gdn_recurrent_layout, config.gdn_recurrent_storage, config.dense_a8,
              (config.fast_prefill_components & kFastPrefillDense) != 0U)),
      gdn_(gdn_config_),
      indexed_attention_(
          indexed_attention_config(
              model,
              config.activation, config.enable_device_graph_inputs, config.prefetch_q8_qsa_input, config.qsa_input_prefetch_layers, config.qsa_input_prefetch_stats, config.wave8_q8_qsa_output, config.qsa_hierarchical_selection, config.qsa_selection_stats, config.qsa_gqa_heads_per_block, config.qsa_gqa_stats, config.dense_a8,
              (config.fast_prefill_components & kFastPrefillDense) != 0U)),
      embedding_(
          embedding_config(
              model,
              config.activation, config.enable_device_graph_inputs)),
      hyper_mix_(
          hyper_mix_config(
              model,
              config.activation, config.parallel_hyper_mix, config.staged_hyper_projections, config.fused_staged_hyper_mix, config.typed_hyper_projection, config.prefetch_typed_hyper_rows, config.prefetch_q8_hyper_output, config.lane_chain_hyper_output,
              config.splitk_hyper_projection, config.splitk_hyper_projection_fp64, config.prefetch_hc_projection_next_tile, config.arithmetic_stats, config.splitk_deferred_norm,
              (config.fast_prefill_components & kFastPrefillHc) != 0U)),
      moe_route_(
          moe_route_config(
              model,
              config.activation, config.parallel_moe_router, config.ranked_moe_router_topk, config.wave_moe_router_topk, config.wave_router_manifest_admission, config.staged_moe_router_logits)),
      moe_experts_(
          moe_experts_config(
              model,
              config.activation, config.q4k_a8g32_gate_up, config.q4k_a8_stats, config.grouped_exact_q51_down, config.deferred_moe_status, config.batched_moe_activation, config.q51_slot_parallel_down, config.a8_down_q51, config.a8_down_q80, config.q5k_a8g32_gate_up)),
      lm_head_(
          lm_head_config(
              model,
              config.activation, config.virtual_q8_lm_head, config.prefetch_q8_lm_head, config.wave8_q8_lm_head)),
      shared_expert_(
          shared_expert_config(
              model,
              config.activation,config.prefetch_q8_shared_expert,config.shared_expert_prefetch_layers,
              config.shared_expert_prefetch_stats, config.dense_a8, (config.fast_prefill_components & kFastPrefillDense) != 0U)) {
    validate_numerical_catalog_config(model, config);
    if (config.numerical_profile != DecodeNumericalProfile::exact_v5 && alternate_lm_head_)
        throw std::invalid_argument("HIP catalog numerical profile excludes alternate integer head");
    validate_gdn_recurrent_storage(config.gdn_recurrent_storage);
    if (config.gdn_recurrent_storage != GdnRecurrentStorage::key_value && config.enable_device_graph_inputs)
        throw std::invalid_argument("GDN recurrent storage candidate requires eager execution");
    (void)gdn_recurrent_layout_marker(config.gdn_recurrent_layout);
    if (config.gdn_recurrent_storage != GdnRecurrentStorage::key_value && alternate_lm_head_)
        throw std::invalid_argument("GDN recurrent storage candidate excludes alternate integer head");
    if (config.gdn_recurrent_layout != HipGdnRecurrentLayout::baseline &&
        (config.enable_device_graph_inputs || alternate_lm_head_))
        throw std::invalid_argument("GDN recurrent LDS layouts support eager floating execution only");
    (void)qsa_gqa_organization_marker(config.qsa_gqa_heads_per_block);
    if (config.qsa_gqa_heads_per_block != 1 &&
        (config.enable_device_graph_inputs || alternate_lm_head_ || !config.deferred_moe_status))
        throw std::invalid_argument("QSA GQA catalog requires eager deferred completion without integer head");
    if (config.qsa_hierarchical_selection &&
        (config.enable_device_graph_inputs || alternate_lm_head_ || !config.deferred_moe_status))
        throw std::invalid_argument("hierarchical QSA catalog requires eager deferred completion without integer head");
    if (config.q51_slot_parallel_down && alternate_lm_head_)
        throw std::invalid_argument("HIP catalog slot-parallel Q5_1 excludes an integer LM head");
    if ((config.prefetch_q8_lm_head || config.wave8_q8_lm_head || config.wave8_q8_gdn_input ||
         config.wave8_q8_gdn_output || config.wave8_q8_qsa_output) &&
        (config.enable_device_graph_inputs || alternate_lm_head_))
        throw std::invalid_argument("HIP catalog Q8 organizations support eager floating execution only");
    if (alternate_lm_head_ && config.virtual_q8_lm_head)
        throw std::invalid_argument("integer and virtual Q8 LM heads are mutually exclusive");
    if (device_graph_inputs_ && alternate_lm_head_)
        throw std::invalid_argument("integer LM-head catalog is eager only");
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
}

void HipLauncherCatalog::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry,
    const ExecutionRuntimeResources* resources,
    std::shared_ptr<HipGdnIslands> gdn_islands) {
    if (splitk_deferred_norm_) validate_deferred_hc_catalog_model(compiled,resources);
    if (gdn_islands && (device_graph_inputs_ || alternate_lm_head_ || !deferred_moe_status_ || !resources ||
        (gdn_config_.wave8_q8_input_projections != gdn_config_.wave8_q8_output_projection) ||
        !gdn_island_recurrent_layout_admitted(gdn_config_.recurrent_layout)))
        throw std::invalid_argument("GDN islands require eager floating execution with retained D1 resources");
    if (compiled.device_graph_inputs != device_graph_inputs_)
        throw std::invalid_argument("launcher catalog graph mode differs from compiled plan");
    // Retain before publishing any borrowed island userdata, including partial registration failure.
    if (gdn_islands) gdn_island_owners_.push_back(gdn_islands);
    if (device_graph_inputs_) register_hip_graph_position_launcher(registry);
    else register_default_runtime_launchers(registry);
    elementwise_.
        register_for_compiled_execution(
            compiled,
            registry);
    ple_.
        register_for_compiled_execution(
            compiled,
            registry);
    gdn_.
        register_for_compiled_execution(
            compiled,
            registry, gdn_islands.get());
    indexed_attention_.
        register_for_compiled_execution(
            compiled,
            registry);
    embedding_.
        register_for_compiled_execution(
            compiled,
            registry);
    hyper_mix_.
        register_for_compiled_execution(
            compiled,
            registry, resources);
    moe_route_.
        register_for_compiled_execution(
            compiled,
            registry);
    moe_experts_.
        register_for_compiled_execution(
            compiled,
            registry);
    if (alternate_lm_head_) alternate_lm_head_->register_for_compiled_execution(compiled, registry);
    else lm_head_.register_for_compiled_execution(compiled, registry);
    shared_expert_.
        register_for_compiled_execution(
            compiled,
            registry);
}

LauncherCoverage
HipLauncherCatalog::audit(
    const CompiledSerialExecution& compiled,
    const ExecutionLauncherRegistry& registry) const {
    return audit_launcher_coverage(
        compiled,
        registry);
}

std::vector<std::uint64_t> HipLauncherCatalog::known_window_position_readers(
    const CompiledSerialExecution& compiled, const ExecutionLauncherRegistry& registry) const {
    if (device_graph_inputs_ || alternate_lm_head_)
        throw std::invalid_argument("known window position readers require the eager floating catalog");
    return indexed_attention_.known_window_position_readers(compiled, registry);
}

}  // namespace strix
