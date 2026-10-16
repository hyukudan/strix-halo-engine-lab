#include "strix/hip_launcher_catalog.hpp"

#include "strix/runtime_launchers.hpp"
#include "strix/hip_integer_projection_runtime.hpp"

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



HipGdnLauncherConfig gdn_config(const ModelContract& model,
    HipActivationStorage activation, bool graph, bool virtual_q8, bool prefetch_q8, bool virtual_output,
    bool wave8, bool wave8_output) {
    HipGdnLauncherConfig config;
    config.model = model;
    config.activation = activation;
    config.enable_device_graph_inputs = graph;
    config.virtual_q8_input_projections = virtual_q8;
    config.prefetch_q8_input_projections = prefetch_q8;
    config.wave8_q8_input_projections = wave8;
    config.virtual_q8_output_projection = virtual_output;
    config.wave8_q8_output_projection = wave8_output;
    return config;
}

HipIndexedAttentionLauncherConfig
indexed_attention_config(
    const ModelContract& model,
    HipActivationStorage activation, bool graph = false, bool prefetch = false,
    std::vector<std::uint32_t> layers = {}, std::shared_ptr<qsa_input_prefetch::RuntimeDispatchStats> stats = {},
    bool wave8_output = false) {
    HipIndexedAttentionLauncherConfig config;
    config.model = model;
    config.activation = activation;
    config.enable_device_graph_inputs = graph;
    config.prefetch_q8_input = prefetch;
    config.prefetch_eligible_layers=std::move(layers);
    config.prefetch_stats=std::move(stats);
    config.wave8_q8_output = wave8_output;
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
    bool splitk, bool splitk_fp64) {
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
    config.splitk_projection_fp64 = splitk_fp64;
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
    HipActivationStorage activation, bool prefetch=false, std::vector<std::uint32_t> layers={}, std::shared_ptr<shared_expert_prefetch::RuntimeDispatchStats> stats={}) {
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
    return config;
}


HipMoeExpertsLauncherConfig
moe_experts_config(
    const ModelContract& model,
    HipActivationStorage activation, bool q4k_a8 = false,
    std::shared_ptr<moe_q4k_a8::RuntimeDispatchStats> stats = {}, bool grouped_q51=false, bool deferred_status=false, bool batched_activation=false) {
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
    config.deferred_moe_status = deferred_status;
    config.batched_slot_activation = batched_activation;
    config.q4k_a8_stats = std::move(stats);
    return config;
}

HipMoeRouteLauncherConfig
moe_route_config(
    const ModelContract& model,
    HipActivationStorage activation, bool parallel = false, bool ranked = false, bool wave = false,
    std::shared_ptr<const WaveRouterManifestAdmission> admission = {}) {
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
    config.ranked_topk = ranked;
    config.wave_topk = wave;
    config.wave_manifest_admission = std::move(admission);
    return config;
}

}  // namespace


void register_complete_hip_kernel_catalog(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    HipLauncherCatalogConfig config,
    kernels::KernelRegistry& registry,
    const integer_projection::RuntimeConfig* alternate_lm_head) {
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
    if (config.deferred_moe_status != request.deferred_moe_status ||
        (config.deferred_moe_status && (config.enable_device_graph_inputs || request.require_graph_safe)))
        throw std::invalid_argument("HIP catalog deferred MoE mode differs from eager request");
    if (config.batched_moe_activation != request.batched_moe_activation ||
        (config.batched_moe_activation && (!config.q4k_a8g32_gate_up || config.enable_device_graph_inputs || request.require_graph_safe)))
        throw std::invalid_argument("HIP catalog batched activation requires matching eager A8 mode");
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

    const auto gdn = gdn_config(model, config.activation,
        config.enable_device_graph_inputs, config.virtual_q8_gdn_input, config.prefetch_q8_gdn_input, config.virtual_q8_gdn_output, config.wave8_q8_gdn_input, config.wave8_q8_gdn_output);
    register_hip_gdn_kernel_variant(
        model,
        request,
        gdn,
        registry,
        qualification);

    const auto indexed =
        indexed_attention_config(
            model,
            config.activation, config.enable_device_graph_inputs, config.prefetch_q8_qsa_input, config.qsa_input_prefetch_layers, config.qsa_input_prefetch_stats, config.wave8_q8_qsa_output);
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
            config.splitk_hyper_projection, config.splitk_hyper_projection_fp64);
    register_hip_hyper_mix_kernel_variant(
        model,
        request,
        hyper_mix,
        registry,
        qualification);
    if(config.lane_chain_hyper_output){hyper_mix.lane_chain_output=false;register_hip_hyper_mix_kernel_variant(model,request,hyper_mix,registry,qualification);}

    const auto moe_route =
        moe_route_config(
            model,
            config.activation, config.parallel_moe_router, config.ranked_moe_router_topk, config.wave_moe_router_topk, config.wave_router_manifest_admission);
    register_hip_moe_route_kernel_variant(
        model,
        request,
        moe_route,
        registry,
        qualification);

    const auto moe_experts =
        moe_experts_config(
            model,
            config.activation, config.q4k_a8g32_gate_up, config.q4k_a8_stats, config.grouped_exact_q51_down, config.deferred_moe_status, config.batched_moe_activation);
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
            config.activation,config.prefetch_q8_shared_expert,config.shared_expert_prefetch_layers);
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
    : device_graph_inputs_(config.enable_device_graph_inputs), alternate_lm_head_(std::move(alternate_lm_head)), elementwise_(
          elementwise_config(
              model,
              config.activation)),
      ple_(
          ple_config(
              model,
              config.activation, config.enable_device_graph_inputs)),
      gdn_(
          gdn_config(model, config.activation,
              config.enable_device_graph_inputs, config.virtual_q8_gdn_input, config.prefetch_q8_gdn_input, config.virtual_q8_gdn_output, config.wave8_q8_gdn_input, config.wave8_q8_gdn_output)),
      indexed_attention_(
          indexed_attention_config(
              model,
              config.activation, config.enable_device_graph_inputs, config.prefetch_q8_qsa_input, config.qsa_input_prefetch_layers, config.qsa_input_prefetch_stats, config.wave8_q8_qsa_output)),
      embedding_(
          embedding_config(
              model,
              config.activation, config.enable_device_graph_inputs)),
      hyper_mix_(
          hyper_mix_config(
              model,
              config.activation, config.parallel_hyper_mix, config.staged_hyper_projections, config.fused_staged_hyper_mix, config.typed_hyper_projection, config.prefetch_typed_hyper_rows, config.prefetch_q8_hyper_output, config.lane_chain_hyper_output,
              config.splitk_hyper_projection, config.splitk_hyper_projection_fp64)),
      moe_route_(
          moe_route_config(
              model,
              config.activation, config.parallel_moe_router, config.ranked_moe_router_topk, config.wave_moe_router_topk, config.wave_router_manifest_admission)),
      moe_experts_(
          moe_experts_config(
              model,
              config.activation, config.q4k_a8g32_gate_up, config.q4k_a8_stats, config.grouped_exact_q51_down, config.deferred_moe_status, config.batched_moe_activation)),
      lm_head_(
          lm_head_config(
              model,
              config.activation, config.virtual_q8_lm_head, config.prefetch_q8_lm_head, config.wave8_q8_lm_head)),
      shared_expert_(
          shared_expert_config(
              model,
              config.activation,config.prefetch_q8_shared_expert,config.shared_expert_prefetch_layers,
              config.shared_expert_prefetch_stats)) {
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
    const ExecutionRuntimeResources* resources) {
    if (compiled.device_graph_inputs != device_graph_inputs_)
        throw std::invalid_argument("launcher catalog graph mode differs from compiled plan");
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
            registry);
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

}  // namespace strix
