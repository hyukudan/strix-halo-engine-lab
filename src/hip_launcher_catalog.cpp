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
    HipActivationStorage activation, bool graph, bool virtual_q8, bool prefetch_q8) {
    HipGdnLauncherConfig config;
    config.model = model;
    config.activation = activation;
    config.enable_device_graph_inputs = graph;
    config.virtual_q8_input_projections = virtual_q8;
    config.prefetch_q8_input_projections = prefetch_q8;
    return config;
}

HipIndexedAttentionLauncherConfig
indexed_attention_config(
    const ModelContract& model,
    HipActivationStorage activation, bool graph = false) {
    HipIndexedAttentionLauncherConfig config;
    config.model = model;
    config.activation = activation;
    config.enable_device_graph_inputs = graph;
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
    HipActivationStorage activation, bool parallel, bool staged, bool fused, bool typed, bool prefetch_rows) {
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
    return config;
}


HipLmHeadLauncherConfig
lm_head_config(
    const ModelContract& model,
    HipActivationStorage activation, bool virtual_wave) {
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
    return config;
}


HipSharedExpertLauncherConfig
shared_expert_config(
    const ModelContract& model,
    HipActivationStorage activation) {
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
    return config;
}


HipMoeExpertsLauncherConfig
moe_experts_config(
    const ModelContract& model,
    HipActivationStorage activation, bool q4k_a8 = false,
    std::shared_ptr<moe_q4k_a8::RuntimeDispatchStats> stats = {}) {
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
    config.q4k_a8_stats = std::move(stats);
    return config;
}

HipMoeRouteLauncherConfig
moe_route_config(
    const ModelContract& model,
    HipActivationStorage activation, bool parallel = false, bool ranked = false) {
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
    if (config.ranked_moe_router_topk != request.ranked_moe_router_topk ||
        (config.ranked_moe_router_topk && !config.parallel_moe_router))
        throw std::invalid_argument("HIP catalog ranked router mode differs from binding request");
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
        config.enable_device_graph_inputs, config.virtual_q8_gdn_input, config.prefetch_q8_gdn_input);
    register_hip_gdn_kernel_variant(
        model,
        request,
        gdn,
        registry,
        qualification);

    const auto indexed =
        indexed_attention_config(
            model,
            config.activation, config.enable_device_graph_inputs);
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

    const auto hyper_mix =
        hyper_mix_config(
            model,
            config.activation, config.parallel_hyper_mix, config.staged_hyper_projections, config.fused_staged_hyper_mix, config.typed_hyper_projection, config.prefetch_typed_hyper_rows);
    register_hip_hyper_mix_kernel_variant(
        model,
        request,
        hyper_mix,
        registry,
        qualification);

    const auto moe_route =
        moe_route_config(
            model,
            config.activation, config.parallel_moe_router, config.ranked_moe_router_topk);
    register_hip_moe_route_kernel_variant(
        model,
        request,
        moe_route,
        registry,
        qualification);

    const auto moe_experts =
        moe_experts_config(
            model,
            config.activation, config.q4k_a8g32_gate_up, config.q4k_a8_stats);
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
            config.activation, config.virtual_q8_lm_head);
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
            config.activation);
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
              config.enable_device_graph_inputs, config.virtual_q8_gdn_input, config.prefetch_q8_gdn_input)),
      indexed_attention_(
          indexed_attention_config(
              model,
              config.activation, config.enable_device_graph_inputs)),
      embedding_(
          embedding_config(
              model,
              config.activation, config.enable_device_graph_inputs)),
      hyper_mix_(
          hyper_mix_config(
              model,
              config.activation, config.parallel_hyper_mix, config.staged_hyper_projections, config.fused_staged_hyper_mix, config.typed_hyper_projection, config.prefetch_typed_hyper_rows)),
      moe_route_(
          moe_route_config(
              model,
              config.activation, config.parallel_moe_router, config.ranked_moe_router_topk)),
      moe_experts_(
          moe_experts_config(
              model,
              config.activation, config.q4k_a8g32_gate_up, config.q4k_a8_stats)),
      lm_head_(
          lm_head_config(
              model,
              config.activation, config.virtual_q8_lm_head)),
      shared_expert_(
          shared_expert_config(
              model,
              config.activation)) {
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
