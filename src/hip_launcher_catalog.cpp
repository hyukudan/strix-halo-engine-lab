#include "strix/hip_launcher_catalog.hpp"

#include "strix/runtime_launchers.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
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



HipIndexedAttentionLauncherConfig
indexed_attention_config(
    const ModelContract& model,
    HipActivationStorage activation) {
    HipIndexedAttentionLauncherConfig config;
    config.model = model;
    config.activation = activation;
    return config;
}

HipPleLauncherConfig
ple_config(
    const ModelContract& model,
    HipActivationStorage activation) {
    HipPleLauncherConfig config;
    config.model = model;
    config.activation =
        activation;
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
    HipActivationStorage activation) {
    HipEmbeddingLauncherConfig config;
    config.hidden_size =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    config.output =
        activation;
    return config;
}

HipHyperMixLauncherConfig
hyper_mix_config(
    const ModelContract& model,
    HipActivationStorage activation) {
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
    return config;
}


HipLmHeadLauncherConfig
lm_head_config(
    const ModelContract& model,
    HipActivationStorage activation) {
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
    HipActivationStorage activation) {
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
    return config;
}

HipMoeRouteLauncherConfig
moe_route_config(
    const ModelContract& model,
    HipActivationStorage activation) {
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
    return config;
}

}  // namespace


void register_complete_hip_kernel_catalog(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    HipLauncherCatalogConfig config,
    kernels::KernelRegistry& registry) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
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
            config.activation);
    register_hip_ple_kernel_variant(
        model,
        request,
        ple,
        registry,
        qualification);

    const HipGdnLauncherConfig gdn{
        model,
        config.activation,
    };
    register_hip_gdn_kernel_variant(
        model,
        request,
        gdn,
        registry,
        qualification);

    const auto indexed =
        indexed_attention_config(
            model,
            config.activation);
    register_hip_indexed_attention_kernel_variant(
        model,
        request,
        indexed,
        registry,
        qualification);

    const auto embedding =
        embedding_config(
            model,
            config.activation);
    register_hip_embedding_kernel_variant(
        model,
        request,
        embedding,
        registry,
        qualification);

    const auto hyper_mix =
        hyper_mix_config(
            model,
            config.activation);
    register_hip_hyper_mix_kernel_variant(
        model,
        request,
        hyper_mix,
        registry,
        qualification);

    const auto moe_route =
        moe_route_config(
            model,
            config.activation);
    register_hip_moe_route_kernel_variant(
        model,
        request,
        moe_route,
        registry,
        qualification);

    const auto moe_experts =
        moe_experts_config(
            model,
            config.activation);
    register_hip_moe_experts_kernel_variant(
        model,
        request,
        moe_experts,
        registry,
        qualification);

    const auto lm_head =
        lm_head_config(
            model,
            config.activation);
    register_hip_lm_head_kernel_variant(
        model,
        request,
        lm_head,
        registry,
        qualification);

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
    HipLauncherCatalogConfig config)
    : elementwise_(
          elementwise_config(
              model,
              config.activation)),
      ple_(
          ple_config(
              model,
              config.activation)),
      gdn_(
          HipGdnLauncherConfig{
              model,
              config.activation,
          }),
      indexed_attention_(
          indexed_attention_config(
              model,
              config.activation)),
      embedding_(
          embedding_config(
              model,
              config.activation)),
      hyper_mix_(
          hyper_mix_config(
              model,
              config.activation)),
      moe_route_(
          moe_route_config(
              model,
              config.activation)),
      moe_experts_(
          moe_experts_config(
              model,
              config.activation)),
      lm_head_(
          lm_head_config(
              model,
              config.activation)),
      shared_expert_(
          shared_expert_config(
              model,
              config.activation)) {
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
    ExecutionLauncherRegistry& registry) {
    register_default_runtime_launchers(
        registry);
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
            registry);
    moe_route_.
        register_for_compiled_execution(
            compiled,
            registry);
    moe_experts_.
        register_for_compiled_execution(
            compiled,
            registry);
    lm_head_.
        register_for_compiled_execution(
            compiled,
            registry);
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
