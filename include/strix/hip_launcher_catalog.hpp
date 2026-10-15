#pragma once

#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_ple_launcher.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"
#include <memory>

namespace strix {
namespace integer_projection { struct RuntimeConfig; class HipIntegerProjectionRuntimeBundle; }

struct HipLauncherCatalogConfig {
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
    bool parallel_hyper_mix = false;
    bool staged_hyper_projections = false;
    bool fused_staged_hyper_mix = false;
    bool typed_hyper_projection = false;
    bool prefetch_typed_hyper_rows = false;
    bool virtual_q8_lm_head = false;
    bool parallel_moe_router = false;
    bool virtual_q8_gdn_input = false;
    bool ranked_moe_router_topk = false;
    bool prefetch_q8_gdn_input = false;
};

void register_complete_hip_kernel_catalog(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    HipLauncherCatalogConfig config,
    kernels::KernelRegistry& registry,
    const integer_projection::RuntimeConfig* alternate_lm_head = nullptr);

class HipLauncherCatalog {
public:
    HipLauncherCatalog(
        const ModelContract& model,
        HipLauncherCatalogConfig config = {},
        std::shared_ptr<integer_projection::HipIntegerProjectionRuntimeBundle> alternate_lm_head = {});

    HipLauncherCatalog(
        const HipLauncherCatalog&) = delete;
    HipLauncherCatalog& operator=(
        const HipLauncherCatalog&) = delete;
    HipLauncherCatalog(
        HipLauncherCatalog&&) = delete;
    HipLauncherCatalog& operator=(
        HipLauncherCatalog&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry,
        const ExecutionRuntimeResources* resources = nullptr);

    [[nodiscard]] LauncherCoverage audit(
        const CompiledSerialExecution& compiled,
        const ExecutionLauncherRegistry& registry) const;

private:
    bool device_graph_inputs_ = false;
    std::shared_ptr<integer_projection::HipIntegerProjectionRuntimeBundle> alternate_lm_head_;
    HipElementwiseLauncherBundle elementwise_;
    HipPleLauncherBundle ple_;
    HipGdnLauncherBundle gdn_;
    HipIndexedAttentionLauncherBundle indexed_attention_;
    HipEmbeddingLauncherBundle embedding_;
    HipHyperMixLauncherBundle hyper_mix_;
    HipMoeRouteLauncherBundle moe_route_;
    HipMoeExpertsLauncherBundle moe_experts_;
    HipLmHeadLauncherBundle lm_head_;
    HipSharedExpertLauncherBundle shared_expert_;
};

}  // namespace strix
