#pragma once

#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"

namespace strix {

struct HipLauncherCatalogConfig {
    HipActivationStorage activation =
        HipActivationStorage::bf16;
};

class HipLauncherCatalog {
public:
    HipLauncherCatalog(
        const ModelContract& model,
        HipLauncherCatalogConfig config = {});

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
        ExecutionLauncherRegistry& registry);

    [[nodiscard]] LauncherCoverage audit(
        const CompiledSerialExecution& compiled,
        const ExecutionLauncherRegistry& registry) const;

private:
    HipElementwiseLauncherBundle elementwise_;
    HipGdnLauncherBundle gdn_;
    HipEmbeddingLauncherBundle embedding_;
    HipHyperMixLauncherBundle hyper_mix_;
    HipMoeRouteLauncherBundle moe_route_;
    HipMoeExpertsLauncherBundle moe_experts_;
    HipLmHeadLauncherBundle lm_head_;
    HipSharedExpertLauncherBundle shared_expert_;
};

}  // namespace strix
