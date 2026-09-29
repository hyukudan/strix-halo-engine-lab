#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipMoeExpertsLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t expert_count = 0;
    std::uint32_t top_k = 0;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
};

void register_hip_moe_experts_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipMoeExpertsLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_moe_experts_launcher(
    std::string kernel_id,
    HipMoeExpertsLauncherConfig* config);

class HipMoeExpertsLauncherBundle {
public:
    explicit HipMoeExpertsLauncherBundle(
        HipMoeExpertsLauncherConfig config)
        : config_(config) {}

    HipMoeExpertsLauncherBundle(
        const HipMoeExpertsLauncherBundle&) = delete;
    HipMoeExpertsLauncherBundle& operator=(
        const HipMoeExpertsLauncherBundle&) = delete;
    HipMoeExpertsLauncherBundle(
        HipMoeExpertsLauncherBundle&&) = delete;
    HipMoeExpertsLauncherBundle& operator=(
        HipMoeExpertsLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipMoeExpertsLauncherConfig config_;
};

}  // namespace strix
