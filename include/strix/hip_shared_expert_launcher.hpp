#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipSharedExpertLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
};

void register_hip_shared_expert_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipSharedExpertLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_shared_expert_launcher(
    std::string kernel_id,
    HipSharedExpertLauncherConfig* config);

class HipSharedExpertLauncherBundle {
public:
    explicit HipSharedExpertLauncherBundle(
        HipSharedExpertLauncherConfig config)
        : config_(config) {}

    HipSharedExpertLauncherBundle(
        const HipSharedExpertLauncherBundle&) = delete;
    HipSharedExpertLauncherBundle& operator=(
        const HipSharedExpertLauncherBundle&) = delete;
    HipSharedExpertLauncherBundle(
        HipSharedExpertLauncherBundle&&) = delete;
    HipSharedExpertLauncherBundle& operator=(
        HipSharedExpertLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipSharedExpertLauncherConfig config_;
};

}  // namespace strix
