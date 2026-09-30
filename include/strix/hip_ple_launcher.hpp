#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipPleLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
};

void register_hip_ple_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipPleLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_ple_launcher(
    std::string kernel_id,
    HipPleLauncherConfig* config);

class HipPleLauncherBundle {
public:
    explicit HipPleLauncherBundle(
        HipPleLauncherConfig config)
        : config_(config) {}

    HipPleLauncherBundle(
        const HipPleLauncherBundle&) = delete;
    HipPleLauncherBundle& operator=(
        const HipPleLauncherBundle&) = delete;
    HipPleLauncherBundle(
        HipPleLauncherBundle&&) = delete;
    HipPleLauncherBundle& operator=(
        HipPleLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipPleLauncherConfig config_;
};

}  // namespace strix
