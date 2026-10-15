#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipGdnLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
    bool virtual_q8_input_projections = false;
};

void register_hip_gdn_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_gdn_launcher(
    std::string kernel_id,
    HipGdnLauncherConfig* config);

class HipGdnLauncherBundle {
public:
    explicit HipGdnLauncherBundle(
        HipGdnLauncherConfig config)
        : config_(config) {}

    HipGdnLauncherBundle(
        const HipGdnLauncherBundle&) = delete;
    HipGdnLauncherBundle& operator=(
        const HipGdnLauncherBundle&) = delete;
    HipGdnLauncherBundle(
        HipGdnLauncherBundle&&) = delete;
    HipGdnLauncherBundle& operator=(
        HipGdnLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipGdnLauncherConfig config_;
};

}  // namespace strix
