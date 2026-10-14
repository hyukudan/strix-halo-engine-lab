#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipHyperMixLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t hc_count = 0;
    std::uint32_t lowrank = 0;
    float rms_epsilon = 1e-6F;
    bool parallel = false;
    bool staged_projections = false;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
};

void register_hip_hyper_mix_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipHyperMixLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_hyper_mix_launcher(
    std::string kernel_id,
    HipHyperMixLauncherConfig* config);

class HipHyperMixLauncherBundle {
public:
    explicit HipHyperMixLauncherBundle(
        HipHyperMixLauncherConfig config)
        : config_(config) {}

    HipHyperMixLauncherBundle(
        const HipHyperMixLauncherBundle&) = delete;
    HipHyperMixLauncherBundle& operator=(
        const HipHyperMixLauncherBundle&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipHyperMixLauncherConfig config_;
};

}  // namespace strix
