#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipLmHeadLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t vocab_size = 0;
    HipActivationStorage input =
        HipActivationStorage::bf16;
};

void register_hip_lm_head_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipLmHeadLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_lm_head_launcher(
    std::string kernel_id,
    HipLmHeadLauncherConfig* config);

class HipLmHeadLauncherBundle {
public:
    explicit HipLmHeadLauncherBundle(
        HipLmHeadLauncherConfig config)
        : config_(config) {}

    HipLmHeadLauncherBundle(
        const HipLmHeadLauncherBundle&) = delete;
    HipLmHeadLauncherBundle& operator=(
        const HipLmHeadLauncherBundle&) = delete;
    HipLmHeadLauncherBundle(
        HipLmHeadLauncherBundle&&) = delete;
    HipLmHeadLauncherBundle& operator=(
        HipLmHeadLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipLmHeadLauncherConfig config_;
};

}  // namespace strix
