#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"

#include <string>

namespace strix {

struct HipIndexedAttentionLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
};

void register_hip_indexed_attention_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipIndexedAttentionLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_indexed_attention_launcher(
    std::string kernel_id,
    HipIndexedAttentionLauncherConfig* config);

class HipIndexedAttentionLauncherBundle {
public:
    explicit HipIndexedAttentionLauncherBundle(
        HipIndexedAttentionLauncherConfig config)
        : config_(config) {}

    HipIndexedAttentionLauncherBundle(
        const HipIndexedAttentionLauncherBundle&) = delete;
    HipIndexedAttentionLauncherBundle& operator=(
        const HipIndexedAttentionLauncherBundle&) = delete;
    HipIndexedAttentionLauncherBundle(
        HipIndexedAttentionLauncherBundle&&) = delete;
    HipIndexedAttentionLauncherBundle& operator=(
        HipIndexedAttentionLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipIndexedAttentionLauncherConfig config_;
};

}  // namespace strix
