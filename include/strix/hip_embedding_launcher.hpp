#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipEmbeddingLauncherConfig {
    std::uint32_t hidden_size = 0;
    HipActivationStorage output =
        HipActivationStorage::bf16;
};

void register_hip_embedding_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipEmbeddingLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_embedding_launcher(
    std::string kernel_id,
    HipEmbeddingLauncherConfig* config);

class HipEmbeddingLauncherBundle {
public:
    explicit HipEmbeddingLauncherBundle(
        HipEmbeddingLauncherConfig config)
        : config_(config) {}

    HipEmbeddingLauncherBundle(
        const HipEmbeddingLauncherBundle&) = delete;
    HipEmbeddingLauncherBundle& operator=(
        const HipEmbeddingLauncherBundle&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipEmbeddingLauncherConfig config_;
};

}  // namespace strix
