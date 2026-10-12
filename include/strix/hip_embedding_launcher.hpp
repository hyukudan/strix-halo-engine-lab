#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/graph_inputs.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipEmbeddingLauncherConfig {
    std::uint32_t hidden_size = 0;
    HipActivationStorage output =
        HipActivationStorage::bf16;
};

// Experimental C1 device-input gather. All allocations must remain owned and
// unchanged through stream completion. status is overwritten on every launch:
// 0=valid, 1=token out of range, 2=position out of range (flags may combine).
// This separate API does not promote the existing eager whole-model launcher.
void launch_hip_graph_embedding(
    const HipEmbeddingLauncherConfig& config, std::uint32_t ggml_type,
    const RuntimeTensorBinding& weights, std::uint32_t vocabulary_size,
    std::uint64_t context_tokens, const GraphDynamicArguments& arguments,
    const ExecutionBufferAddress& output, const ExecutionMemoryBinding& status,
    void* stream);

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
