#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"
#include "strix/graph_attention_state.hpp"
#include "strix/qsa_input_prefetch.hpp"

#include <string>
#include <memory>

namespace strix {

struct HipIndexedAttentionLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
    bool prefetch_q8_input = false;
    std::vector<std::uint32_t> prefetch_eligible_layers;
    std::shared_ptr<qsa_input_prefetch::RuntimeDispatchStats> prefetch_stats;
};

enum class HipQsaInputProjectionMode {legacy256,virtual256,prefetch8};
struct HipQsaInputProjectionView {std::uintptr_t address=0;std::uint64_t bytes=0;};
// Standalone diagnostic, no allocation/sync. Caller retains all owners through
// stream drain (also on enqueue failure). Exact disjoint output views, BF16 input,
// original Q8_0 weights and BF16-rounded values represented in FP32 output.
void launch_hip_qsa_input_projection_probe(HipQsaInputProjectionMode,
    HipQsaInputProjectionView input,HipQsaInputProjectionView weights,
    HipQsaInputProjectionView output,std::uint32_t width,std::uint32_t outputs,void* stream);

// Experimental post-projection C1 norm/RoPE/KV stage. Async on caller's
// explicitly owned stream; retain all resource owners through completion.
// Read status only after completion: 0=stage valid, 2=position out of bounds,
// 4=nonfinite source/norm, 8=nonfinite arithmetic. A nonzero status invalidates
// the stage. This does not publish a serial token/state commit.
void launch_hip_graph_attention_state(
    const GraphAttentionStatePlan& plan, std::uint64_t scalar_control_position,
    void* stream);

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
        : config_(config), fallback_config_(config) {fallback_config_.prefetch_q8_input=false;fallback_config_.prefetch_eligible_layers.clear();}

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
    HipIndexedAttentionLauncherConfig fallback_config_;
};

}  // namespace strix
