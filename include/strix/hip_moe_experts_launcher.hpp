#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>
#include <memory>
#include "strix/moe_q4k_a8_runtime.hpp"

namespace strix {
class HipSetupQuiescenceGuard;

struct HipMoeExpertsLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t expert_count = 0;
    std::uint32_t top_k = 0;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool q4k_a8g32_gate_up = false;
    bool batched_slot_activation = false;
    bool grouped_exact_q51_down = false;
    bool q51_slot_parallel_down = false;
    bool deferred_moe_status = false;
    std::shared_ptr<const HipSetupQuiescenceGuard> q4k_a8_quiescence;
    std::shared_ptr<moe_q4k_a8::RuntimeDispatchStats> q4k_a8_stats;
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
