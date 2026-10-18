#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>
#include <memory>
#include "strix/moe_q4k_a8_runtime.hpp"
#include "strix/q51_known_rows.hpp"
#include <span>

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
    // Explicit experimental arithmetic; defaults retain every original path.
    bool a8_down_q51 = false;
    bool a8_down_q80 = false;
    bool q5k_a8g32_gate_up = false;
    // fast-prefill-f16wmma-v1: known-prompt prefill blocks (T rows) run the routed MoE as FP16
    // WMMA (known_rows::enqueue_prefill form 3). Serial and window rows are unaffected.
    bool prefill_f16wmma = false;
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

// Explicit retained known-prefix window R2/R4/R8 organization, not an R1
// variant or automatic strategy. Requires the selected Q4K A8 / Q5_1
// slot-parallel layer, BF16 C1 rows and owned eager D1 scope. Quantization,
// gate/up and SiLU retain their incumbent per-row kernels. Device CSR groups
// only selected assignments; down reuses each decoded coefficient across the
// expert's rows and scatters raw dots before the original ordered slot FMA.
// Every row and the exclusive CSR owner are admitted before any enqueue.
// The caller admits session/epoch, holds exclusive allocation leases, retains
// all descriptors/owners and drains the stream before reuse or destruction.
// No host payload reads, allocation, synchronization, capture or publication.
void moe_experts_known_window_rows_preflight(
    std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config,
    const q51_known_rows::Workspace& workspace);
void moe_experts_known_window_rows_launch(
    std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config,
    const q51_known_rows::Workspace& workspace);
// Known-prompt prefill, A8 profile: all T rows (multiple of 8) of one routed step in one
// call. Every 8-row chunk passes the window admission above (with the window's own
// 8-row workspace); each per-row view must then be row 0 plus r times one stride.
// One CSR over the T rows in `prefill` (q51_known_rows::prefill_scratch_bytes(T)), so
// each touched expert's weights are read once per batch of its assignments. Each
// (row, slot) and each row's slot merge keep the window arithmetic and order.
void moe_experts_known_prefill_rows_preflight(
    std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config,
    const q51_known_rows::Workspace& window, const q51_known_rows::Workspace& prefill);
void moe_experts_known_prefill_rows_launch(
    std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config,
    const q51_known_rows::Workspace& window, const q51_known_rows::Workspace& prefill);

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
