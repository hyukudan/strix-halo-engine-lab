#pragma once

#include "strix/distributed_native_backend.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_launch_context.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace strix {
// Explicit controller binding, established at both peers outside this adapter.
// No wire field or connection generation is silently interpreted as a position.
struct HipDistributedStageToken {
    DistributedRuntimeRequest request;
    std::uint64_t position = 0;
    std::uint32_t input_token = 0;
    PersistentStateGenerations initial_state;
    // Explicit suffix-controller mode only: full begin preflight validates the
    // host word extent but does not require its old value to be the new input
    // position yet. The controller snapshots first and materializes position
    // only after both peers prepared. Every actual position reader still
    // checks the exact word. No additional state generation is published.
    bool controller_materializes_position = false;
};
struct HipDistributedStageResult {
    HipDistributedStageToken token;
    PersistentStateGenerations local_state;
    std::uint32_t completed_steps = 0;
    // Provisional local completion only: no peer transaction, serial session
    // commit, arithmetic lineage publication or cross-host rollback authority.
};
// Real eager local executor consumed by NativeDistributedBackend or
// DistributedControlReceiver. First increment deliberately retains a complete
// snapshot and ALL weights locally; it does not implement memory sharding.
// Projection weights are GPU-visible; the sole host-only exception is a
// retained file-backed ngram_table served by the actual FileMappedPleLookupBackend.
// The complete local mapping/residency/state/scratch/D1 budget remains required.
// Exclusive caller ownership covers resources and their reachable metadata.
// Each callback fences/checks D1 before return, since return permits peer ACK.
class HipDistributedStage final : public DistributedNativeLocalExecutor {
public:
    HipDistributedStage(CompiledSerialExecution serial, DistributedPlacementPlan placement,
        std::uint32_t first_step, std::uint32_t last_step, ModelContract model,
        HipLauncherCatalogConfig config, const ExecutionRuntimeResources& resources,
        HipLaunchContext scope, std::shared_ptr<const void> actual_owner);
    ~HipDistributedStage();
    HipDistributedStage(const HipDistributedStage&) = delete;
    HipDistributedStage& operator=(const HipDistributedStage&) = delete;
    // Preflights the entire stage before starting PLE or writing state. Known
    // boundary inputs must already be retained in their exact scratch views.
    // No new token may begin until complete_stage retires the previous one.
    // Reuse keeps the same request/epoch, increasing generation and position+1;
    // restore/reset/reconnect requires a fresh stage owner.
    void begin_token(const HipDistributedStageToken& token);
    void execute_local_step(const DistributedStepPlacement& step) override;
    [[nodiscard]] HipDistributedStageResult complete_stage();
    // Abort after possible mutation is sticky and drains before returning,
    // INCLUDING after complete_stage: local retirement is still provisional.
    // No implicit rollback or peer semantic commit is supplied here. Draining
    // the GPU does not retire external DMA leases; transport retains all
    // exported source owners until its own proven release.
    void abort() noexcept;
    [[nodiscard]] bool failed() const noexcept;
    // Completed producer views only. Caller retains this owner until the
    // transport has released every view. The next begin_token is forbidden by
    // the caller ownership contract until ALL exported views are retired; strong
    // owner retention alone does not prevent overwriting them. An ACK is not a
    // semantic token commit.
    [[nodiscard]] std::vector<ExecutionBufferAddress> completed_boundary_views(
        const DistributedExecutionBoundary& boundary) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
