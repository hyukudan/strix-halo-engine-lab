#pragma once

#include "strix/hip_model_residency.hpp"
#include "strix/mtp_target_checkpoint_plan.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include <hip/hip_runtime_api.h>
#include <memory>
#include <vector>

namespace strix {
class HipMtpTargetWindowCapture;
class HipMtpTargetWindowBatch;
// Exclusive serialized setup/rollback owner. No fake ready frontier is used to
// size pools. The caller retains exclusive access to the same target session,
// stream and state until each synchronous call returns. Device reset recovery
// is unsupported; observed device identity and allocation leases are checked.
class HipQsaCommittedPool;
class HipMtpTargetCheckpoint final {
public:
    HipMtpTargetCheckpoint(const ModelContract&,const CompiledSerialExecution&,
        const SerialSessionStateLayout&,
        std::shared_ptr<const RuntimeExecutionMemoryOwner>,
        std::shared_ptr<const HipModelResidency>,std::shared_ptr<HipRuntimeAllocator>,
        hipStream_t owned_stream,std::shared_ptr<const void> actual_stream_owner,
        std::shared_ptr<const HipSetupQuiescenceGuard>,SessionSnapshotPolicy,
        std::uint64_t retained_byte_budget,AllocationGenerationSource&,
        std::shared_ptr<const HipMtpTargetWindowCapture> bounded_window_producer = {},
        std::shared_ptr<HipQsaCommittedPool> qsa_pool = {});
    ~HipMtpTargetCheckpoint();
    HipMtpTargetCheckpoint(const HipMtpTargetCheckpoint&)=delete;
    HipMtpTargetCheckpoint& operator=(const HipMtpTargetCheckpoint&)=delete;
    void capture(const SerialSessionContext&);
    void capture_window(const SerialSessionContext&);
    // Requires the actual successful whole-chunk producer proof. Cannot be used
    // to recover an uncertain/failed enqueue or a foreign mutation footprint.
    void restore_window(SerialSessionContext&,const HipMtpTargetWindowBatch&);
    // Restores payload first, then republishes exactly the recorded ready
    // metadata and validated arithmetic history, never a synthesized history.
    void restore(SerialSessionContext&);
    void discard();
    [[nodiscard]] bool has_checkpoint() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] std::uint64_t reserved_bytes() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix


