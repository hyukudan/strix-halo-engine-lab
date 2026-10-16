#pragma once

#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include <hip/hip_runtime_api.h>
#include <memory>
#include <vector>

namespace strix {
class HipMtpTargetWindowCapture;
class HipMtpTargetWindowBatch;
struct HipMtpTargetCheckpointReservation {
    std::uint64_t device_payload_bytes=0,host_payload_bytes=0;
    // Upper bound: full-component pools plus worst-case alignment reservations;
    // excludes driver/allocator metadata and retained upstream target owners.
    std::uint64_t retained_bytes=0;
};
[[nodiscard]] HipMtpTargetCheckpointReservation plan_hip_mtp_target_checkpoint(
    const ModelContract&,const SerialSessionStateLayout&,SessionSnapshotPolicy);

// Pure geometry, not an execution capability. Mutation authority is supplied
// separately by an actual minted closed-window Capture and completed Batch.
struct HipMtpTargetRollbackRange {
    SessionStateComponent component=SessionStateComponent::attention_kv;
    SessionStateStorage storage=SessionStateStorage::device;
    std::uint64_t source_offset=0,snapshot_offset=0,bytes=0;
    bool deferred_status=false; // component/source_offset unused for D1
};
struct HipMtpTargetRollbackPlan {
    std::uint64_t first_position=0;
    std::uint32_t logical_rows=0;
    HipMtpTargetCheckpointReservation reservation;
    std::vector<HipMtpTargetRollbackRange> ranges;
};
[[nodiscard]] HipMtpTargetRollbackPlan plan_hip_mtp_target_window_rollback(
    const ModelContract&,const SerialSessionStateLayout&,std::uint64_t first_position,
    std::uint32_t logical_rows,bool save_deferred_status);

// Exclusive serialized setup/rollback owner. No fake ready frontier is used to
// size pools. The caller retains exclusive access to the same target session,
// stream and state until each synchronous call returns. Device reset recovery
// is unsupported; observed device identity and allocation leases are checked.
class HipMtpTargetCheckpoint final {
public:
    HipMtpTargetCheckpoint(const ModelContract&,const CompiledSerialExecution&,
        const SerialSessionStateLayout&,
        std::shared_ptr<const RuntimeExecutionMemoryOwner>,
        std::shared_ptr<const HipModelResidency>,std::shared_ptr<HipRuntimeAllocator>,
        hipStream_t owned_stream,std::shared_ptr<const void> actual_stream_owner,
        std::shared_ptr<const HipSetupQuiescenceGuard>,SessionSnapshotPolicy,
        std::uint64_t retained_byte_budget,AllocationGenerationSource&,
        std::shared_ptr<const HipMtpTargetWindowCapture> bounded_window_producer = {});
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


