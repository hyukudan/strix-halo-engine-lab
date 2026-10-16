#pragma once

#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include <hip/hip_runtime_api.h>
#include <memory>

namespace strix {
struct HipMtpTargetCheckpointReservation {
    std::uint64_t device_payload_bytes=0,host_payload_bytes=0;
    // Upper bound: full-component pools plus worst-case alignment reservations;
    // excludes driver/allocator metadata and retained upstream target owners.
    std::uint64_t retained_bytes=0;
};
[[nodiscard]] HipMtpTargetCheckpointReservation plan_hip_mtp_target_checkpoint(
    const ModelContract&,const SerialSessionStateLayout&,SessionSnapshotPolicy);

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
        std::uint64_t retained_byte_budget,AllocationGenerationSource&);
    ~HipMtpTargetCheckpoint();
    HipMtpTargetCheckpoint(const HipMtpTargetCheckpoint&)=delete;
    HipMtpTargetCheckpoint& operator=(const HipMtpTargetCheckpoint&)=delete;
    void capture(const SerialSessionContext&);
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

