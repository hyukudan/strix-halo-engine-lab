#pragma once
#include "strix/session_snapshot_transfer.hpp"
#include <vector>
namespace strix {
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

} // namespace strix
