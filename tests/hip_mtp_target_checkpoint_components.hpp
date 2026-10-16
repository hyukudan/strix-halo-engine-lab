#pragma once
#include "strix/hip_mtp_window_verifier.hpp"
namespace strix::tests {
struct HipMtpTargetCheckpointComponentReceipt {
    bool full_device_bytes_restored=false,full_host_bytes_restored=false,deferred_status_restored=false;
    bool exact_session_metadata_restored=false;
    std::uint64_t device_bytes_checked=0,host_bytes_checked=0,deferred_bytes_checked=0,checkpoint_reserved_bytes=0;
    // Observed successful mutation execution, before byte-exact restoration.
    // Launch savings are the runtime's source-ledger count, not device time.
    std::uint32_t logical_rows=0,r2_dispatches=0,projection_launches_saved=0;
    std::uint64_t window_execution_identity=0,window_kernel_plan_identity=0,chunk_begin=0,chunk_end=0;
};
// Explicit callable model fixture, not automatic CTest or a numerical oracle.
// Starts at an ACTUAL ready target frontier with actual admitted known inputs.
[[nodiscard]] HipMtpTargetCheckpointComponentReceipt qualify_hip_mtp_target_window_checkpoint(
    std::shared_ptr<HipMtpTargetHidden>,std::shared_ptr<HipKnownPrefixWindow>,SerialSessionContext&,
    const ExecutionRuntimeResources&,std::shared_ptr<const void> actual_resource_owner,
    std::span<const std::uint32_t> inputs,HipMtpTargetWindowCapturePolicy,
    std::uint64_t checkpoint_retained_budget,std::uint64_t host_readback_budget,AllocationGenerationSource&);
}
