#pragma once
#include "strix/hip_mtp_target_hidden.hpp"
#include "strix/hip_mtp_target_window.hpp"
#include "strix/hip_mtp_target_checkpoint.hpp"
#include "strix/hip_known_prefix_window.hpp"
namespace strix {
struct HipMtpWindowVerificationPolicy {
    HipMtpTargetWindowCapturePolicy capture;
    SessionSnapshotPolicy snapshot_policy=SessionSnapshotPolicy::occupied_kv_v1;
    std::uint64_t checkpoint_retained_byte_budget=0,total_retained_byte_budget=0;
    bool bounded_window_rollback=false; // explicit candidate, full fallback remains default
};
struct HipMtpVerifiedTargetRows {
    std::vector<HipMtpTargetHiddenTicket> committed_rows;
    std::vector<std::uint32_t> emitted_tokens;
    std::uint32_t accepted_candidates=0;
    bool attempted_window=false,accepted_window=false,restored_checkpoint=false;
    std::uint32_t r2_dispatches=0,projection_launches_saved=0;
};
// One actual closed known-row window per cycle, with two retained capture pools.
// Every speculative input except the pending seed must match the preceding
// finite target prediction. A rejected/EOS-truncated window is restored before
// serial greedy replay; no provisional hidden row reaches MTP reconciliation.
class HipMtpWindowVerifier final {
public:
    HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden>,std::shared_ptr<HipKnownPrefixWindow>,
        const ExecutionRuntimeResources&,std::shared_ptr<const void> actual_target_resource_owner,
        HipMtpWindowVerificationPolicy,AllocationGenerationSource&);
    ~HipMtpWindowVerifier();
    HipMtpWindowVerifier(const HipMtpWindowVerifier&)=delete;
    HipMtpWindowVerifier& operator=(const HipMtpWindowVerifier&)=delete;
    [[nodiscard]] std::shared_ptr<HipMtpTargetHidden> target_owner() const noexcept;
    [[nodiscard]] std::shared_ptr<HipKnownPrefixWindow> window_owner() const noexcept;
    [[nodiscard]] std::uint32_t logical_rows() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] std::uint64_t retained_payload_bytes() const noexcept;
    [[nodiscard]] HipMtpVerifiedTargetRows verify(SerialSessionContext&,DirectLaunchDispatcher&,
        const HipMtpTargetHiddenTicket& initial,std::span<const std::uint32_t> candidates,
        std::uint32_t output_budget,std::span<const std::uint32_t> eos_tokens);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
