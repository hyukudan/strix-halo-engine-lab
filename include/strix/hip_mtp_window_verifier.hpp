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
    // Adaptive owner choice (prefix owners only): 0 keeps the largest fitting owner.
    float min_probability=0.0F;
};
struct HipMtpVerifiedTargetRows {
    std::vector<HipMtpTargetHiddenTicket> committed_rows;
    std::vector<std::uint32_t> emitted_tokens;
    std::uint32_t accepted_candidates=0;
    bool attempted_window=false,accepted_window=false,restored_checkpoint=false;
    std::uint32_t r2_dispatches=0,projection_launches_saved=0;
    std::uint32_t chosen_rows=1; // window owner rows, 1 = serial target steps
};
struct HipMtpWindowOwnerStats {std::uint32_t rows=0;std::uint64_t cycles=0;double cycle_ms_ema=0.0;};
// One actual closed known-row window per cycle, with retained capture pools per owner.
// Several prefix owners (distinct 2/4/8 rows over one session) share one deferred
// GDN record; per cycle T = 1 + leading drafts with probability >= min_probability,
// rounded up to an owner, then the owner (or serial at T = 1) with the best
// expected tokens per ms under an EMA of measured cycle time.
// Every speculative input except the pending seed must match the preceding
// finite target prediction. A rejected/EOS-truncated window is restored before
// serial greedy replay; no provisional hidden row reaches MTP reconciliation.
class HipMtpWindowVerifier final {
public:
    HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden>,std::shared_ptr<HipKnownPrefixWindow>,
        const ExecutionRuntimeResources&,std::shared_ptr<const void> actual_target_resource_owner,
        HipMtpWindowVerificationPolicy,AllocationGenerationSource&);
    HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden>,std::vector<std::shared_ptr<HipKnownPrefixWindow>>,
        const ExecutionRuntimeResources&,std::shared_ptr<const void> actual_target_resource_owner,
        HipMtpWindowVerificationPolicy,AllocationGenerationSource&);
    ~HipMtpWindowVerifier();
    HipMtpWindowVerifier(const HipMtpWindowVerifier&)=delete;
    HipMtpWindowVerifier& operator=(const HipMtpWindowVerifier&)=delete;
    [[nodiscard]] std::shared_ptr<HipMtpTargetHidden> target_owner() const noexcept;
    // Largest owner; logical_rows() is its row count.
    [[nodiscard]] std::shared_ptr<HipKnownPrefixWindow> window_owner() const noexcept;
    [[nodiscard]] std::vector<std::shared_ptr<HipKnownPrefixWindow>> window_owners() const;
    [[nodiscard]] std::vector<HipMtpWindowOwnerStats> owner_stats() const;
    [[nodiscard]] std::uint32_t logical_rows() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] std::uint64_t retained_payload_bytes() const noexcept;
    [[nodiscard]] HipMtpVerifiedTargetRows verify(SerialSessionContext&,DirectLaunchDispatcher&,
        const HipMtpTargetHiddenTicket& initial,std::span<const std::uint32_t> candidates,
        std::uint32_t output_budget,std::span<const std::uint32_t> eos_tokens,
        std::span<const float> candidate_probabilities={});
    // Whether one more draft can still pay off: the next larger window, with the remaining drafts assumed
    // as likely as the last one, beats the best window the current drafts already allow.
    [[nodiscard]] bool worth_drafting(std::span<const float> probabilities) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
