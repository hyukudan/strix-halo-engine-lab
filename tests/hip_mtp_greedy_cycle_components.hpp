#pragma once
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include <array>
namespace strix::tests {
struct HipMtpGreedyCycleGoldenRow {
    std::uint64_t initial_source_position=0,final_source_position=0,final_committed_tokens=0;
    std::uint64_t consumed_target_rows=0,reconciled_next_source_position=0;
    std::uint32_t initial_pending_token=0,final_pending_token=0;
    std::uint32_t maximum_candidates=0,token_budget=0,proposed_candidates=0,accepted_candidates=0;
    std::uint32_t target_r2_dispatches=0,target_projection_launches_saved=0;
    std::vector<std::uint32_t> eos_tokens,emitted_tokens;
    bool eos=false,terminal_budget=false,attempted_window=false,accepted_window=false,restored_checkpoint=false;
    std::vector<std::uint16_t> final_target_wide_bf16,reconciled_mtp_wide_bf16;
    std::vector<float> reconciled_full_logits;
};
struct HipMtpGreedyCycleGolden {
    ModelContentIdentity independent_source_content,target_content,sidecar_content;
    std::uint64_t independent_reference_identity=0,structural_plan_identity=0,semantic_identity=0,numerical_profile_identity=0;
    std::vector<HipMtpGreedyCycleGoldenRow> cycles;
};
struct HipMtpGreedyCycleCoverage {
    bool accepted_window=false,rejected_window_restored=false,eos=false,budget=false,continuation=false;
};
struct HipMtpGreedyCycleComponentRequest {
    ModelContentIdentity expected_payload;
    std::uint64_t maximum_cycles=0,host_payload_budget_bytes=0;
    HipMtpGreedyCycleCoverage required_coverage;
};
struct HipMtpGreedyCycleComponentRowReceipt {
    std::uint64_t final_source_position=0,consumed_target_rows=0;
    ModelContentIdentity emitted_tokens,target_wide,mtp_wide,mtp_logits;
};
struct HipMtpGreedyCycleComponentReceipt {
    ModelContentIdentity reference_payload;
    HipMtpGreedyCycleCoverage coverage;
    std::vector<HipMtpGreedyCycleComponentRowReceipt> cycles;
    HipMtpTargetHiddenTicket final_target;
};
// Binds every supplied full payload; does not attest reference authenticity.
[[nodiscard]] ModelContentIdentity hip_mtp_greedy_cycle_golden_identity(const HipMtpGreedyCycleGolden&);
// Caller supplies the actual qualified native consumer and its same native
// predictor, already primed to initial_target. Goldens must come from an
// independent, externally authenticated source. No qualification is minted.
// initial_target's pending token is ALREADY emitted; subsequent cycles receive
// exactly prior.final_target and compare only newly emitted outputs. Transfer
// the initial ticket with std::move and release other obsolete external batch
// pins; the fixture retains only its single current ticket, not old batches.
[[nodiscard]] HipMtpGreedyCycleComponentReceipt qualify_hip_mtp_greedy_cycles(
    HipMtpGreedyRuntime&,HipMtpRuntime&,HipMtpTargetHiddenTicket initial_target,
    const HipMtpGreedyCycleGolden&,const HipMtpGreedyCycleComponentRequest&);
// Regression for the real exhausted-history route: construct/attach an actual
// verifier whose declared capture bound is already exhausted, with otherwise
// window-eligible independent first-cycle trace. Must be called before the
// consumer's first cycle. Capacity is NOT raised and no history is retagged.
[[nodiscard]] HipMtpGreedyCycleComponentReceipt qualify_hip_mtp_history_capacity_fallback(
    HipMtpGreedyRuntime&,HipMtpRuntime&,std::shared_ptr<HipMtpTargetHidden>,
    std::shared_ptr<HipKnownPrefixWindow>,const ExecutionRuntimeResources&,
    std::shared_ptr<const void> actual_target_owner,HipMtpWindowVerificationPolicy,
    AllocationGenerationSource&,HipMtpTargetHiddenTicket initial_target,
    const HipMtpGreedyCycleGolden&,const HipMtpGreedyCycleComponentRequest&);
} // namespace strix::tests
