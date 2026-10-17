#pragma once
#include "strix/hip_mtp_draft_head.hpp"
#include "strix/hip_mtp_resources.hpp"
#include "strix/hip_mtp_target_hidden.hpp"
#include "strix/qualified_mtp_forward_runtime.hpp"
#include <span>
#include <vector>

namespace strix {
struct HipMtpReplayRow {
    HipMtpTargetHiddenTicket hidden;
    std::uint32_t next_input_token=0;
};
struct HipMtpNativeRowResult {
    std::uint64_t source_position=0,seed_position=0,numerical_profile_identity=0;
    std::uint32_t token=0,completed_steps=0;
    MtpDeviceView wide_state,logits;
    bool device_completed=false;
};
// Only these durable occupied prefixes are meaningful after teacher replay.
// KV rows are [K[kv_heads*head_dim],V[kv_heads*head_dim]] BF16; raw-index
// rows are BF16[indexer_head_dim]. Unoccupied future bytes and work/logit/wide
// outputs after a cache-only row are unspecified, never restoration evidence.
struct HipMtpCacheStateView {
    MtpDeviceView kv,raw_index;
    std::uint64_t logical_rows=0,kv_row_bytes=0,raw_index_row_bytes=0;
};
struct HipMtpExecutionStats {
    // Completed rows only. Full rows include the real finite token reader;
    // cache rows have no head, selector or wide-next publication.
    std::uint64_t full_forward_rows=0,cache_only_rows=0;
    // Actual first proposal cache row retained under its private completion
    // proof; not a new GPU cachewriter submission and never token equality.
    std::uint64_t cache_reused_rows=0;
};
// Concrete eager C1 native proposal owner/protocol. The diagnostic entry is an
// explicit candidate execution to acquire independent replay evidence; it does
// not mint activation/qualification or claim checkpoint numerical equivalence.
// Qualified adapter calls remain subject to their existing qualification gates.
// Output views are reused on the next input. One serialized caller retains all
// real target tickets and resources until native work has completed.
class HipMtpRuntime final : public MtpForwardExecutionBackend,
    public MtpRuntimeCycleInputProvider,public MtpRuntimeTokenReader {
public:
    explicit HipMtpRuntime(std::shared_ptr<HipMtpResources>);
    ~HipMtpRuntime();
    HipMtpRuntime(const HipMtpRuntime&)=delete;
    HipMtpRuntime& operator=(const HipMtpRuntime&)=delete;
    [[nodiscard]] const HipMtpResources& resources() const noexcept;
    [[nodiscard]] std::uint64_t next_source_position() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    void cancel() noexcept;
    [[nodiscard]] std::uint64_t numerical_profile_identity() const noexcept;
    // Requires healthy idle ownership outside a live proposal. Views borrow
    // this actual owner and are valid until the next runtime operation.
    [[nodiscard]] HipMtpCacheStateView cache_state() const;
    [[nodiscard]] HipMtpExecutionStats execution_stats() const noexcept;
    [[nodiscard]] HipMtpNativeRowResult execute_diagnostic(
        const HipMtpTargetHiddenTicket&,std::uint32_t next_input_token);
    void begin_cycle(std::uint32_t actual_pending_token,const HipMtpTargetHiddenTicket&);
    // Qualified teacher rows rebuild only exact raw-index/K/V cache state;
    // full diagnostic replay above always executes all eleven stages.
    void prime_committed_rows(const QualifiedMtpDraftPlan&,const ActivatedMtpDraftPlan&,
        std::span<const HipMtpReplayRow>);
    // All source pairs after the logical cache frontier, plus the final committed target
    // ticket. Equality of predicted tokens is never cache authority.
    void set_reconciliation_rows(std::span<const HipMtpReplayRow>,const HipMtpTargetHiddenTicket& final_target);
    // Opt-in sync-free draft chain (before the first cycle): every draft row is
    // fed its token from device memory, the draft head (vocabulary subset or
    // full) selects on device into mapped memory, and one completion ends the
    // chain. Teacher cache rows of priming/reconciliation also complete once.
    // Drafts only steer acceptance; target outputs are unchanged.
    void enable_draft_chain(HipMtpDraftHeadConfig);
    [[nodiscard]] const HipMtpDraftHead* draft_head() const noexcept;
    // Context ids for the draft vocabulary (prompt; cycles add their own tokens).
    void observe_draft_context(std::span<const std::uint32_t>);
    [[nodiscard]] std::vector<std::uint32_t> propose_chain(const QualifiedMtpDraftPlan&,const ActivatedMtpDraftPlan&,
        const SerialSessionContext&,const DraftProposalStateContext&,std::uint32_t depth);
    void dispatch_mtp_step(const MtpForwardStepContext&) override;
    [[nodiscard]] MtpDraftCycleInput make_cycle_input(const QualifiedMtpDraftPlan&,const ActivatedMtpDraftPlan&,
        const MtpDraftRuntimeBinding&,const MtpDraftStateMemoryBinding&,const SerialSessionContext&,
        const DraftProposalStateContext&,std::span<const std::uint32_t> prior_candidates) override;
    void reconcile_after_target(const QualifiedMtpDraftPlan&,const ActivatedMtpDraftPlan&,
        const MtpDraftRuntimeBinding&,const MtpDraftStateMemoryBinding&,const DraftProposalStateContext&,
        const SerialSessionContext&,std::span<const std::uint32_t> emitted_tokens) override;
    [[nodiscard]] std::uint32_t read_greedy_token(const QualifiedMtpDraftPlan&,const MtpDraftCycleInput&,
        const MtpForwardExecutionResult&,std::uint32_t candidate_index) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
using HipMtpForwardBackend=HipMtpRuntime;
} // namespace strix
