#pragma once
#include "strix/hip_mtp_draft_head.hpp"
#include "strix/hip_mtp_resources.hpp"
#include "strix/hip_mtp_target_hidden.hpp"
#include "strix/qualified_mtp_forward_runtime.hpp"
#include "strix/token_sampling.hpp"
#include <functional>
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
    // Tiled-prefill prompt rows: the same teacher cache rows built from committed wide hidden rows a
    // tiled prefill block produced (device copies owned by the caller, contiguous source positions
    // from next_source_position()); next_tokens[i] is the target input after hidden row i. No
    // ticket chain exists for these rows: the next cycle starts without a primed predecessor ticket.
    void prime_hidden_rows(const QualifiedMtpDraftPlan&,const ActivatedMtpDraftPlan&,
        std::span<const MtpDeviceView> hidden,std::span<const std::uint32_t> next_tokens);
    // Prefix state cache: the caller wrote durable cache rows [0, source) back into this fresh idle
    // runtime's KV/raw-index backing; republish `source` as its frontier.
    void restore_cache_frontier(const QualifiedMtpDraftPlan&,const ActivatedMtpDraftPlan&,std::uint64_t source);
    // All source pairs after the logical cache frontier, plus the final committed target
    // ticket. Equality of predicted tokens is never cache authority.
    void set_reconciliation_rows(std::span<const HipMtpReplayRow>,const HipMtpTargetHiddenTicket& final_target);
    // Opt-in sync-free draft chain (before the first cycle): every draft row is
    // fed its token from device memory, the draft head (vocabulary subset or
    // full) selects on device into mapped memory, and one completion ends the
    // chain. Teacher cache rows of priming/reconciliation also complete once.
    // Drafts only steer acceptance; target outputs are unchanged. Unless exact,
    // draft rows use the fused input projection and the 2-launch Q8 MoE.
    void enable_draft_chain(HipMtpDraftHeadConfig,bool exact=false);
    [[nodiscard]] const HipMtpDraftHead* draft_head() const noexcept;
    // Context ids for the draft vocabulary (prompt; cycles add their own tokens).
    void observe_draft_context(std::span<const std::uint32_t>);
    // min_probability > 0 completes each draft before the next and stops after
    // the first draft whose max softmax probability is below it (that draft is
    // kept). Returns 1..depth drafts; probabilities of the last chain below.
    // keep_drafting (optional; it also completes each draft before the next) sees the probabilities
    // so far (runner-ups in draft_runner_up_probabilities()) and may stop the chain.
    [[nodiscard]] std::vector<std::uint32_t> propose_chain(const QualifiedMtpDraftPlan&,const ActivatedMtpDraftPlan&,
        const SerialSessionContext&,const DraftProposalStateContext&,std::uint32_t depth,float min_probability=0.0F,
        const std::function<bool(std::span<const float>)>& keep_drafting={});
    // The same chain in phases, so that a cohort can enqueue step k of every session (each on its own
    // stream) before waiting for any: begin (false: depth 0, nothing to draft), then while launch_step()
    // and after_step() (false: the chain stops), then finish() for the drafts. Each session's launches and
    // waits are exactly those of propose_chain.
    [[nodiscard]] bool propose_chain_begin(const QualifiedMtpDraftPlan&,const ActivatedMtpDraftPlan&,
        const SerialSessionContext&,const DraftProposalStateContext&,std::uint32_t depth,float min_probability=0.0F,
        const std::function<bool(std::span<const float>)>& keep_drafting={});
    [[nodiscard]] bool propose_chain_launch_step();
    [[nodiscard]] bool propose_chain_after_step();
    [[nodiscard]] std::vector<std::uint32_t> propose_chain_finish();
    [[nodiscard]] std::span<const float> draft_probabilities() const noexcept;
    // Sampled drafts (temperature > 0), set once on an enabled draft chain before cycles:
    // every draft step completes, its token is drawn from the draft distribution and
    // feeds the next step. sampled_drafts() holds the last chain's drafts in order.
    void set_draft_sampler(std::shared_ptr<TokenSampler>);
    [[nodiscard]] std::span<const TokenDraft> sampled_drafts() const noexcept;
    // Runner-up probability per draft of the last chain (zeros unless the head reduces runner-ups).
    [[nodiscard]] std::span<const float> draft_runner_up_probabilities() const noexcept;
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
