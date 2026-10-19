#pragma once
#include "strix/hip_mtp_runtime.hpp"
#include "strix/mtp_ngram_draft.hpp"
namespace strix {
class HipMtpWindowVerifier;
class HipKnownPrefixWindow;
struct HipMtpVerifiedTargetRows;
struct HipMtpGreedyCycleResult {
    std::vector<std::uint32_t> emitted_tokens;
    std::uint32_t proposed_candidates=0,accepted_candidates=0;
    std::uint64_t consumed_target_rows=0;
    bool eos=false,terminal_budget=false;
    HipMtpTargetHiddenTicket final_target;
    bool attempted_target_window=false,accepted_target_window=false,target_checkpoint_restored=false;
    std::uint32_t target_r2_dispatches=0,target_projection_launches_saved=0;
    std::uint32_t target_window_rows=0; // verifier choice: owner rows, 1 = serial; 0 without verifier
};
// Explicit qualified native MTP consumer, with serial or prepared closed-window target verification.
// Native execution/performance qualification is separate from source admission.
// Each returned token comes from a completed target head. Accepted candidates
// only decide how far to continue; they never replace actual target selection.
// The caller retains exclusive target session/dispatcher/resource ownership.
class HipMtpGreedyRuntime final {
public:
    HipMtpGreedyRuntime(std::shared_ptr<HipMtpRuntime>,
        QualifiedMtpDraftPlan,ActivatedMtpDraftPlan,
        std::shared_ptr<HipMtpTargetHidden>,SerialSessionContext&,
        const ExecutionRuntimeResources&,DirectLaunchDispatcher&,
        std::shared_ptr<const void> actual_target_resource_owner);
    ~HipMtpGreedyRuntime();
    HipMtpGreedyRuntime(const HipMtpGreedyRuntime&)=delete;
    HipMtpGreedyRuntime& operator=(const HipMtpGreedyRuntime&)=delete;
    // Explicit prepared closed-window target verification. Must be configured
    // before the first cycle; absent configuration preserves serial verification.
    void use_window_verifier(std::shared_ptr<HipMtpWindowVerifier>);
    // Draft chains stop after the first draft below this max softmax probability
    // (0 = full depth); the verifier receives every draft probability.
    void set_draft_min_probability(float);
    // Context n-gram extension of the draft chain (strix/mtp_ngram_draft.hpp): `prompt` is the request's
    // prompt. Needs the window verifier and greedy selection; set before the first cycle. Cohort cycles
    // (begin_cycle_draft without the cost model) do not use it.
    void enable_ngram_draft(std::span<const std::uint32_t> prompt,MtpNgramDraftConfig config={});
    [[nodiscard]] const MtpNgramDraftStats* ngram_stats() const noexcept;
    // Prime source positions in order with actual next prompt tokens. The last
    // prompt row is reserved for begin/cycle with its actual target prediction.
    void prime_prompt(std::span<const HipMtpReplayRow>);
    // Fresh serial prompt ingestion retains only the adjacent target tickets;
    // each known next prompt token primes the preceding predictor row. Returns
    // the final prompt row and its actual first generated pending selection.
    [[nodiscard]] HipMtpTargetHiddenTicket prefill_prompt(std::span<const std::uint32_t>);
    // initial_target.predicted_token is an already emitted, unconsumed pending
    // token. Consume it and return only subsequent newly verified outputs.
    // EOS pending consumes nothing; terminal EOS/budget leaves the last output
    // pending. Continue with final_target without emitting that token twice.
    [[nodiscard]] HipMtpGreedyCycleResult cycle(const HipMtpTargetHiddenTicket& initial_target,
        std::uint32_t maximum_candidates,std::uint32_t token_budget,
        std::span<const std::uint32_t> eos_tokens={});
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] const HipMtpRuntime& native_runtime() const noexcept;
    // cycle() in two phases, for a caller that verifies the drafts of several sessions at once
    // (hip_mtp_greedy_cohort_cycle). begin_cycle_draft drafts for initial_target exactly as cycle
    // does (cost_model = false drafts up to the minimum probability only) and returns the draft
    // depth limit; drafting() is false afterwards only when the pending token is EOS (nothing to
    // verify). finish_cycle publishes the target-verified rows as cycle does. An exception between
    // the phases must be followed by fail_cycle().
    std::uint32_t begin_cycle_draft(const HipMtpTargetHiddenTicket& initial_target,
        std::uint32_t maximum_candidates,std::uint32_t token_budget,
        std::span<const std::uint32_t> eos_tokens,bool cost_model);
    [[nodiscard]] bool drafting() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> drafted_candidates() const noexcept;
    [[nodiscard]] std::span<const float> drafted_probabilities() const noexcept;
    // Sampled decoding: the distribution of every drafted candidate (empty when the target selects first-max).
    [[nodiscard]] std::span<const TokenDraft> drafted_distributions() const noexcept;
    [[nodiscard]] HipMtpGreedyCycleResult finish_cycle(const HipMtpTargetHiddenTicket& initial_target,
        HipMtpVerifiedTargetRows verified,std::uint32_t token_budget,std::span<const std::uint32_t> eos_tokens);
    void fail_cycle() noexcept;
    [[nodiscard]] HipMtpWindowVerifier* window_verifier() const noexcept;
    [[nodiscard]] SerialSessionContext& target_session() const noexcept;
    [[nodiscard]] DirectLaunchDispatcher& target_dispatcher() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// One generation cycle of several independent sessions, each with its own window-verified greedy
// runtime. Every session drafts as in cycle() (up to the minimum probability; no cost-model stop)
// and gets the prefix owner verify() would size for its drafts; sessions are packed into cohorts
// of 4 or 8 rows (a smaller window may be rounded up with padding rows, which are never accepted)
// verified by one cohort window each. A session whose owner does not fit, or that is left alone,
// verifies with its own verify(). Each result is the session's own cycle result: every emitted
// token is that session's target greedy token. An EOS pending token emits nothing (result eos).
struct HipMtpGreedyCohortMember {
    HipMtpGreedyRuntime* runtime=nullptr;
    const HipMtpTargetHiddenTicket* initial=nullptr;
    std::uint32_t maximum_candidates=0,budget=0;
    std::span<const std::uint32_t> eos;
};
struct HipMtpGreedyCohortStats {std::uint64_t cohort_windows=0,cohort_sessions=0,cohort_rows=0,solo_windows=0,cohort_chunks=0;};
[[nodiscard]] std::vector<HipMtpGreedyCycleResult> hip_mtp_greedy_cohort_cycle(
    std::span<const HipMtpGreedyCohortMember> members,HipMtpGreedyCohortStats* stats=nullptr);
} // namespace strix
