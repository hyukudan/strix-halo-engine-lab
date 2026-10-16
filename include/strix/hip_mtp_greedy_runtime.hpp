#pragma once
#include "strix/hip_mtp_runtime.hpp"
namespace strix {
class HipMtpWindowVerifier;
struct HipMtpGreedyCycleResult {
    std::vector<std::uint32_t> emitted_tokens;
    std::uint32_t proposed_candidates=0,accepted_candidates=0;
    std::uint64_t consumed_target_rows=0;
    bool eos=false,terminal_budget=false;
    HipMtpTargetHiddenTicket final_target;
    bool attempted_target_window=false,accepted_target_window=false,target_checkpoint_restored=false;
    std::uint32_t target_r2_dispatches=0,target_projection_launches_saved=0;
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
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
