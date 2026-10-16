#pragma once
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/prompt_lookup_draft.hpp"
#include "strix/speculation_executor.hpp"
#include "strix/speculation_controller.hpp"
#include <array>
#include <memory>
#include <optional>

namespace strix {
class HipSetupQuiescenceGuard;
struct HipPromptLookupAdaptivePolicy {
    bool enabled = false;
    ObservedSpeculationPolicy controller{};
    std::uint64_t context_bucket_tokens = 256;
};
struct HipPromptLookupAdaptiveStats {
    bool enabled = false;
    std::uint32_t selected_depth = 0;
    std::uint64_t completed_cycles = 0, censored_cycles = 0, context_resets = 0;
    std::array<std::uint64_t, 3> depth_selections{}, samples{}, emitted_tokens{}, committed_tokens{};
    std::array<double, 3> complete_cycle_seconds{};
};
// Explicit experimental greedy target. No sampling/MTP. D=2 uses the actual
// known-input R2 owner; D=1 is identified serial
// fallback; D=0 selects the already-computed target frontier without consuming
// state. Both complete R2 heads are selected on device with finite/firstmax.
//
// serial_resources and window must describe the SAME immutable target owners.
// actual_owner retains all allocations, descriptors, PLE, stream, allocator and
// generation source through destruction. Exclusive serialized access only;
// graph/prepared/overlap/island wrappers are not accepted by this backend.
// Both scopes borrow the SAME owned stream, with separate serial/window D1
// records bound to their exact execution identities; no record is cleared here.
class HipPromptLookupRuntime final : public GreedySpeculationExecutionBackend {
public:
    HipPromptLookupRuntime(CompiledSerialExecution serial, ModelContract model,
        HipLauncherCatalogConfig config, const ExecutionRuntimeResources& serial_resources,
        CompiledKnownPrefixWindow compiled_window, std::shared_ptr<HipKnownPrefixWindow> window,
        ExecutionMemoryBinding window_scratch, HipLaunchContext serial_scope, HipLaunchContext window_scope,
        std::shared_ptr<const void> actual_owner, std::shared_ptr<const HipSetupQuiescenceGuard> guard,
        RuntimeMemoryAllocator& snapshot_allocator, AllocationGenerationSource& generations,
        HipPromptLookupAdaptivePolicy adaptive = {});
    ~HipPromptLookupRuntime();
    HipPromptLookupRuntime(const HipPromptLookupRuntime&) = delete;
    HipPromptLookupRuntime& operator=(const HipPromptLookupRuntime&) = delete;
    // Caller has completed the last SERIAL prompt input, with its full head
    // still intact. This does not consume a token or change producer history.
    void prime_from_serial_logits(SerialSessionContext& session);
    [[nodiscard]] GreedyDraftExecutionResult cycle(SerialSessionContext& session,
        PromptLookupDraftBackend& lookup, DraftProposalStateContext& draft_state,
        const GreedyDraftExecutionRequest& request);
    // Call ONLY when continuing after a returned cycle, never at EOS/budget or
    // cancellation. The last emitted token is still unconsumed. Its existing
    // history entry is not appended again; lifecycle metadata is rebound to the
    // now-consumed target state before the next proposal.
    void consume_pending_bonus(SerialSessionContext& session,
        const PromptLookupDraftBackend& lookup, DraftProposalStateContext& draft_state);
    // Opt-in wall scope: cycle entry through this call, AFTER bonus consumption
    // and caller publication/bookkeeping. An unconsumed terminal bonus censors
    // the observation. No GPU-time/correctness qualification is inferred.
    // Finish once before another adaptive cycle. Disabled mode reads no clock.
    void finish_adaptive_cycle(const SerialSessionContext& session) noexcept;
    [[nodiscard]] const HipPromptLookupAdaptiveStats& adaptive_stats() const noexcept;
    [[nodiscard]] std::optional<std::uint32_t> pending_bonus() const noexcept;
    [[nodiscard]] std::vector<std::uint32_t> verify_target_candidates(SerialSessionContext&,
        const ExecutionRuntimeResources&, std::span<const std::uint32_t>) override;
    void replay_target_token(SerialSessionContext&, const ExecutionRuntimeResources&, std::uint32_t) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
