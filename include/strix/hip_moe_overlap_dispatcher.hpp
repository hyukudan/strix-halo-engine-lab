#pragma once
#include "strix/hip_launch_context.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/moe_fork_join_arena.hpp"
#include "strix/serial_session.hpp"
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace strix {
class HipSetupQuiescenceGuard;
// Explicit override, NOT a nested HipLaunchScope. Outer D1 scope must remain
// alive. The side branch has no access to D1's single-publisher error record.
class HipSharedExpertSideScope final {
public:
    HipSharedExpertSideScope(hipStream_t main, hipStream_t side) {
        previous_ = current_hip_launch_context();
        if (!previous_ || previous_->graph_capture || previous_->stream != main ||
            !main || main == hipStreamLegacy || main == hipStreamPerThread ||
            !side || side == main || side == hipStreamLegacy || side == hipStreamPerThread ||
            previous_->deferred_moe_status.arena_bytes != 8 ||
            !previous_->deferred_moe_status.execution_identity ||
            !previous_->deferred_moe_status.allocation_generation ||
            !previous_->deferred_moe_status.arena_base)
            throw std::invalid_argument("shared side scope requires original eager D1 context and distinct owned stream");
        copy_ = *previous_; copy_.stream = side; copy_.deferred_moe_status = {};
        active_hip_launch_context = &copy_;
    }
    ~HipSharedExpertSideScope() noexcept { active_hip_launch_context = previous_; }
    HipSharedExpertSideScope(const HipSharedExpertSideScope&) = delete;
    HipSharedExpertSideScope& operator=(const HipSharedExpertSideScope&) = delete;
private:
    const HipLaunchContext* previous_ = nullptr;
    HipLaunchContext copy_;
};
struct HipMoeOverlapStats {
    std::uint64_t fork_records = 0, join_waits = 0, verified_tokens = 0;
    std::uint64_t committed_tokens = 0, committed_layers = 0, aborted_tokens = 0;
};
enum class HipMoeWorkState { idle, prepared, submitted, gpu_completed, validated, committed, recyclable, failed_drained };
// Observation only, never an admission permit. Host callback return is not GPU
// completion. Epochs are owner-local leases, not a replacement for allocation
// generations. A failed/cancelled owner is permanently unusable after draining.
struct HipMoeWorkTicket {
    std::uint64_t owner_epoch = 0, token_epoch = 0, session_epoch = 0;
    std::uint64_t input_position = 0, committed_before = 0;
    std::uint64_t binding_identity = 0, scratch_generation = 0;
    std::uint64_t state_device_generation = 0, state_host_generation = 0, status_generation = 0;
    HipMoeWorkState state = HipMoeWorkState::idle;
    // validated covers branch completion/metadata. The enclosing D1 completion
    // hook must still validate its main-only logical record before real commit.
    bool gpu_completed = false, validated = false, committed = false;
    bool completion_queried = false;
    hipError_t main_completion = hipSuccess, side_completion = hipSuccess;
};
// Low-level persistent event owner, also exercised directly by the native event
// fixture. Main is borrowed and must outlive this owner. Resources retain actual
// arena/weights/state/launch-config leases. No per-layer stream/event allocation.
class HipMoeSiblingStreams final {
public:
    HipMoeSiblingStreams(hipStream_t main, std::shared_ptr<const void> resources_owner,
                        std::shared_ptr<const HipSetupQuiescenceGuard>);
    ~HipMoeSiblingStreams();
    HipMoeSiblingStreams(const HipMoeSiblingStreams&) = delete;
    HipMoeSiblingStreams& operator=(const HipMoeSiblingStreams&) = delete;
    void begin_token(std::uint64_t input_position, std::uint64_t committed_before, std::uint32_t expected_layers);
    void fork_shared(std::uint32_t layer);
    void record_shared_done(std::uint32_t layer);
    void begin_experts(std::uint32_t layer);
    void join_before_combine(std::uint32_t layer);
    // Must precede D1 complete_before_commit (which publishes shadow histories).
    // Drains main, whose joins cover side; independently queries side success.
    void verify_before_commit();
    void mark_committed(std::uint64_t next_position, std::uint64_t committed_tokens);
    void abort() noexcept; // Device-wide original-device drain or fail-stop.
    [[nodiscard]] hipStream_t main_stream() const noexcept;
    [[nodiscard]] hipStream_t side_stream() const noexcept;
    [[nodiscard]] HipMoeOverlapStats stats() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] HipMoeWorkTicket ticket() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Real dispatcher adapter for the immutable expanded S04a arena and a FRESH
// launch plan compiled against that new identity. It keeps original host step
// order route/shared/experts/combine; ONLY shared launches on side. All kernels,
// launcher math, catalog IDs and DirectLaunchDispatcher behavior stay original.
// compiled/resources/telemetry objects themselves must outlive this adapter;
// supplied pins must retain their actual allocation and launcher-config owners.
// The enclosing token transaction must call abort() on ANY exception, including
// admission/precondition failures, and poison the session after a failed commit.
// One serialized caller owns this adapter. The exact session object is borrowed
// for its lifetime and must not be moved/destroyed/recreated while retained;
// owners and metadata remain stable during each active token. Rebinding requires
// a new adapter after quiescence. Cancellation aborts; it does not roll back state.
class HipMoeOverlapDispatcher final : public EagerExecutionDispatcher {
public:
    HipMoeOverlapDispatcher(const CompiledSerialExecution& expanded,
        const MoeForkJoinArenaPlan&, const CompiledLaunchPlan& fresh_launches,
        const ExecutionRuntimeResources& d1_resources, hipStream_t main,
        std::shared_ptr<const void> resources_owner, std::shared_ptr<const void> launcher_owner,
        std::shared_ptr<const HipSetupQuiescenceGuard>, ExecutionTelemetry* = nullptr);
    ~HipMoeOverlapDispatcher();
    HipMoeOverlapDispatcher(const HipMoeOverlapDispatcher&) = delete;
    HipMoeOverlapDispatcher& operator=(const HipMoeOverlapDispatcher&) = delete;
    void begin_token(const SerialSessionContext&);
    void dispatch_kernel(const ExecutionDispatchContext&) override;
    void dispatch_runtime_step(const ExecutionDispatchContext&) override;
    // Explicit composition with state-only prefix output: only the single final
    // stateless kernel LM head may be absent; never counts as GPU submission.
    void verify_before_commit(bool terminal_head_omitted = false);
    void mark_committed(const SerialSessionContext&);
    void abort() noexcept;
    [[nodiscard]] HipMoeOverlapStats stats() const noexcept;
    [[nodiscard]] HipMoeWorkTicket ticket() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
