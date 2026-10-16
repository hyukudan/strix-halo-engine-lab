#pragma once

#include "strix/execution_telemetry.hpp"
#include <cstddef>
#include <memory>

namespace strix {
class HipSetupQuiescenceGuard;
// Explicit profiling owner. Timing events perturb the schedule. This observer
// measures launcher queue spans (including gaps/waits), not kernel busy time.
// The serialized caller retains actual streams and allocations through export
// and keeps this observer alive through dispatch. No capture is instrumented.
// Setup requires no concurrent capture; ordinary captured execution is outside
// this eager owner contract. The forwarded host observer must also outlive it.
class HipEventQueueTrace final : public ExecutionTelemetry {
public:
    HipEventQueueTrace(ExecutionTelemetry& host, std::shared_ptr<const void> actual_owner,
        std::shared_ptr<const HipSetupQuiescenceGuard>, std::size_t capacity=2048);
    ~HipEventQueueTrace();
    HipEventQueueTrace(const HipEventQueueTrace&)=delete;
    HipEventQueueTrace& operator=(const HipEventQueueTrace&)=delete;
    void observe(const HostDispatchObservation&) noexcept override;
    void observe_activity(const ExecutionActivity&) noexcept override;
    [[nodiscard]] std::uint64_t begin_queue_span(const ExecutionDispatchContext&) noexcept override;
    void end_queue_span(std::uint64_t,DispatchOutcome) noexcept override;
    void queue_dependency(QueueDependencyKind,std::uintptr_t stream,std::uintptr_t event) noexcept override;
    // Existing transaction completion calls this before publishing any commit.
    // Stored tracing/API failure is not cleared or hidden as observational noise.
    void require_healthy_before_commit() const;
    // Call after existing completion. Query/elapsed only: no synchronization,
    // no assumption that host return or recorded events prove completion.
    // Invalid/incomplete intervals are exported with unknown times.
    void write_after_completion(std::ostream&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
