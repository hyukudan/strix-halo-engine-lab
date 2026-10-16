#pragma once
#include "strix/serial_request.hpp"
#include <optional>
#include <functional>
#include <span>
namespace strix {
// Optional observer wrapper forwarding its own request telemetry TLS while the
// cohort executes the underlying native member. Does not count a fabricated
// ordinary backend call or re-run request policy.
class ServingCohortObservation {
public:
    virtual ~ServingCohortObservation() = default;
    virtual void observe_serving_target(const std::function<void()>&) = 0;
};
struct ServingCohortTarget {
    SerialGreedyStepBackend* backend = nullptr;
    SerialGreedyTargetInput input;
    std::optional<std::uint32_t> selected;
};
// Execute exactly one already prepared cursor target per independent session.
// Completion includes producer/status/finite delivery before return. It does
// not accept the cursor ticket or emit a token; the scheduler does that once.
class ServingCohortExecutor {
public:
    virtual ~ServingCohortExecutor() = default;
    [[nodiscard]] virtual bool compatible(std::span<SerialGreedyStepBackend* const>) const noexcept = 0;
    virtual void execute(std::span<ServingCohortTarget>) = 0;
};
void validate_serving_cohort_targets(std::span<const ServingCohortTarget>);
// Honest C1 fallback, never reported as tensor batching.
void execute_serving_target_fallback(ServingCohortTarget&);
} // namespace strix
