#pragma once

#include "strix/execution_plan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>

namespace strix {

// Host launcher boundaries only. A returned launcher is not proof of a GPU
// submit, GPU completion, visibility or token commit. One serialized owner
// retains the observer until all host dispatch calls return.
enum class DispatchOutcome { returned, rejected, failed };
struct HostDispatchObservation {
    std::uint64_t execution_identity = 0;
    std::uint64_t position = 0;
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::optional<std::uint32_t> layer;
    std::uint32_t logical_rows = 1;
    std::uint32_t request_concurrency = 1;
    bool runtime_step = false;
    DispatchOutcome outcome = DispatchOutcome::rejected;
    // steady_clock nanoseconds; includes launcher lookup/validation. Never
    // subtract this host-clock interval from a GPU-clock interval.
    std::uint64_t host_elapsed_ns = 0;
};

class ExecutionTelemetry {
public:
    virtual ~ExecutionTelemetry() = default;
    virtual void observe(const HostDispatchObservation&) noexcept = 0;
};

struct HostDispatchTotals {
    std::uint64_t returned = 0;
    std::uint64_t rejected = 0;
    std::uint64_t failed = 0;
    std::uint64_t host_elapsed_ns = 0;
};

// Fixed storage: no allocation or locks in the observer. Saturation is reported
// explicitly rather than wrapping or throwing into the execution path.
class HostDispatchCounters final : public ExecutionTelemetry {
public:
    void observe(const HostDispatchObservation&) noexcept override;
    [[nodiscard]] const HostDispatchTotals& totals() const noexcept { return totals_; }
    [[nodiscard]] const auto& operators() const noexcept { return operators_; }
    [[nodiscard]] bool saturated() const noexcept { return saturated_; }
private:
    HostDispatchTotals totals_;
    std::array<HostDispatchTotals, static_cast<std::size_t>(SerialOp::lm_head) + 1> operators_{};
    bool saturated_ = false;
};

void write_host_dispatch_telemetry(std::ostream&, const HostDispatchCounters&);

// Offline profiler records must share a clock domain (even across queues).
// Zero-length intervals are valid. Invalid ranges and sum overflow throw.
struct GpuTraceInterval {
    std::uint64_t begin_ns = 0;
    std::uint64_t end_ns = 0;
    std::uint64_t queue_id = 0;
    std::uint64_t clock_domain = 0;
};
struct GpuTraceTime {
    std::uint64_t summed_ns = 0;
    std::uint64_t busy_union_ns = 0;
};
[[nodiscard]] GpuTraceTime summarize_gpu_trace(std::span<const GpuTraceInterval>);

}  // namespace strix
