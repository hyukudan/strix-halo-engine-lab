#pragma once

#include "strix/execution_plan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>

namespace strix {
struct ExecutionDispatchContext;
enum class QueueDependencyKind { event_record, event_wait };

// Host launcher boundaries only. A returned launcher is not proof of a GPU
// submit, GPU completion, visibility or token commit. One serialized owner
// retains the observer until all host dispatch calls return.
enum class DispatchOutcome { returned, rejected, failed };
enum class ExecutionPhase : std::uint32_t { setup, admission, prepare, prefill, token, operator_step, selection, commit, delivery, count };
struct ExecutionCorrelation {
    std::uint64_t request_id = 0, session_epoch = 0, execution_identity = 0, position = 0, graph_replay = 0;
    // UINT32_MAX means a joint multi-row operation or an unknown individual row.
    std::uint32_t row = UINT32_MAX, rows = 0, step_id = UINT32_MAX;
    SerialOp op = SerialOp::position_prepare;
    ExecutionPhase phase = ExecutionPhase::setup;
    std::uint32_t layer = UINT32_MAX;
};
enum class ExecutionActivityKind : std::uint32_t {
    observation_begin, phase_elapsed, token_commit, proposed_state_write,
    copy_sync, copy_async, memset_sync, memset_async,
    device_alloc, device_free, host_alloc, host_free, host_register, host_unregister,
    device_sync, stream_sync, event_sync, stream_wait_event, event_record,
    stream_create, stream_destroy, event_create, event_destroy,
    graph_capture_begin, graph_capture_end, graph_instantiate, graph_replay, graph_destroy, graph_exec_destroy, host_launcher,
    count
};
enum class CopyDirection : std::uint32_t { unspecified, host_to_host, host_to_device, device_to_host, device_to_device, automatic, count };
struct ExecutionActivity {
    ExecutionActivityKind kind = ExecutionActivityKind::observation_begin;
    ExecutionCorrelation correlation;
    std::uint64_t requested_bytes = 0, units = 1, host_elapsed_ns = 0;
    std::uintptr_t stream = 0;
    CopyDirection direction = CopyDirection::unspecified;
    std::int32_t result_code = 0;
    bool success = true, during_capture = false, byte_count_saturated = false;
};
struct ExecutionActivityTotals {
    std::uint64_t calls = 0, succeeded = 0, failed = 0, requested_bytes = 0, accepted_bytes = 0;
    std::uint64_t units = 0, host_elapsed_ns = 0;
};
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
    ExecutionCorrelation correlation;
};

class ExecutionTelemetry {
public:
    virtual ~ExecutionTelemetry() = default;
    virtual void observe(const HostDispatchObservation&) noexcept = 0;
    virtual void observe_activity(const ExecutionActivity&) noexcept {}
    // Optional device observer. Tickets delimit launcher queue spans, never
    // individual kernels. Callbacks never throw; stored API failures belong to
    // the enclosing owner's precommit completion/error boundary.
    [[nodiscard]] virtual std::uint64_t begin_queue_span(const ExecutionDispatchContext&) noexcept { return 0; }
    virtual void end_queue_span(std::uint64_t, DispatchOutcome) noexcept {}
    virtual void queue_dependency(QueueDependencyKind, std::uintptr_t, std::uintptr_t) noexcept {}
};

// One serialized host owner per thread. Null observers do not read a clock,
// allocate, lock or invoke a HIP API. External-library/driver calls are outside
// this source instrumentation, and no GPU kernel count is inferred.
[[nodiscard]] ExecutionTelemetry* current_execution_telemetry() noexcept;
void observe_host_dispatch(ExecutionTelemetry&, const HostDispatchObservation&) noexcept;
[[nodiscard]] ExecutionCorrelation current_execution_correlation() noexcept;
void set_execution_correlation(ExecutionCorrelation) noexcept;
[[nodiscard]] bool execution_telemetry_during_capture() noexcept;
void execution_telemetry_capture_change(bool begin) noexcept;
void execution_telemetry_replay_attempted() noexcept;
void observe_execution_activity(ExecutionActivityKind, std::uint64_t bytes = 0,
    std::uint64_t units = 1, bool success = true, std::uint64_t elapsed_ns = 0,
    CopyDirection = CopyDirection::unspecified, std::uintptr_t stream = 0,
    std::int32_t result = 0, bool byte_count_saturated = false) noexcept;
class ExecutionTelemetryBinding final {
public:
    explicit ExecutionTelemetryBinding(ExecutionTelemetry*, ExecutionCorrelation = {}) noexcept;
    ~ExecutionTelemetryBinding();
    ExecutionTelemetryBinding(const ExecutionTelemetryBinding&) = delete;
    ExecutionTelemetryBinding& operator=(const ExecutionTelemetryBinding&) = delete;
private:
    ExecutionTelemetry* previous_ = nullptr;
    ExecutionCorrelation previous_correlation_;
    std::uint32_t previous_capture_depth_ = 0;
};
class ExecutionCorrelationScope final {
public:
    explicit ExecutionCorrelationScope(ExecutionCorrelation) noexcept;
    ~ExecutionCorrelationScope();
    ExecutionCorrelationScope(const ExecutionCorrelationScope&) = delete;
    ExecutionCorrelationScope& operator=(const ExecutionCorrelationScope&) = delete;
private:
    ExecutionCorrelation previous_;
    bool enabled_ = false;
};
class ExecutionPhaseScope final {
public:
    explicit ExecutionPhaseScope(ExecutionPhase) noexcept;
    ~ExecutionPhaseScope();
    ExecutionPhaseScope(const ExecutionPhaseScope&) = delete;
    ExecutionPhaseScope& operator=(const ExecutionPhaseScope&) = delete;
private:
    ExecutionCorrelation previous_;
    std::uint64_t start_ns_ = 0;
    int exceptions_ = 0;
    bool enabled_ = false;
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
    void observe_activity(const ExecutionActivity&) noexcept override;
    void enable_trace(bool enabled) noexcept { trace_enabled_ = enabled; }
    [[nodiscard]] bool activity_observed() const noexcept { return activity_observed_; }
    [[nodiscard]] const auto& activities() const noexcept { return activities_; }
    [[nodiscard]] const auto& captured_activities() const noexcept { return captured_activities_; }
    [[nodiscard]] const auto& copy_directions() const noexcept { return copy_directions_; }
    [[nodiscard]] const auto& operator_activities() const noexcept { return operator_activities_; }
    [[nodiscard]] const auto& phases() const noexcept { return phases_; }
    [[nodiscard]] std::span<const ExecutionActivity> trace() const noexcept { return {trace_.data(), trace_size_}; }
    [[nodiscard]] std::uint64_t trace_dropped() const noexcept { return trace_dropped_; }
private:
    HostDispatchTotals totals_;
    std::array<HostDispatchTotals, static_cast<std::size_t>(SerialOp::lm_head) + 1> operators_{};
    bool saturated_ = false;
    bool activity_observed_ = false, trace_enabled_ = false;
    std::array<ExecutionActivityTotals, static_cast<std::size_t>(ExecutionActivityKind::count)> activities_{};
    std::array<ExecutionActivityTotals, static_cast<std::size_t>(ExecutionActivityKind::count)> captured_activities_{};
    std::array<ExecutionActivityTotals, static_cast<std::size_t>(CopyDirection::count)> copy_directions_{};
    std::array<ExecutionActivityTotals, static_cast<std::size_t>(ExecutionPhase::count)> phases_{};
    std::array<ExecutionActivityTotals, static_cast<std::size_t>(SerialOp::lm_head) + 1> operator_activities_{};
    std::array<ExecutionActivity, 256> trace_{};
    std::size_t trace_size_ = 0;
    std::uint64_t trace_dropped_ = 0;
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
