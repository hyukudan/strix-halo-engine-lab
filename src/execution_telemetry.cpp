#include "strix/execution_telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace strix {
namespace {
void add(std::uint64_t& value, std::uint64_t amount, bool& saturated) noexcept {
    if (amount > std::numeric_limits<std::uint64_t>::max() - value) {
        value = std::numeric_limits<std::uint64_t>::max();
        saturated = true;
    } else {
        value += amount;
    }
}
void count(HostDispatchTotals& total, const HostDispatchObservation& event, bool& saturated) noexcept {
    switch (event.outcome) {
    case DispatchOutcome::returned: add(total.returned, 1, saturated); break;
    case DispatchOutcome::rejected: add(total.rejected, 1, saturated); break;
    case DispatchOutcome::failed: add(total.failed, 1, saturated); break;
    }
    add(total.host_elapsed_ns, event.host_elapsed_ns, saturated);
}
void write_totals(std::ostream& out, const HostDispatchTotals& total) {
    out << "\"returned\":" << total.returned << ",\"rejected\":" << total.rejected
        << ",\"failed\":" << total.failed << ",\"host_elapsed_ns\":" << total.host_elapsed_ns;
}
}

void HostDispatchCounters::observe(const HostDispatchObservation& event) noexcept {
    count(totals_, event, saturated_);
    const auto index = static_cast<std::size_t>(event.op);
    if (index < operators_.size()) count(operators_[index], event, saturated_);
    ExecutionActivity boundary;
    boundary.kind = ExecutionActivityKind::host_launcher;
    boundary.correlation = event.correlation;
    boundary.host_elapsed_ns = event.host_elapsed_ns;
    boundary.success = event.outcome == DispatchOutcome::returned;
    boundary.result_code = static_cast<std::int32_t>(event.outcome);
    boundary.during_capture = execution_telemetry_during_capture();
    observe_activity(boundary);
}

namespace {
thread_local ExecutionTelemetry* active_observer = nullptr;
thread_local ExecutionCorrelation active_correlation;
thread_local std::uint32_t capture_depth = 0;
thread_local bool inside_observer = false;
std::uint64_t host_now() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
void count_activity(ExecutionActivityTotals& total, const ExecutionActivity& event, bool& saturated) noexcept {
    add(total.calls, 1, saturated);
    add(event.success ? total.succeeded : total.failed, 1, saturated);
    add(total.requested_bytes, event.requested_bytes, saturated);
    if (event.success) add(total.accepted_bytes, event.requested_bytes, saturated);
    if (event.success) add(total.units, event.units, saturated);
    add(total.host_elapsed_ns, event.host_elapsed_ns, saturated);
    saturated = saturated || event.byte_count_saturated;
}
void write_activity(std::ostream& out, const ExecutionActivityTotals& total) {
    out << "{\"calls\":" << total.calls << ",\"succeeded\":" << total.succeeded
        << ",\"failed\":" << total.failed << ",\"requested_bytes\":" << total.requested_bytes
        << ",\"accepted_logical_bytes\":" << total.accepted_bytes << ",\"units\":" << total.units
        << ",\"host_elapsed_ns\":" << total.host_elapsed_ns << '}';
}
constexpr std::array<const char*, static_cast<std::size_t>(ExecutionActivityKind::count)> activity_names{
    "observation_begin", "phase_elapsed", "token_commit", "proposed_state_write",
    "copy_sync", "copy_async", "memset_sync", "memset_async", "device_alloc", "device_free",
    "host_alloc", "host_free", "host_register", "host_unregister", "device_sync", "stream_sync",
    "event_sync", "stream_wait_event", "event_record", "stream_create", "stream_destroy", "event_create",
    "event_destroy", "graph_capture_begin", "graph_capture_end", "graph_instantiate", "graph_replay",
    "graph_destroy", "graph_exec_destroy", "host_launcher"};
constexpr std::array<const char*, static_cast<std::size_t>(ExecutionPhase::count)> phase_names{
    "setup", "admission", "prepare", "prefill", "token", "operator", "selection", "commit", "delivery"};
constexpr std::array<const char*, static_cast<std::size_t>(CopyDirection::count)> direction_names{
    "unspecified", "host_to_host", "host_to_device", "device_to_host", "device_to_device", "automatic"};
}
ExecutionTelemetry* current_execution_telemetry() noexcept { return inside_observer ? nullptr : active_observer; }
void observe_host_dispatch(ExecutionTelemetry& observer, const HostDispatchObservation& event) noexcept {
    if (inside_observer) return;
    inside_observer = true;
    observer.observe(event);
    inside_observer = false;
}
ExecutionCorrelation current_execution_correlation() noexcept { return active_correlation; }
void set_execution_correlation(ExecutionCorrelation value) noexcept { if (active_observer) active_correlation = value; }
bool execution_telemetry_during_capture() noexcept { return capture_depth != 0; }
void execution_telemetry_capture_change(bool begin) noexcept {
    if (!current_execution_telemetry()) return;
    if (begin) { if (capture_depth != UINT32_MAX) ++capture_depth; }
    else if (capture_depth) --capture_depth;
}
void execution_telemetry_replay_attempted() noexcept {
    if (current_execution_telemetry() && active_correlation.graph_replay != UINT64_MAX) ++active_correlation.graph_replay;
}
void observe_execution_activity(ExecutionActivityKind kind, std::uint64_t bytes, std::uint64_t units,
    bool success, std::uint64_t ns, CopyDirection direction, std::uintptr_t stream,
    std::int32_t result, bool byte_saturated) noexcept {
    if (!active_observer || inside_observer) return;
    inside_observer = true;
    active_observer->observe_activity({kind, active_correlation, bytes, units, ns,
        stream, direction, result, success, capture_depth != 0, byte_saturated});
    inside_observer = false;
}
ExecutionTelemetryBinding::ExecutionTelemetryBinding(ExecutionTelemetry* observer, ExecutionCorrelation correlation) noexcept
    : previous_(active_observer), previous_correlation_(active_correlation), previous_capture_depth_(capture_depth) {
    active_observer = observer; active_correlation = correlation; capture_depth = 0;
    observe_execution_activity(ExecutionActivityKind::observation_begin);
}
ExecutionTelemetryBinding::~ExecutionTelemetryBinding() {
    active_observer = previous_; active_correlation = previous_correlation_; capture_depth = previous_capture_depth_;
}
ExecutionCorrelationScope::ExecutionCorrelationScope(ExecutionCorrelation correlation) noexcept
    : previous_(active_correlation), enabled_(active_observer != nullptr) {
    if (enabled_) active_correlation = correlation;
}
ExecutionCorrelationScope::~ExecutionCorrelationScope() {
    if (enabled_) { previous_.graph_replay = active_correlation.graph_replay; active_correlation = previous_; }
}
ExecutionPhaseScope::ExecutionPhaseScope(ExecutionPhase phase) noexcept
    : previous_(active_correlation), enabled_(active_observer != nullptr) {
    if (enabled_) { active_correlation.phase = phase; start_ns_ = host_now(); exceptions_ = std::uncaught_exceptions(); }
}
ExecutionPhaseScope::~ExecutionPhaseScope() {
    if (enabled_) {
        observe_execution_activity(ExecutionActivityKind::phase_elapsed, 0, 1,
            std::uncaught_exceptions() == exceptions_, host_now() - start_ns_);
        active_correlation.phase = previous_.phase;
    }
}
void HostDispatchCounters::observe_activity(const ExecutionActivity& event) noexcept {
    const auto kind = static_cast<std::size_t>(event.kind);
    if (kind >= activities_.size()) return;
    if (event.kind != ExecutionActivityKind::host_launcher) activity_observed_ = true;
    count_activity(activities_[kind], event, saturated_);
    if (event.during_capture) count_activity(captured_activities_[kind], event, saturated_);
    if (event.kind == ExecutionActivityKind::copy_sync || event.kind == ExecutionActivityKind::copy_async) {
        const auto direction = static_cast<std::size_t>(event.direction);
        if (direction < copy_directions_.size()) count_activity(copy_directions_[direction], event, saturated_);
    }
    if (event.kind == ExecutionActivityKind::phase_elapsed) {
        const auto phase = static_cast<std::size_t>(event.correlation.phase);
        if (phase < phases_.size()) count_activity(phases_[phase], event, saturated_);
    }
    if (kind >= static_cast<std::size_t>(ExecutionActivityKind::copy_sync) &&
        kind < static_cast<std::size_t>(ExecutionActivityKind::host_launcher) &&
        event.correlation.phase == ExecutionPhase::operator_step) {
        const auto op = static_cast<std::size_t>(event.correlation.op);
        if (op < operator_activities_.size()) count_activity(operator_activities_[op], event, saturated_);
    }
    if (trace_enabled_) {
        if (trace_size_ < trace_.size()) trace_[trace_size_++] = event;
        else add(trace_dropped_, 1, saturated_);
    }
}
void write_host_dispatch_telemetry(std::ostream& out, const HostDispatchCounters& counters) {
    out << "{\"schema_version\":2,\"kind\":\"host_dispatch_telemetry\","
           "\"scope\":\"instrumented-host-calls-and-session-publication\",\"clock\":\"steady_clock\","
           "\"coverage\":\"direct runtime and decoder HIP calls; external library and driver internals unobserved\","
           "\"capture_activity_scope\":\"host-thread capture interval, not graph node execution\","
           "\"session_epoch_source\":\"state-device-allocation-generation; not a reset counter\",\"saturated\":"
        << (counters.saturated() ? "true" : "false") << ",\"totals\":{";
    write_totals(out, counters.totals()); out << "},\"operators\":[";
    bool first = true;
    for (std::size_t i = 0; i < counters.operators().size(); ++i) {
        const auto& total = counters.operators()[i];
        if (!total.returned && !total.rejected && !total.failed) continue;
        if (!first) out << ',';
        first = false; out << "{\"op\":\"" << to_string(static_cast<SerialOp>(i)) << "\",";
        write_totals(out, total); out << '}';
    }
    out << "],\"gpu_submits\":null,\"gpu_completed\":null,\"physical_transfer_bytes\":null,\"freed_bytes\":null";
    if (!counters.activity_observed()) {
        out << ",\"token_commits\":null,\"hip_copy_bytes\":null,\"hip_allocations\":null,"
               "\"hip_synchronizations\":null,\"graph_captures\":null,\"graph_replays\":null}";
        return;
    }
    const auto& activities = counters.activities();
    const auto units = [&](ExecutionActivityKind kind) { return activities[static_cast<std::size_t>(kind)].units; };
    out << ",\"token_commits\":" << units(ExecutionActivityKind::token_commit)
        << ",\"graph_captures\":" << activities[static_cast<std::size_t>(ExecutionActivityKind::graph_capture_end)].succeeded
        << ",\"graph_replays\":" << activities[static_cast<std::size_t>(ExecutionActivityKind::graph_replay)].succeeded
        << ",\"hip_activities\":{";
    for (std::size_t i = 0; i < activities.size(); ++i) {
        if (i) out << ',';
        out << '"' << activity_names[i] << "\":"; write_activity(out, activities[i]);
    }
    out << "},\"capture_activities\":{";
    for (std::size_t i = 0; i < counters.captured_activities().size(); ++i) {
        if (i) out << ',';
        out << '\"' << activity_names[i] << "\":"; write_activity(out, counters.captured_activities()[i]);
    }
    out << "},\"copy_directions\":{";
    for (std::size_t i = 0; i < counters.copy_directions().size(); ++i) {
        if (i) out << ',';
        out << '"' << direction_names[i] << "\":"; write_activity(out, counters.copy_directions()[i]);
    }
    out << "},\"operator_api_activities\":{";
    for (std::size_t i = 0; i < counters.operator_activities().size(); ++i) {
        if (i) out << ',';
        out << '\"' << to_string(static_cast<SerialOp>(i)) << "\":";
        write_activity(out, counters.operator_activities()[i]);
    }
    out << "},\"phases\":{";
    for (std::size_t i = 0; i < counters.phases().size(); ++i) {
        if (i) out << ',';
        out << '"' << phase_names[i] << "\":"; write_activity(out, counters.phases()[i]);
    }
    out << "},\"trace_dropped\":" << counters.trace_dropped() << ",\"trace\":[";
    first = true;
    for (const auto& event : counters.trace()) {
        if (!first) out << ',';
        first = false;
        const auto& c = event.correlation;
        out << "{\"kind\":\"" << activity_names[static_cast<std::size_t>(event.kind)]
            << "\",\"request_id\":" << c.request_id << ",\"session_epoch\":" << c.session_epoch
            << ",\"execution_identity\":" << c.execution_identity << ",\"position\":" << c.position
            << ",\"row\":" << c.row << ",\"rows\":" << c.rows << ",\"step_id\":" << c.step_id
            << ",\"layer\":" << c.layer << ",\"op\":\"" << to_string(c.op) << "\",\"phase\":" << static_cast<std::uint32_t>(c.phase)
            << ",\"graph_replay\":" << c.graph_replay << ",\"during_capture\":" << (event.during_capture ? "true" : "false")
            << ",\"stream\":" << event.stream << ",\"success\":" << (event.success ? "true" : "false")
            << ",\"result_code\":" << event.result_code << ",\"requested_bytes\":" << event.requested_bytes
            << ",\"units\":" << event.units << ",\"host_elapsed_ns\":" << event.host_elapsed_ns << '}';
    }
    out << "]}";
}

GpuTraceTime summarize_gpu_trace(std::span<const GpuTraceInterval> intervals) {
    GpuTraceTime result;
    if (intervals.empty()) return result;
    std::vector<GpuTraceInterval> ordered(intervals.begin(), intervals.end());
    for (const auto& interval : ordered) {
        if (interval.end_ns < interval.begin_ns || interval.clock_domain != ordered.front().clock_domain)
            throw std::invalid_argument("GPU trace requires nonnegative durations in one clock domain");
        const auto duration = interval.end_ns - interval.begin_ns;
        if (duration > std::numeric_limits<std::uint64_t>::max() - result.summed_ns)
            throw std::overflow_error("GPU trace duration sum overflow");
        result.summed_ns += duration;
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        return left.begin_ns < right.begin_ns;
    });
    auto begin = ordered.front().begin_ns;
    auto end = ordered.front().end_ns;
    for (const auto& interval : ordered) {
        if (interval.begin_ns > end) {
            result.busy_union_ns += end - begin;
            begin = interval.begin_ns;
        }
        end = std::max(end, interval.end_ns);
    }
    result.busy_union_ns += end - begin;
    return result;
}
}  // namespace strix
