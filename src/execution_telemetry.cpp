#include "strix/execution_telemetry.hpp"

#include <algorithm>
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
}

void write_host_dispatch_telemetry(std::ostream& out, const HostDispatchCounters& counters) {
    out << "{\"schema_version\":1,\"kind\":\"host_dispatch_telemetry\","
           "\"scope\":\"host-launcher-calls-only\",\"clock\":\"steady_clock\",\"saturated\":"
        << (counters.saturated() ? "true" : "false") << ",\"totals\":{";
    write_totals(out, counters.totals());
    out << "},\"operators\":[";
    bool first = true;
    for (std::size_t i = 0; i < counters.operators().size(); ++i) {
        const auto& total = counters.operators()[i];
        if (!total.returned && !total.rejected && !total.failed) continue;
        if (!first) out << ',';
        first = false;
        out << "{\"op\":\"" << to_string(static_cast<SerialOp>(i)) << "\",";
        write_totals(out, total);
        out << '}';
    }
    // These quantities are unobserved by this host hook, including after a
    // launcher that partially enqueues and throws. They must stay unknown.
    out << "],\"gpu_submits\":null,\"gpu_completed\":null,\"token_commits\":null,"
           "\"hip_copy_bytes\":null,\"hip_allocations\":null,\"hip_synchronizations\":null,"
           "\"graph_captures\":null,\"graph_replays\":null}";
}

GpuTraceTime summarize_gpu_trace(std::span<const GpuTraceInterval> intervals) {
    GpuTraceTime result;
    if (intervals.empty()) return result;
    std::vector<GpuTraceInterval> ordered(intervals.begin(), intervals.end());
    for (const auto& interval : ordered) {
        if (interval.end_ns < interval.begin_ns || interval.clock_domain != ordered.front().clock_domain)
            throw std::invalid_argument("GPU trace requires ordered ranges in one clock domain");
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
