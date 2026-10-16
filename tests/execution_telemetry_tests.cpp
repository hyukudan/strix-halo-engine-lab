#include "strix/execution_telemetry.hpp"
#include "strix/launch_dispatch.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class Exception, class Function> void rejects(Function operation) {
    try { operation(); } catch (const Exception&) { return; }
    throw std::runtime_error("missing expected exception");
}
struct State { unsigned calls = 0; bool fail = false; };
void launch(const strix::ExecutionDispatchContext&, void* data) {
    auto& state = *static_cast<State*>(data);
    ++state.calls;
    if (state.fail) throw std::runtime_error("original launcher error");
}
struct Observer final : strix::ExecutionTelemetry {
    strix::HostDispatchCounters counters;
    std::array<strix::HostDispatchObservation, 8> events{};
    std::size_t size = 0;
    void observe(const strix::HostDispatchObservation& event) noexcept override {
        counters.observe(event);
        if (size < events.size()) events[size++] = event;
    }
};
}

int main() try {
    using namespace strix;
    State state;
    CompiledLaunchPlan plan;
    plan.execution_identity = 17;
    plan.plan_identity = 21;
    plan.steps = {{0, SerialOp::lm_head, false, 5, launch, &state},
                  {1, SerialOp::position_prepare, true, 6, launch, &state}};
    ExecutionDispatchContext context;
    context.execution_identity = 17;
    context.position = 103;
    context.op = SerialOp::lm_head;
    context.layer = 47;
    Observer observer;
    DirectLaunchDispatcher plain(plan);
    DirectLaunchDispatcher measured(plan, &observer);
    plain.dispatch_kernel(context);
    require(state.calls == 1 && observer.size == 0, "disabled telemetry has no callbacks");
    measured.dispatch_kernel(context);
    require(state.calls == 2 && observer.counters.totals().returned == 1,
            "enabled dispatch calls exactly the same launcher");
    const auto& event = observer.events[0];
    require(event.execution_identity == 17 && event.position == 103 && event.layer == 47 &&
            event.op == SerialOp::lm_head && !event.runtime_step && event.step_id == 0 &&
            event.logical_rows == 1 && event.request_concurrency == 1,
            "correlation describes actual dispatch, not just aggregate count");

    context.execution_identity = 18;
    rejects<std::invalid_argument>([&] { measured.dispatch_kernel(context); });
    require(state.calls == 2 && observer.counters.totals().rejected == 1,
            "prelaunch rejection does not invoke launcher");
    context.execution_identity = 17;
    state.fail = true;
    try { measured.dispatch_kernel(context); }
    catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "original launcher error", "original error preserved");
    }
    require(state.calls == 3 && observer.counters.totals().failed == 1,
            "failed launcher is not a returned call or claimed GPU submit");
    state.fail = false;
    context.step_id = 1;
    context.op = SerialOp::position_prepare;
    measured.dispatch_runtime_step(context);
    require(observer.counters.totals().returned == 2 && observer.events[3].runtime_step,
            "runtime-managed host work is observed and attributed to its operator");
    require(observer.counters.operators()[static_cast<std::size_t>(SerialOp::lm_head)].returned == 1,
            "operator attribution excludes runtime step");

    std::ostringstream json;
    write_host_dispatch_telemetry(json, observer.counters);
    require(json.str().find("\"gpu_submits\":null") != std::string::npos &&
            json.str().find("\"graph_replays\":null") != std::string::npos &&
            json.str().find("\"token_commits\":null") != std::string::npos,
            "unobserved GPU, replay and publication metrics stay unknown");
    HostDispatchCounters saturated;
    HostDispatchObservation enormous;
    enormous.host_elapsed_ns = std::numeric_limits<std::uint64_t>::max();
    saturated.observe(enormous);
    enormous.host_elapsed_ns = 1;
    saturated.observe(enormous);
    require(saturated.saturated() && saturated.totals().host_elapsed_ns ==
            std::numeric_limits<std::uint64_t>::max(), "counter overflow is explicit and contained");

    // Source fixtures: these are not a runtime/driver observation or a test run.
    HostDispatchCounters activity;
    activity.enable_trace(true);
    ExecutionCorrelation correlation;
    correlation.request_id = 17; correlation.session_epoch = 29;
    correlation.position = 41; correlation.row = 1; correlation.rows = 1;
    correlation.step_id = 53; correlation.layer = 7; correlation.op = SerialOp::lm_head;
    {
        ExecutionTelemetryBinding binding(&activity, correlation);
        observe_execution_activity(ExecutionActivityKind::copy_async, 64, 1, true, 3,
            CopyDirection::host_to_device, 11);
        observe_execution_activity(ExecutionActivityKind::copy_sync, 128, 1, false, 5,
            CopyDirection::device_to_host, 12, -1);
        observe_execution_activity(ExecutionActivityKind::device_alloc, 256);
        observe_execution_activity(ExecutionActivityKind::device_free);
        observe_execution_activity(ExecutionActivityKind::proposed_state_write);
        require(activity.activities()[static_cast<std::size_t>(ExecutionActivityKind::token_commit)].units == 0,
            "proposed persistent generations do not commit a token");
        observe_execution_activity(ExecutionActivityKind::token_commit, 0, 2);
        execution_telemetry_capture_change(true);
        observe_execution_activity(ExecutionActivityKind::copy_async, 8, 1, true, 7,
            CopyDirection::host_to_device, 11);
        observe_execution_activity(ExecutionActivityKind::graph_capture_end);
        execution_telemetry_capture_change(false);
        auto inner = current_execution_correlation(); inner.row = 0; inner.position = 42;
        {
            ExecutionCorrelationScope nested(inner);
            for (unsigned repeat = 0; repeat < 3; ++repeat) {
                execution_telemetry_replay_attempted();
                observe_execution_activity(ExecutionActivityKind::graph_replay);
            }
        }
        require(current_execution_correlation().row == 1 && current_execution_correlation().position == 41 &&
            current_execution_correlation().graph_replay == 3, "scope restores row but retains global replay ordinal");
        {
            ExecutionTelemetryBinding disabled(nullptr);
            observe_execution_activity(ExecutionActivityKind::token_commit, 0, 99);
            require(current_execution_telemetry() == nullptr, "nested null binding disables observation");
        }
        require(current_execution_telemetry() == &activity, "nested binding restores owner");
        auto invalid_event = ExecutionActivity{};
        invalid_event.kind = static_cast<ExecutionActivityKind>(UINT32_MAX);
        activity.observe_activity(invalid_event); // Invalid kind cannot index storage or produce a trace entry.
    }
    require(current_execution_telemetry() == nullptr && !execution_telemetry_during_capture(),
        "binding lifetime restores outer observer/capture state");
    const auto& h2d = activity.copy_directions()[static_cast<std::size_t>(CopyDirection::host_to_device)];
    const auto& d2h = activity.copy_directions()[static_cast<std::size_t>(CopyDirection::device_to_host)];
    require(h2d.calls == 2 && h2d.accepted_bytes == 72 && h2d.requested_bytes == 72 &&
        d2h.failed == 1 && d2h.requested_bytes == 128 && !d2h.accepted_bytes,
        "failed copies retain requested bytes but never count accepted logical bytes");
    require(activity.captured_activities()[static_cast<std::size_t>(ExecutionActivityKind::copy_async)].calls == 1 &&
        activity.activities()[static_cast<std::size_t>(ExecutionActivityKind::graph_replay)].succeeded == 3 &&
        activity.activities()[static_cast<std::size_t>(ExecutionActivityKind::graph_capture_end)].succeeded == 1,
        "capture host calls, a completed capture API and N replay APIs remain separate");
    require(activity.trace()[1].correlation.request_id == 17 && activity.trace()[1].correlation.session_epoch == 29 &&
        activity.trace()[1].correlation.step_id == 53 && activity.trace()[1].correlation.layer == 7 &&
        activity.trace()[1].stream == 11, "trace contains actual supplied request/epoch/operator/row/stream correlation");
    std::ostringstream activity_json;
    write_host_dispatch_telemetry(activity_json, activity);
    require(activity_json.str().find("\"token_commits\":2") != std::string::npos &&
        activity_json.str().find("\"gpu_submits\":null") != std::string::npos &&
        activity_json.str().find("\"graph_replays\":3") != std::string::npos,
        "publication/capture/replay events do not manufacture kernel submits or completions");
    HostDispatchCounters bounded;
    bounded.enable_trace(true);
    {
        ExecutionTelemetryBinding binding(&bounded);
        for (unsigned i = 0; i < 300; ++i) observe_execution_activity(ExecutionActivityKind::stream_sync);
    }
    require(bounded.trace().size() == 256 && bounded.trace_dropped() == 45 &&
        bounded.activities()[static_cast<std::size_t>(ExecutionActivityKind::stream_sync)].calls == 300,
        "fixed trace truncation preserves complete aggregate counters");
    struct ReentrantObserver final : ExecutionTelemetry {
        unsigned calls = 0;
        void observe(const HostDispatchObservation&) noexcept override {}
        void observe_activity(const ExecutionActivity&) noexcept override {
            ++calls;
            observe_execution_activity(ExecutionActivityKind::device_alloc, 99);
        }
    } reentrant;
    {
        ExecutionTelemetryBinding binding(&reentrant);
        observe_execution_activity(ExecutionActivityKind::device_free);
    }
    require(reentrant.calls == 2, "observer callback recursion is suppressed, including instrumented HIP calls");
    ExecutionActivity huge;
    huge.kind = ExecutionActivityKind::copy_sync; huge.requested_bytes = UINT64_MAX;
    bounded.observe_activity(huge); bounded.observe_activity(huge);
    require(bounded.saturated() && bounded.activities()[static_cast<std::size_t>(ExecutionActivityKind::copy_sync)].accepted_bytes == UINT64_MAX,
        "logical byte counters saturate instead of wrapping into the execution path");
    {
        ExecutionTelemetryBinding binding(&activity, correlation);
        try { ExecutionPhaseScope phase(ExecutionPhase::selection); throw std::runtime_error("original"); }
        catch (const std::runtime_error&) {}
    }
    require(activity.phases()[static_cast<std::size_t>(ExecutionPhase::selection)].failed == 1,
        "semantic host phase exceptions are observed without altering propagation");

    const std::array<GpuTraceInterval, 5> overlap{{
        {30, 35, 2, 9}, {0, 10, 1, 9}, {5, 20, 2, 9}, {8, 12, 3, 9}, {20, 20, 3, 9}}};
    const auto time = summarize_gpu_trace(overlap);
    require(time.summed_ns == 34 && time.busy_union_ns == 25,
            "multi-queue union handles overlap, nesting, gaps and zero-length intervals");
    require(summarize_gpu_trace({}).busy_union_ns == 0, "empty actual trace has zero duration");
    const std::array<GpuTraceInterval, 2> mixed{{{0, 1, 1, 1}, {0, 1, 2, 2}}};
    rejects<std::invalid_argument>([&] { static_cast<void>(summarize_gpu_trace(mixed)); });
    const std::array<GpuTraceInterval, 1> reversed{{{2, 1, 0, 0}}};
    rejects<std::invalid_argument>([&] { static_cast<void>(summarize_gpu_trace(reversed)); });
    const auto max = std::numeric_limits<std::uint64_t>::max();
    const std::array<GpuTraceInterval, 2> overflow{{{0, max, 1, 0}, {0, 1, 2, 0}}};
    rejects<std::overflow_error>([&] { static_cast<void>(summarize_gpu_trace(overflow)); });
    std::cout << "execution telemetry PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
