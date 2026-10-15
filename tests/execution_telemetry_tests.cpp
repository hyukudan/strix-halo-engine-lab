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
