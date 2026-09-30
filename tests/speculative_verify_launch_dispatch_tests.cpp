#include "strix/launch_dispatch.hpp"
#include "strix/runtime_launchers.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct Counter {
    std::uint32_t calls = 0;
};

void count_launch(
    const strix::ExecutionDispatchContext&,
    void* user_data) {
    ++static_cast<Counter*>(user_data)->calls;
}

strix::CompiledSpeculativeVerifyExecution fixture() {
    strix::CompiledSpeculativeVerifyExecution compiled;
    compiled.execution_identity =
        0x5645524946590001ULL;
    compiled.candidate_depth = 4;
    compiled.request_concurrency = 1;

    strix::ExecutionStep embedding;
    embedding.id = 0;
    embedding.op =
        strix::SerialOp::token_embedding;
    compiled.topology.steps.push_back(
        embedding);

    strix::ExecutionStep position;
    position.id = 1;
    position.op =
        strix::SerialOp::position_prepare;
    position.dependencies = {0};
    compiled.topology.steps.push_back(
        position);

    strix::ExecutionKernelBinding kernel;
    kernel.step_id = 0;
    kernel.op =
        strix::SerialOp::token_embedding;
    kernel.family =
        strix::kernels::OperatorFamily::
            embedding_lookup;
    kernel.kernel_id =
        "verify.embedding";
    compiled.bindings.bindings.push_back(
        kernel);
    ++compiled.bindings.bound_steps;

    strix::ExecutionKernelBinding runtime;
    runtime.step_id = 1;
    runtime.op =
        strix::SerialOp::position_prepare;
    runtime.host_or_runtime_step = true;
    compiled.bindings.bindings.push_back(
        runtime);
    ++compiled.bindings.host_or_runtime_steps;

    return compiled;
}

}  // namespace

int main() {
    auto compiled = fixture();

    Counter kernel_counter;
    Counter runtime_counter;

    strix::ExecutionLauncherRegistry launchers;
    launchers.add_kernel_launcher(
        {
            "verify.embedding",
            0x1001ULL,
            &count_launch,
            &kernel_counter,
        });
    launchers.add_runtime_launcher(
        strix::SerialOp::position_prepare,
        {
            "runtime.position_verify_window",
            strix::position_verify_window_launcher_identity,
            &count_launch,
            &runtime_counter,
        });

    const auto coverage =
        strix::audit_launcher_coverage(
            compiled,
            launchers);
    expect(coverage.covered_steps == 2 &&
               coverage.missing_steps == 0,
           "verify launcher audit covers kernel and runtime steps");

    const auto plan =
        strix::compile_launch_plan(
            compiled,
            launchers);
    expect(plan.execution_identity ==
               compiled.execution_identity &&
               plan.steps.size() == 2 &&
               plan.plan_identity != 0,
           "verify launch plan freezes dense launcher table");

    strix::DirectLaunchDispatcher dispatcher(
        plan);

    strix::ExecutionDispatchContext kernel_context;
    kernel_context.execution_identity = compiled.execution_identity;
    kernel_context.step_id = 0;
    kernel_context.op =
        strix::SerialOp::token_embedding;
    dispatcher.dispatch_kernel(
        kernel_context);

    strix::ExecutionDispatchContext runtime_context;
    runtime_context.execution_identity = compiled.execution_identity;
    runtime_context.step_id = 1;
    runtime_context.op =
        strix::SerialOp::position_prepare;
    dispatcher.dispatch_runtime_step(
        runtime_context);

    expect(kernel_counter.calls == 1 &&
               runtime_counter.calls == 1,
           "verify direct dispatcher executes pre-resolved launchers");

    strix::ExecutionLauncherRegistry serial_runtime;
    serial_runtime.add_kernel_launcher(
        {
            "verify.embedding",
            0x1001ULL,
            &count_launch,
            &kernel_counter,
        });
    serial_runtime.add_runtime_launcher(
        strix::SerialOp::position_prepare,
        strix::make_position_prepare_launcher());

    const auto serial_runtime_plan =
        strix::compile_launch_plan(
            compiled,
            serial_runtime);
    expect(serial_runtime_plan.plan_identity !=
               plan.plan_identity,
           "serial and verify position launchers have distinct plan identity");

    strix::ExecutionLauncherRegistry missing;
    auto other_execution = compiled;
    ++other_execution.execution_identity;
    const auto other_plan = strix::compile_launch_plan(
        other_execution, launchers);
    strix::DirectLaunchDispatcher other_dispatcher(other_plan);
    for (const auto identity : {std::uint64_t{0}, compiled.execution_identity}) {
        auto foreign_kernel = kernel_context;
        foreign_kernel.execution_identity = identity;
        auto foreign_runtime = runtime_context;
        foreign_runtime.execution_identity = identity;
        bool kernel_rejected = false;
        bool runtime_rejected = false;
        try {
            other_dispatcher.dispatch_kernel(foreign_kernel);
        } catch (const std::invalid_argument&) {
            kernel_rejected = true;
        }
        try {
            other_dispatcher.dispatch_runtime_step(foreign_runtime);
        } catch (const std::invalid_argument&) {
            runtime_rejected = true;
        }
        expect(kernel_rejected && runtime_rejected,
               "dispatcher rejects missing or foreign execution identity");
    }
    expect(kernel_counter.calls == 1 && runtime_counter.calls == 1,
           "rejected contexts never invoke kernel or runtime launchers");
    auto uninitialized_plan = plan;
    uninitialized_plan.execution_identity = 0;
    strix::DirectLaunchDispatcher uninitialized_dispatcher(uninitialized_plan);
    auto uncompiled_context = kernel_context;
    uncompiled_context.execution_identity = 0;
    bool uninitialized_rejected = false;
    try {
        uninitialized_dispatcher.dispatch_kernel(uncompiled_context);
    } catch (const std::invalid_argument&) {
        uninitialized_rejected = true;
    }
    expect(uninitialized_rejected && kernel_counter.calls == 1,
           "zero identities cannot bypass compiled dispatch validation");

    missing.add_kernel_launcher(
        {
            "verify.embedding",
            0x1001ULL,
            &count_launch,
            &kernel_counter,
        });

    const auto incomplete =
        strix::audit_launcher_coverage(
            compiled,
            missing);
    expect(incomplete.missing_steps == 1 &&
               incomplete.missing[0].runtime_step,
           "verify audit reports missing position-window runtime launcher");

    bool compile_rejected = false;
    try {
        static_cast<void>(
            strix::compile_launch_plan(
                compiled,
                missing));
    } catch (const std::invalid_argument&) {
        compile_rejected = true;
    }
    expect(compile_rejected,
           "verify launch compilation rejects incomplete registry");

    if (failures == 0) {
        std::cout
            << "speculative verify launch dispatch tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
