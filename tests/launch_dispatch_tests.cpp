#include "strix/launch_dispatch.hpp"

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
    std::uint32_t last_step = 0;
};

void count_launch(
    const strix::ExecutionDispatchContext& context,
    void* user_data) {
    auto& counter =
        *static_cast<Counter*>(
            user_data);
    ++counter.calls;
    counter.last_step =
        context.step_id;
}

}  // namespace

int main() {
    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x1234ULL;

    strix::ExecutionStep kernel_step;
    kernel_step.id = 0;
    kernel_step.op =
        strix::SerialOp::token_embedding;
    compiled.execution.steps.push_back(
        kernel_step);

    strix::ExecutionStep runtime_step;
    runtime_step.id = 1;
    runtime_step.op =
        strix::SerialOp::position_prepare;
    runtime_step.dependencies = {0};
    compiled.execution.steps.push_back(
        runtime_step);

    strix::ExecutionKernelBinding kernel_binding;
    kernel_binding.step_id = 0;
    kernel_binding.op =
        strix::SerialOp::token_embedding;
    kernel_binding.family =
        strix::kernels::OperatorFamily::embedding_lookup;
    kernel_binding.kernel_id =
        "kernel.embedding";
    compiled.bindings.bindings.push_back(
        kernel_binding);
    ++compiled.bindings.bound_steps;

    strix::ExecutionKernelBinding runtime_binding;
    runtime_binding.step_id = 1;
    runtime_binding.op =
        strix::SerialOp::position_prepare;
    runtime_binding.host_or_runtime_step = true;
    compiled.bindings.bindings.push_back(
        runtime_binding);
    ++compiled.bindings.host_or_runtime_steps;

    Counter kernel_counter;
    Counter runtime_counter;

    strix::ExecutionLauncherRegistry registry;
    registry.add_kernel_launcher(
        {
            "kernel.embedding",
            0x1001ULL,
            &count_launch,
            &kernel_counter,
        });
    registry.add_runtime_launcher(
        strix::SerialOp::position_prepare,
        {
            "runtime.position",
            0x2001ULL,
            &count_launch,
            &runtime_counter,
        });

    const auto plan =
        strix::compile_launch_plan(
            compiled,
            registry);
    expect(plan.steps.size() == 2 &&
               plan.plan_identity != 0,
           "launch plan pre-resolves every execution step");

    strix::DirectLaunchDispatcher dispatcher(
        plan);

    strix::ExecutionDispatchContext kernel_context;
    kernel_context.step_id = 0;
    kernel_context.op =
        strix::SerialOp::token_embedding;
    dispatcher.dispatch_kernel(
        kernel_context);

    strix::ExecutionDispatchContext runtime_context;
    runtime_context.step_id = 1;
    runtime_context.op =
        strix::SerialOp::position_prepare;
    dispatcher.dispatch_runtime_step(
        runtime_context);

    expect(kernel_counter.calls == 1 &&
               kernel_counter.last_step == 0,
           "kernel launcher dispatches by dense step id");
    expect(runtime_counter.calls == 1 &&
               runtime_counter.last_step == 1,
           "runtime launcher dispatches by dense step id");


    const auto complete_coverage =
        strix::audit_launcher_coverage(
            compiled,
            registry);
    expect(
        complete_coverage.covered_steps == 2 &&
            complete_coverage.missing_steps == 0 &&
            complete_coverage.missing.empty(),
        "coverage audit reports complete launcher registry");

    strix::ExecutionLauncherRegistry kernel_only;
    kernel_only.add_kernel_launcher(
        {
            "kernel.embedding",
            0x1001ULL,
            &count_launch,
            &kernel_counter,
        });
    const auto missing_coverage =
        strix::audit_launcher_coverage(
            compiled,
            kernel_only);
    expect(
        missing_coverage.covered_steps == 1 &&
            missing_coverage.missing_steps == 1 &&
            missing_coverage.missing.size() == 1 &&
            missing_coverage.missing[0].step_id == 1 &&
            missing_coverage.missing[0].op ==
                strix::SerialOp::position_prepare &&
            missing_coverage.missing[0].runtime_step,
        "coverage audit reports missing launchers with step identity");

    auto registry_changed =
        strix::ExecutionLauncherRegistry{};
    registry_changed.add_kernel_launcher(
        {
            "kernel.embedding",
            0x1002ULL,
            &count_launch,
            &kernel_counter,
        });
    registry_changed.add_runtime_launcher(
        strix::SerialOp::position_prepare,
        {
            "runtime.position",
            0x2001ULL,
            &count_launch,
            &runtime_counter,
        });
    const auto changed =
        strix::compile_launch_plan(
            compiled,
            registry_changed);
    expect(changed.plan_identity !=
               plan.plan_identity,
           "launcher implementation identity changes compiled launch identity");

    strix::ExecutionLauncherRegistry missing;
    missing.add_kernel_launcher(
        {
            "kernel.embedding",
            0x1001ULL,
            &count_launch,
            &kernel_counter,
        });
    bool missing_rejected = false;
    try {
        static_cast<void>(
            strix::compile_launch_plan(
                compiled,
                missing));
    } catch (const std::invalid_argument&) {
        missing_rejected = true;
    }
    expect(missing_rejected,
           "launch compilation rejects missing runtime launcher");

    if (failures == 0) {
        std::cout << "launch dispatch tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
