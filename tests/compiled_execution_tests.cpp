#include "strix/compiled_execution.hpp"

#include <set>
#include <string>
#include <utility>

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

strix::kernels::KernelVariant make_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape,
    double measured_seconds) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::decode;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "test";
    variant.output_layout = "test";
    variant.compiler_identity = "test-compiler";
    variant.measured_seconds = measured_seconds;
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    const strix::ExecutionBindingRequest& request,
    double measured_seconds = 0.0,
    std::string suffix = "") {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(model);

    std::set<std::string> added;
    for (const auto& step : execution.steps) {
        const auto family =
            strix::operator_family_for(step.op);
        if (!family.has_value()) {
            continue;
        }

        const auto shape =
            strix::execution_shape_for(
                model, step.op, request);

        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(shape.logical_rows) +
            "." +
            std::to_string(shape.request_concurrency) +
            "." +
            std::to_string(shape.hidden_size) +
            "." +
            std::to_string(shape.auxiliary_size);

        if (!added.insert(key).second) {
            continue;
        }

        registry.add(
            make_variant(
                "test." + key + suffix,
                *family,
                shape,
                measured_seconds));
    }

    return registry;
}

struct NullDispatcher final
    : strix::EagerExecutionDispatcher {
    std::uint32_t kernels = 0;
    std::uint32_t runtime = 0;

    void dispatch_kernel(
        const strix::ExecutionDispatchContext&) override {
        ++kernels;
    }

    void dispatch_runtime_step(
        const strix::ExecutionDispatchContext&) override {
        ++runtime;
    }
};

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::ExecutionBindingRequest binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;

    const auto registry =
        make_registry(
            model,
            binding_request);

    const auto compiled =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            256);

    expect(compiled.bindings.missing_steps == 0,
           "compiled execution is fully bound");
    expect(compiled.kernel_plan_identity != 0 &&
               compiled.execution_identity != 0,
           "compiled execution has stable non-zero identities");
    expect(compiled.arena.total_bytes > 0,
           "compiled execution contains a physical scratch arena");
    expect(!compiled.whole_plan_graph_safe,
           "full decode plan remains non-graph-safe while runtime/dynamic steps exist");
    expect(compiled.dynamic_steps > 0,
           "compiled plan counts dynamic-shape steps");

    const auto compiled_again =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            256);
    expect(compiled_again.kernel_plan_identity ==
               compiled.kernel_plan_identity &&
               compiled_again.execution_identity ==
                   compiled.execution_identity,
           "compiled execution identity is reproducible");

    const auto remeasured_registry =
        make_registry(
            model,
            binding_request,
            9e-6);
    const auto remeasured =
        strix::compile_serial_execution(
            model,
            remeasured_registry,
            binding_request,
            buffer_request,
            256);
    expect(remeasured.execution_identity ==
               compiled.execution_identity,
           "benchmark-time noise does not change compiled execution identity");

    const auto different_registry =
        make_registry(
            model,
            binding_request,
            0.0,
            ".alternate");
    const auto different_kernel_plan =
        strix::compile_serial_execution(
            model,
            different_registry,
            binding_request,
            buffer_request,
            256);
    expect(different_kernel_plan.execution_identity !=
               compiled.execution_identity,
           "kernel identity change invalidates compiled execution identity");

    const auto different_alignment =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            512);
    expect(different_alignment.execution_identity !=
               compiled.execution_identity,
           "arena layout identity participates in compiled execution identity");

    strix::EagerExecutionRequest run;
    run.logical_rows = 1;
    run.request_concurrency = 1;
    run.state.valid = all_state();

    NullDispatcher dispatcher;
    const auto result =
        strix::execute_compiled_serial_eager(
            compiled,
            run,
            dispatcher);
    expect(result.completed_steps ==
               compiled.execution.steps.size(),
           "compiled eager execution completes the frozen DAG");
    expect(dispatcher.kernels ==
               compiled.bindings.bound_steps &&
               dispatcher.runtime ==
                   compiled.bindings.host_or_runtime_steps,
           "compiled eager execution dispatches frozen kernel/runtime steps");

    auto wrong_shape = run;
    wrong_shape.logical_rows = 2;
    bool shape_rejected = false;
    try {
        NullDispatcher unused;
        static_cast<void>(
            strix::execute_compiled_serial_eager(
                compiled,
                wrong_shape,
                unused));
    } catch (const std::invalid_argument&) {
        shape_rejected = true;
    }
    expect(shape_rejected,
           "compiled execution rejects runtime shape mismatch");

    if (failures == 0) {
        std::cout << "compiled execution tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
