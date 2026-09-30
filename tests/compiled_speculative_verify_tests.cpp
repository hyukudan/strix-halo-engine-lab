#include "strix/compiled_speculative_verify.hpp"

#include "strix/execution_binding.hpp"

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
    double measured_seconds = 0.0) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::
            speculative_verify;
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
        strix::kernels::QualificationStatus::
            oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "verify";
    variant.output_layout = "verify";
    variant.compiler_identity = "fixture";
    variant.measured_seconds =
        measured_seconds;
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    std::uint32_t depth,
    std::string suffix = "") {
    strix::kernels::KernelRegistry registry;
    const auto verify =
        strix::build_speculative_verify_plan(
            model,
            depth);

    strix::ExecutionBindingRequest shape_request;
    shape_request.logical_rows = depth;
    shape_request.request_concurrency = 1;

    std::set<std::string> unique;
    for (const auto& step :
         verify.steps) {
        if (step.op ==
            strix::SerialOp::position_prepare) {
            continue;
        }
        const auto family =
            strix::operator_family_for(
                step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model,
                step.op,
                shape_request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(
                shape.auxiliary_size);
        if (!unique.insert(key).second) {
            continue;
        }
        registry.add(
            make_variant(
                "verify." + key + suffix,
                *family,
                shape));
    }
    return registry;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SpeculativeVerifyBindingRequest binding;
    binding.request_concurrency = 1;

    strix::ExecutionBufferRequest buffers;
    buffers.logical_rows = 4;

    const auto registry =
        make_registry(model, 4);
    const auto compiled =
        strix::compile_speculative_verify_execution(
            model,
            4,
            registry,
            binding,
            buffers,
            256);

    expect(compiled.candidate_depth == 4 &&
               compiled.bindings.missing_steps == 0 &&
               compiled.execution_identity != 0 &&
               compiled.kernel_plan_identity != 0,
           "compiled verify freezes depth, bindings and identities");
    expect(compiled.arena.total_bytes > 0 &&
               compiled.sequential_window_steps == 37 &&
               compiled.causal_window_steps == 12,
           "compiled verify includes D-row arena and stateful window counts");
    expect(!compiled.whole_plan_graph_safe,
           "whole verify plan remains non-graph-safe while dynamic/runtime steps exist");

    const auto compiled_again =
        strix::compile_speculative_verify_execution(
            model,
            4,
            registry,
            binding,
            buffers,
            256);
    expect(compiled_again.execution_identity ==
               compiled.execution_identity,
           "verify execution identity is reproducible");

    const auto different_registry =
        make_registry(
            model,
            4,
            ".alternate");
    const auto different =
        strix::compile_speculative_verify_execution(
            model,
            4,
            different_registry,
            binding,
            buffers,
            256);
    expect(different.execution_identity !=
               compiled.execution_identity,
           "verify kernel identity change invalidates execution identity");

    auto wrong_rows = buffers;
    wrong_rows.logical_rows = 2;
    bool rows_rejected = false;
    try {
        static_cast<void>(
            strix::compile_speculative_verify_execution(
                model,
                4,
                registry,
                binding,
                wrong_rows,
                256));
    } catch (const std::invalid_argument&) {
        rows_rejected = true;
    }
    expect(rows_rejected,
           "verify compile rejects buffer rows that differ from depth");

    auto wrong_concurrency = binding;
    wrong_concurrency.request_concurrency = 2;
    bool concurrency_rejected = false;
    try {
        static_cast<void>(
            strix::compile_speculative_verify_execution(
                model,
                4,
                registry,
                wrong_concurrency,
                buffers,
                256));
    } catch (const std::invalid_argument&) {
        concurrency_rejected = true;
    }
    expect(concurrency_rejected,
           "initial verify compile keeps request concurrency at one");

    if (failures == 0) {
        std::cout << "compiled speculative verify tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
