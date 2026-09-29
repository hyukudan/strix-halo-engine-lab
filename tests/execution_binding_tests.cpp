#include "strix/execution_binding.hpp"

#include <iostream>
#include <string>
#include <utility>

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
    const strix::kernels::ShapeClass& shape) {
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
    variant.compiler_identity = "test";
    variant.measured_seconds = 0.0;
    return variant;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(model);

    strix::kernels::KernelRegistry registry;

    strix::kernels::ShapeClass recurrent_shape;
    recurrent_shape.logical_rows = 1;
    recurrent_shape.request_concurrency = 1;
    recurrent_shape.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);
    recurrent_shape.auxiliary_size =
        static_cast<std::uint32_t>(
            model.gdn_value_head_dim);

    registry.add(
        make_variant(
            "gdn.reference",
            strix::kernels::OperatorFamily::recurrent,
            recurrent_shape));

    strix::ExecutionBindingRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;

    const auto bindings =
        strix::bind_serial_execution_plan(
            model, plan, registry, request);

    expect(bindings.bindings.size() == plan.steps.size(),
           "binding plan preserves every execution step");
    expect(bindings.host_or_runtime_steps == 1,
           "position preparation is explicitly runtime-managed");
    expect(bindings.bound_steps == 36,
           "one recurrent variant binds all thirty-six recurrent layers");
    expect(bindings.missing_steps ==
               plan.steps.size() -
                   bindings.bound_steps -
                   bindings.host_or_runtime_steps,
           "all unresolved kernel families are counted explicitly");

    const auto family =
        strix::operator_family_for(
            strix::SerialOp::moe_route);
    expect(family.has_value() &&
               *family ==
                   strix::kernels::OperatorFamily::moe_route,
           "MoE route maps to the route kernel family");

    const auto no_family =
        strix::operator_family_for(
            strix::SerialOp::position_prepare);
    expect(!no_family.has_value(),
           "position preparation remains runtime-managed");

    request.logical_rows = 2;
    const auto rows_two =
        strix::bind_serial_execution_plan(
            model, plan, registry, request);
    expect(rows_two.bound_steps == 0,
           "row-class mismatch prevents accidental C1 kernel reuse");

    if (failures == 0) {
        std::cout << "execution binding tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
