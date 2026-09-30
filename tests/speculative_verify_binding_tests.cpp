#include "strix/speculative_verify_binding.hpp"

#include <set>
#include <string>
#include <utility>

#include <iostream>

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
    strix::kernels::ExecutionPhase phase,
    const strix::kernels::ShapeClass& shape,
    double measured_seconds) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase = phase;
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
    variant.input_layout = "fixture";
    variant.output_layout = "fixture";
    variant.compiler_identity =
        "fixture-compiler";
    variant.measured_seconds =
        measured_seconds;
    return variant;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto verify =
        strix::build_speculative_verify_plan(
            model,
            4);

    strix::ExecutionBindingRequest shape_request;
    shape_request.logical_rows = 4;
    shape_request.request_concurrency = 1;

    strix::kernels::KernelRegistry registry;
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
                static_cast<unsigned>(
                    *family)) +
            "." +
            std::to_string(
                shape.auxiliary_size);
        if (!unique.insert(key).second) {
            continue;
        }

        registry.add(
            make_variant(
                "verify." + key,
                *family,
                strix::kernels::ExecutionPhase::
                    speculative_verify,
                shape,
                2e-6));

        // Deliberately faster decode variant must never satisfy verify query.
        registry.add(
            make_variant(
                "decode." + key,
                *family,
                strix::kernels::ExecutionPhase::
                    decode,
                shape,
                1e-9));
    }

    const auto bindings =
        strix::bind_speculative_verify_plan(
            model,
            verify,
            registry);

    expect(bindings.missing_steps == 0,
           "verify binder resolves every non-runtime step");
    expect(bindings.host_or_runtime_steps == 1,
           "verify binder keeps one runtime position-window step");

    bool all_verify = true;
    for (const auto& binding :
         bindings.bindings) {
        if (binding.host_or_runtime_step) {
            continue;
        }
        if (!binding.kernel_id.has_value() ||
            !binding.kernel_id->starts_with(
                "verify.")) {
            all_verify = false;
            break;
        }
    }
    expect(all_verify,
           "verify binder never substitutes decode-phase kernels");

    auto incomplete_registry =
        registry;
    strix::kernels::KernelRegistry only_decode;
    for (const auto& variant :
         incomplete_registry.variants()) {
        if (variant.phase ==
            strix::kernels::ExecutionPhase::decode) {
            only_decode.add(variant);
        }
    }

    const auto missing =
        strix::bind_speculative_verify_plan(
            model,
            verify,
            only_decode);
    expect(missing.missing_steps > 0 &&
               missing.bound_steps == 0,
           "decode-only registry cannot bind speculative verify plan");

    if (failures == 0) {
        std::cout << "speculative verify binding tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
