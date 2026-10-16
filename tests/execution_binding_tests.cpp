#include "strix/execution_binding.hpp"
#include "strix/shared_expert_prefetch.hpp"

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

    {
        strix::ExecutionBindingRequest scoped;
        scoped.prefetch_q8_shared_expert = true;
        const auto shape = strix::execution_shape_for(model, strix::SerialOp::shared_expert, scoped);
        auto ordinary = make_variant("shared.reference", strix::kernels::OperatorFamily::shared_expert, shape);
        ordinary.qualification = strix::kernels::QualificationStatus::target_qualified;
        ordinary.measured_seconds = 1e-6; ordinary.numerical_contract_identity = 17;
        auto prefetched = ordinary;
        prefetched.id += strix::shared_expert_prefetch::kernel_suffix;
        prefetched.measured_seconds = 2e-6;
        strix::kernels::KernelRegistry scoped_registry;
        scoped_registry.add(ordinary); scoped_registry.add(prefetched);
        strix::kernels::OfflineTuningKey key;
        key.gfx_target = "gfx1151"; key.device_sku = "fixture"; key.compute_units = 40;
        key.compiler_identity = "test"; key.hip_identity = "fixture";
        key.binary_sha256 = std::string(64, 'a'); key.model_content_sha256 = std::string(64, 'b');
        key.compiler_flags_sha256 = std::string(64, 'c');
        key.input_precision = key.output_precision = "bf16"; key.accumulator_precision = "fp32";
        key.weight_format = "q8"; key.input_layout = key.output_layout = "test";
        key.state_layout = "fixture"; key.context_bucket = "1-8192"; key.memory_regime = "registered";
        key.family = ordinary.family; key.shape = shape; key.m = key.n = key.k = 1;
        key.numerical_contract_identity = 17;
        strix::kernels::OfflineTuningRecord baseline;
        baseline.key = key; baseline.kernel_id = ordinary.id;
        baseline.qualification_receipt_sha256 = std::string(64, 'd');
        baseline.median_complete_seconds = 1e-6; baseline.samples = 3;
        baseline.complete_workload = baseline.owners_and_resources_qualified = true;
        auto prefetch_record = baseline;
        prefetch_record.kernel_id = prefetched.id; prefetch_record.median_complete_seconds = 2e-6;
        std::vector<strix::kernels::OfflineTuningKey> keys(plan.steps.size(), key);
        std::vector<strix::kernels::OfflineTuningRecord> records{baseline, prefetch_record};
        strix::kernels::OfflineTuningPlan tuning{keys, records}; scoped.offline_tuning = &tuning;
        const auto result = strix::bind_serial_execution_plan(model, plan, scoped_registry, scoped);
        std::size_t checked = 0;
        for (const auto& binding : result.bindings) if (binding.op == strix::SerialOp::shared_expert) {
            ++checked;
            expect(binding.kernel_id == prefetched.id && binding.tuning_selection_reason ==
                strix::kernels::TuningSelectionReason::selection_policy_mismatch,
                "offline timings must not undo an explicit shared-expert layer policy");
        }
        expect(checked > 0, "scoped tuning fixture covers real shared-expert steps");
    }

    if (failures == 0) {
        std::cout << "execution binding tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
