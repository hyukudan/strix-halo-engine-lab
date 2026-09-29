#include "strix/kernel_registry.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::kernels::KernelVariant variant(
    std::string id,
    strix::kernels::WaveSize wave,
    double measured_seconds,
    strix::kernels::QualificationStatus qualification =
        strix::kernels::QualificationStatus::target_qualified) {
    strix::kernels::KernelVariant value;
    value.id = std::move(id);
    value.family = strix::kernels::OperatorFamily::rms_norm;
    value.phase = strix::kernels::ExecutionPhase::decode;
    value.shape.logical_rows = 1;
    value.shape.request_concurrency = 1;
    value.shape.hidden_size = 2560;
    value.wave = wave;
    value.resources.workgroup_threads = 256;
    value.numerics.input_precision = "fp32";
    value.numerics.accumulator_precision = "fp32";
    value.numerics.output_precision = "fp32";
    value.numerics.row_invariant = true;
    value.numerics.deterministic = true;
    value.qualification = qualification;
    value.graph_safe = true;
    value.input_layout = "row-major";
    value.output_layout = "row-major";
    value.compiler_identity = "test-compiler";
    value.measured_seconds = measured_seconds;
    return value;
}

}  // namespace

int main() {
    using namespace strix::kernels;

    auto valid = variant("rms.w32", WaveSize::wave32, 2e-6);
    expect(valid_variant(valid), "valid kernel metadata passes");

    auto invalid_threads = valid;
    invalid_threads.resources.workgroup_threads = 48;
    expect(!valid_variant(invalid_threads),
           "workgroup must be an integer number of waves");

    KernelRegistry registry;
    registry.add(valid);
    registry.add(variant("rms.w64", WaveSize::wave64, 1.5e-6));
    registry.add(variant("rms.unmeasured", WaveSize::wave32, 0.0,
                         QualificationStatus::production));

    bool duplicate_rejected = false;
    try {
        registry.add(valid);
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    expect(duplicate_rejected, "duplicate kernel id is rejected");

    KernelQuery query;
    query.family = OperatorFamily::rms_norm;
    query.phase = ExecutionPhase::decode;
    query.shape = valid.shape;
    query.require_graph_safe = true;
    query.require_row_invariant = true;
    query.require_deterministic = true;
    query.minimum_qualification = QualificationStatus::oracle_matched;

    const auto* selected = registry.select(query);
    expect(selected != nullptr && selected->id == "rms.w64",
           "measured fastest qualified variant wins");

    query.required_wave = WaveSize::wave32;
    selected = registry.select(query);
    expect(selected != nullptr && selected->id == "rms.w32",
           "required wave filters selection");

    query.shape.logical_rows = 2;
    expect(registry.select(query) == nullptr,
           "shape mismatch yields no kernel");

    const auto* first = registry.find("rms.w32");
    const auto* second = registry.find("rms.w64");
    const std::vector<const KernelVariant*> plan{first, second};
    const auto identity_a = plan_identity(plan);
    const auto identity_b = plan_identity(plan);
    expect(identity_a != 0 && identity_a == identity_b,
           "plan identity is stable");

    const std::vector<const KernelVariant*> reversed{second, first};
    expect(plan_identity(reversed) != identity_a,
           "plan identity preserves operator order");

    if (failures == 0) {
        std::cout << "kernel registry tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
