#include "strix/kernel_registry.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
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

    auto unmeasured_qualified = valid;
    unmeasured_qualified.measured_seconds = 0.0;
    expect(!valid_variant(unmeasured_qualified),
           "target-qualified kernel requires a measured cost");

    KernelRegistry registry;
    registry.add(valid);
    registry.add(variant("rms.w64", WaveSize::wave64, 1.5e-6));
    registry.add(variant("rms.oracle-only", WaveSize::wave32, 0.0,
                         QualificationStatus::oracle_matched));

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

    KernelRegistry same_kernel_new_measurement;
    auto remeasured = variant("rms.w32", WaveSize::wave32, 9e-6);
    same_kernel_new_measurement.add(remeasured);
    const std::vector<const KernelVariant*> original_plan{registry.find("rms.w32")};
    const std::vector<const KernelVariant*> remeasured_plan{
        same_kernel_new_measurement.find("rms.w32")};
    expect(plan_identity(original_plan) == plan_identity(remeasured_plan),
           "measurement noise does not alter plan identity");

    KernelRegistry offline_registry;
    auto reference = variant("rms.reference", WaveSize::wave32, 2e-6);
    auto candidate = variant("rms.candidate", WaveSize::wave32, 3e-6);
    reference.numerical_contract_identity = candidate.numerical_contract_identity = 0x101U;
    offline_registry.add(reference); offline_registry.add(candidate);
    KernelQuery offline_query;
    offline_query.shape = reference.shape;
    OfflineTuningKey key;
    key.gfx_target = "gfx1151"; key.device_sku = "fixture-sku"; key.compute_units = 40;
    key.compiler_identity = "test-compiler"; key.hip_identity = "fixture-runtime";
    key.binary_sha256 = std::string(64, 'a'); key.model_content_sha256 = std::string(64, 'b');
    key.compiler_flags_sha256 = std::string(64, 'c');
    key.input_precision = key.accumulator_precision = key.output_precision = "fp32";
    key.weight_format = "f32"; key.input_layout = key.output_layout = "row-major";
    key.state_layout = "fixture-state-v1"; key.context_bucket = "1-8192"; key.memory_regime = "weight-sweep";
    key.shape = reference.shape; key.m = key.n = 1; key.k = 2560;
    key.numerical_contract_identity = 0x101U;
    OfflineTuningRecord record;
    record.key = key; record.kernel_id = candidate.id; record.qualification_receipt_sha256 = std::string(64, 'd');
    record.median_complete_seconds = 9e-6; record.samples = 3;
    record.complete_workload = record.owners_and_resources_qualified = true;
    auto baseline_record = record;
    baseline_record.kernel_id = reference.id;
    baseline_record.median_complete_seconds = 10e-6;
    std::vector<OfflineTuningRecord> records{record, baseline_record};
    auto tuned = offline_registry.select_tuned(offline_query, key, records);
    expect(tuned.variant && tuned.variant->id == candidate.id && tuned.reason == TuningSelectionReason::qualified_record,
        "qualified complete-workload cost overrides an isolated measured cost");
    for (unsigned mismatch = 0; mismatch < 8U; ++mismatch) {
        records = {record, baseline_record};
        auto& stale = records.front();
        if (mismatch == 0U) stale.key.binary_sha256[0] = 'e';
        if (mismatch == 1U) stale.key.hip_identity += ".other";
        if (mismatch == 2U) stale.key.weight_format = "q8";
        if (mismatch == 3U) stale.key.context_bucket = "8193-32768";
        if (mismatch == 4U) stale.key.numerical_contract_identity ^= 1U;
        if (mismatch == 5U) stale.complete_workload = false;
        if (mismatch == 6U) stale.owners_and_resources_qualified = false;
        if (mismatch == 7U) stale.median_complete_seconds = std::numeric_limits<double>::quiet_NaN();
        tuned = offline_registry.select_tuned(offline_query, key, records);
        expect(tuned.variant && tuned.variant->id == reference.id && tuned.reason == TuningSelectionReason::qualified_record,
            "stale, incomplete or invalid evidence preserves the original selector");
    }
    records = {record};
    tuned = offline_registry.select_tuned(offline_query, key, records);
    expect(tuned.reason == TuningSelectionReason::no_qualified_record && tuned.variant->id == reference.id,
        "a candidate cannot replace a baseline without comparable complete baseline cost");
    records = {record, record, baseline_record};
    records[1].median_complete_seconds = 1e-9;
    tuned = offline_registry.select_tuned(offline_query, key, records);
    expect(tuned.reason == TuningSelectionReason::ambiguous_records && tuned.variant->id == reference.id,
        "duplicate timing records require offline reconciliation instead of fastest-sample selection");
    KernelRegistry different_contract;
    candidate.numerical_contract_identity = 0x202U;
    different_contract.add(reference); different_contract.add(candidate);
    records = {record, baseline_record};
    tuned = different_contract.select_tuned(offline_query, key, records);
    expect(tuned.reason == TuningSelectionReason::qualified_record && tuned.variant->id == reference.id,
        "matching dtype cannot authorize a different numerical contract");

    if (failures == 0) {
        std::cout << "kernel registry tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
