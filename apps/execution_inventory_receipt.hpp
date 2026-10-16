#pragma once

#include "hip_decode_receipt_json.hpp"
#include "strix/launch_dispatch.hpp"

#include <array>
#include <cstdlib>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::decode_receipt {
namespace execution_inventory_detail {
inline const char* tuning_reason(kernels::TuningSelectionReason reason) {
    switch (reason) {
    case kernels::TuningSelectionReason::qualified_record: return "qualified_record";
    case kernels::TuningSelectionReason::invalid_key: return "invalid_key";
    case kernels::TuningSelectionReason::ambiguous_records: return "ambiguous_records";
    case kernels::TuningSelectionReason::no_qualified_record: return "no_qualified_record";
    case kernels::TuningSelectionReason::selection_policy_mismatch: return "selection_policy_mismatch";
    }
    throw std::invalid_argument("inventory tuning selection reason is invalid");
}
inline const char* qualification_name(kernels::QualificationStatus status) {
    switch (status) {
    case kernels::QualificationStatus::experimental: return "experimental";
    case kernels::QualificationStatus::oracle_matched: return "oracle_matched";
    case kernels::QualificationStatus::target_qualified: return "target_qualified";
    case kernels::QualificationStatus::production: return "production";
    case kernels::QualificationStatus::rejected: return "rejected";
    }
    throw std::invalid_argument("inventory variant qualification is invalid");
}

inline void string(std::ostream& out, std::string_view value) {
    out << '"' << escape_json_string(value) << '"';
}
} // namespace execution_inventory_detail

// Explicit decoder-process settings only. Unset/empty strings are distinct;
// environment values do not prove the runtime's effective queue topology or
// observed kernel overlap. This never changes a process setting.
template<class Getter>
inline std::string hip_runtime_environment_receipt(Getter&& get) {
    constexpr std::array names{"GPU_MAX_HW_QUEUES", "HSA_ENABLE_SDMA",
        "HSA_OVERRIDE_GFX_VERSION", "AMD_SERIALIZE_KERNEL", "AMD_SERIALIZE_COPY",
        "HIP_LAUNCH_BLOCKING", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"};
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "  \"hip_runtime_environment\":{\"source\":\"decoder-process\",\"requested\":{";
    bool first = true;
    for (const auto* name : names) {
        if (!first) out << ',';
        first = false;
        execution_inventory_detail::string(out, name);
        out << ':';
        if (const auto* value = get(name)) execution_inventory_detail::string(out, value);
        else out << "null";
    }
    out << "},\"effective_hw_queues\":null,\"concurrent_execution_observed\":null},\n";
    return out.str();
}
inline std::string hip_runtime_environment_receipt() {
    return hip_runtime_environment_receipt([](const char* name) { return std::getenv(name); });
}

// Describes this compiled decode selection, not a general capability catalog or
// native qualification. No launcher is called and no pointer address is emitted.
inline std::string execution_inventory_receipt(
    const CompiledSerialExecution& compiled,
    const CompiledLaunchPlan& launches,
    const kernels::KernelRegistry& registry) {
    using execution_inventory_detail::string;
    const auto fail = [] { throw std::invalid_argument("execution inventory compiled/launcher evidence differs"); };
    const auto count = compiled.execution.steps.size();
    if (!compiled.execution_identity || !compiled.kernel_plan_identity || !launches.plan_identity ||
        launches.execution_identity != compiled.execution_identity || !count ||
        compiled.bindings.bindings.size() != count || launches.steps.size() != count ||
        !compiled.logical_rows || !compiled.request_concurrency || compiled.bindings.missing_steps)
        fail();

    // Reuse the actual launch-plan identity algorithm rather than maintaining a
    // second hash implementation. Repeated keys must resolve consistently.
    ExecutionLauncherRegistry bound_launchers;
    std::vector<const kernels::KernelVariant*> selected;
    std::uint32_t runtime_count = 0, kernel_count = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& step = compiled.execution.steps[index];
        const auto& binding = compiled.bindings.bindings[index];
        const auto& launch = launches.steps[index];
        if (step.id != index || binding.step_id != step.id || binding.op != step.op ||
            launch.step_id != step.id || launch.op != step.op ||
            launch.runtime_step != binding.host_or_runtime_step ||
            !launch.implementation_identity || !launch.function || to_string(step.op) == "unknown")
            fail();
        for (const auto dependency : step.dependencies) if (dependency >= step.id) fail();
        const ExecutionLauncher* previous = nullptr;
        std::string key;
        if (binding.host_or_runtime_step) {
            if (binding.kernel_id || binding.family || binding.tuning_selection_reason || operator_family_for(step.op)) fail();
            previous = bound_launchers.find_runtime_launcher(step.op);
            key = "runtime:" + std::string(to_string(step.op));
            ++runtime_count;
        } else {
            if (!binding.kernel_id || binding.kernel_id->empty() || !binding.family ||
                binding.family != operator_family_for(step.op)) fail();
            const auto* variant = registry.find(*binding.kernel_id);
            if (!variant || !kernels::valid_variant(*variant) || variant->family != *binding.family ||
                variant->phase != kernels::ExecutionPhase::decode ||
                variant->shape.logical_rows != compiled.logical_rows ||
                variant->shape.request_concurrency != compiled.request_concurrency)
                fail();
            selected.push_back(variant);
            previous = bound_launchers.find_kernel_launcher(*binding.kernel_id);
            key = *binding.kernel_id;
            ++kernel_count;
        }
        if (previous) {
            if (previous->function != launch.function || previous->user_data != launch.user_data ||
                previous->implementation_identity != launch.implementation_identity) fail();
        } else {
            ExecutionLauncher launcher{key, launch.implementation_identity, launch.function, launch.user_data};
            if (binding.host_or_runtime_step) bound_launchers.add_runtime_launcher(step.op, launcher);
            else bound_launchers.add_kernel_launcher(launcher);
        }
    }
    if (runtime_count != compiled.bindings.host_or_runtime_steps || kernel_count != compiled.bindings.bound_steps ||
        kernels::plan_identity(std::span<const kernels::KernelVariant* const>(selected.data(), selected.size())) != compiled.kernel_plan_identity ||
        compile_launch_plan(compiled, bound_launchers).plan_identity != launches.plan_identity)
        fail();

    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "  \"execution_inventory\":{\"schema_version\":1,\"evidence_kind\":\"compiled-selected-launchers\","
           "\"phase\":\"decode\",\"selected_logical_rows\":" << compiled.logical_rows
        << ",\"selected_request_concurrency\":" << compiled.request_concurrency
        << ",\"verify_admitted_by_this_receipt\":false,\"prefill_admitted_by_this_receipt\":false,"
           "\"execution_identity\":" << compiled.execution_identity
        << ",\"kernel_plan_identity\":" << compiled.kernel_plan_identity
        << ",\"launch_plan_identity\":" << launches.plan_identity
        << ",\"whole_plan_graph_safe_declared\":" << (compiled.whole_plan_graph_safe ? "true" : "false")
        << ",\"native_execution_observed\":null,\"qualification\":null,\"steps\":[";
    for (std::size_t index = 0; index < count; ++index) {
        const auto& step = compiled.execution.steps[index];
        const auto& binding = compiled.bindings.bindings[index];
        const auto& launch = launches.steps[index];
        if (index) out << ',';
        out << "{\"step_id\":" << step.id << ",\"operator\":";
        string(out, to_string(step.op));
        out << ",\"layer\":";
        if (step.layer) out << *step.layer; else out << "null";
        out << ",\"runtime_step\":" << (binding.host_or_runtime_step ? "true" : "false")
            << ",\"launcher_bound\":true,\"launcher_implementation_identity\":" << launch.implementation_identity
            << ",\"kernel_id\":";
        if (binding.kernel_id) string(out, *binding.kernel_id); else out << "null";
        out << ",\"offline_tuning_selection\":";
        if (binding.tuning_selection_reason)
            string(out, execution_inventory_detail::tuning_reason(*binding.tuning_selection_reason));
        else out << "null";
        out << ",\"declared_variant\":";
        if (binding.host_or_runtime_step) out << "null";
        else {
            const auto& variant = *registry.find(*binding.kernel_id);
            out << "{\"id\":"; string(out, variant.id);
            out << ",\"phase\":\"decode\",\"family\":" << static_cast<unsigned>(variant.family)
                << ",\"logical_rows\":" << variant.shape.logical_rows
                << ",\"request_concurrency\":" << variant.shape.request_concurrency
                << ",\"hidden_size\":" << variant.shape.hidden_size
                << ",\"auxiliary_size\":" << variant.shape.auxiliary_size
                << ",\"wave_size\":" << static_cast<unsigned>(variant.wave)
                << ",\"numerics\":{\"input_precision\":"; string(out, variant.numerics.input_precision);
            out << ",\"accumulator_precision\":"; string(out, variant.numerics.accumulator_precision);
            out << ",\"output_precision\":"; string(out, variant.numerics.output_precision);
            out << ",\"row_invariant\":" << (variant.numerics.row_invariant ? "true" : "false")
                << ",\"deterministic\":" << (variant.numerics.deterministic ? "true" : "false") << '}'
                << ",\"input_layout\":"; string(out, variant.input_layout);
            out << ",\"output_layout\":"; string(out, variant.output_layout);
            out << ",\"graph_safe\":" << (variant.graph_safe ? "true" : "false")
                << ",\"qualification\":";
            string(out, execution_inventory_detail::qualification_name(variant.qualification));
            out << ",\"compiler_identity\":"; string(out, variant.compiler_identity);
            out << ",\"numerical_contract_identity\":";
            if (variant.numerical_contract_identity) out << variant.numerical_contract_identity;
            else out << "null";
            out << '}';
        }
        out << '}';
    }
    out << "]},\n";
    return out.str();
}
} // namespace strix::decode_receipt
