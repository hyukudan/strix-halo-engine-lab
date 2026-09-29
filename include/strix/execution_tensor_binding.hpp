#pragma once

#include "strix/execution_plan.hpp"
#include "strix/model_manifest.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strix {

enum class TensorBindingResolution {
    resolved,
    ambiguous,
    missing,
    not_required,
};

struct StepTensorBinding {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::optional<std::uint32_t> layer;
    TensorBindingResolution resolution =
        TensorBindingResolution::not_required;
    std::vector<std::uint32_t> tensor_indices;
    std::string reason;
};

struct ExecutionTensorBindingAudit {
    std::uint64_t manifest_layout_identity = 0;
    std::uint64_t binding_identity = 0;
    std::uint32_t resolved_steps = 0;
    std::uint32_t ambiguous_steps = 0;
    std::uint32_t missing_steps = 0;
    std::uint32_t not_required_steps = 0;
    std::vector<StepTensorBinding> steps;

    [[nodiscard]] bool complete() const noexcept {
        return ambiguous_steps == 0 && missing_steps == 0;
    }
};

[[nodiscard]] ExecutionTensorBindingAudit audit_execution_tensor_bindings(
    const ModelManifest& manifest,
    const SerialExecutionPlan& plan);

[[nodiscard]] const StepTensorBinding* find_step_tensor_binding(
    const ExecutionTensorBindingAudit& audit,
    std::uint32_t step_id) noexcept;

}  // namespace strix
