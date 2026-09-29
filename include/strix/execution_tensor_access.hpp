#pragma once

#include "strix/execution_plan.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_tensor_index.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

enum class TensorRequirementCardinality {
    exactly_one,
    one_or_more,
};

struct ExecutionTensorRequirement {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::uint32_t binding_slot = 0;
    ModelTensorSelector selector;
    TensorRequirementCardinality cardinality =
        TensorRequirementCardinality::exactly_one;
};

struct ExecutionTensorAccess {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::uint32_t binding_slot = 0;
    std::vector<std::uint32_t> tensor_indices;
};

struct ExecutionTensorAccessPlan {
    std::uint64_t manifest_layout_identity = 0;
    std::uint64_t plan_identity = 0;
    std::vector<ExecutionTensorAccess> accesses;
};

[[nodiscard]] std::vector<ExecutionTensorRequirement>
build_serial_execution_tensor_requirements(
    const SerialExecutionPlan& execution_plan);

[[nodiscard]] ExecutionTensorAccessPlan
compile_execution_tensor_access_plan(
    const SerialExecutionPlan& execution_plan,
    const ModelManifest& manifest,
    const ModelTensorIndex& tensor_index,
    std::span<const ExecutionTensorRequirement> requirements);

[[nodiscard]] const ExecutionTensorAccess*
find_execution_tensor_access(
    const ExecutionTensorAccessPlan& plan,
    std::uint32_t step_id,
    std::uint32_t binding_slot) noexcept;

}  // namespace strix
