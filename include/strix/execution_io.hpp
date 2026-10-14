#pragma once

#include "strix/execution_plan.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

enum class ExecutionBufferClass {
    expanded_hidden,
    hidden,
    injection_weights,
    route_indices,
    route_weights,
    logits,
    mlp_intermediate,
    gdn_workspace,
    ple_workspace,
    indexed_attention_workspace,
    hyper_workspace,
};

struct ExecutionBufferRef {
    ExecutionBufferClass buffer_class =
        ExecutionBufferClass::hidden;
    std::uint32_t slot = 0;

    friend constexpr bool operator==(
        const ExecutionBufferRef&,
        const ExecutionBufferRef&) = default;
};

struct ExecutionBufferUse {
    ExecutionBufferRef buffer;
    std::uint64_t generation = 0;

    friend constexpr bool operator==(
        const ExecutionBufferUse&,
        const ExecutionBufferUse&) = default;
};

struct ExecutionStepIo {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::vector<ExecutionBufferUse> reads;
    std::vector<ExecutionBufferUse> writes;
    std::vector<ExecutionBufferUse> temporaries;
};

struct SerialExecutionIoPlan {
    std::vector<ExecutionStepIo> steps;
    std::uint32_t expanded_hidden_slots = 0;
    std::uint32_t hidden_slots = 0;
    std::uint32_t injection_slots = 0;
    std::uint32_t route_index_slots = 0;
    std::uint32_t route_weight_slots = 0;
    std::uint32_t logits_slots = 0;
    std::uint32_t mlp_intermediate_slots = 0;
    std::uint32_t gdn_workspace_slots = 0;
    std::uint32_t ple_workspace_slots = 0;
    std::uint32_t indexed_attention_workspace_slots = 0;
    std::uint32_t hyper_workspace_slots = 0;
};

[[nodiscard]] SerialExecutionIoPlan build_serial_execution_io_plan(
    const SerialExecutionPlan& execution_plan);

void validate_serial_execution_io_plan(
    const SerialExecutionPlan& execution_plan,
    const SerialExecutionIoPlan& io_plan);

}  // namespace strix
