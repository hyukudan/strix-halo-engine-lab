#pragma once

#include "strix/execution_arena.hpp"
#include "strix/execution_binding.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/execution_io.hpp"
#include "strix/kernel_registry.hpp"
#include "strix/graph_eligibility.hpp"
#include "strix/model_contract.hpp"
#include "strix/speculative_verify_binding.hpp"
#include "strix/speculative_verify_plan.hpp"

#include <cstdint>

namespace strix {

struct CompiledSpeculativeVerifyExecution {
    SerialExecutionPlan topology;
    SpeculativeVerifyPlan verify;
    ExecutionBindingPlan bindings;
    SerialExecutionIoPlan io;
    ExecutionArenaPlan arena;

    std::uint64_t kernel_plan_identity = 0;
    std::uint64_t execution_identity = 0;

    std::uint64_t vocab_size = 0;
    std::uint32_t candidate_depth = 0;
    std::uint32_t request_concurrency = 1;
    std::uint32_t sequential_window_steps = 0;
    std::uint32_t causal_window_steps = 0;
    std::uint32_t dynamic_steps = 0;
    bool whole_plan_graph_safe = false;
};

[[nodiscard]] CompiledSpeculativeVerifyExecution
compile_speculative_verify_execution(
    const ModelContract& model,
    std::uint32_t candidate_depth,
    const kernels::KernelRegistry& registry,
    const SpeculativeVerifyBindingRequest& binding_request = {},
    const ExecutionBufferRequest& buffer_request = {},
    std::uint64_t arena_alignment = 256);

[[nodiscard]] GraphCapturePlan prepare_compiled_verify_graph_capture(
    const CompiledSpeculativeVerifyExecution& compiled, const ExecutionRuntimeResources& resources,
    const GraphDynamicArguments& arguments, const GraphMemoryRequest& memory);

[[nodiscard]] GraphEligibility assess_compiled_verify_graph_replay(
    const CompiledSpeculativeVerifyExecution& compiled, const GraphCapturePlan& capture,
    const ExecutionRuntimeResources& resources, const GraphDynamicArguments& arguments,
    std::uint64_t position);

}  // namespace strix
