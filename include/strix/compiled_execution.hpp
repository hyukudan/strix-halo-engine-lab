#pragma once

#include "strix/execution_arena.hpp"
#include "strix/execution_binding.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/kernel_registry.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

struct CompiledSerialExecution {
    SerialExecutionPlan execution;
    ExecutionBindingPlan bindings;
    SerialExecutionIoPlan io;
    ExecutionArenaPlan arena;

    std::uint64_t kernel_plan_identity = 0;
    std::uint64_t execution_identity = 0;

    std::uint32_t dynamic_steps = 0;
    std::uint32_t non_graph_safe_steps = 0;
    bool whole_plan_graph_safe = false;

    std::uint32_t logical_rows = 1;
    std::uint32_t request_concurrency = 1;
};

[[nodiscard]] CompiledSerialExecution compile_serial_execution(
    const ModelContract& model,
    const kernels::KernelRegistry& registry,
    const ExecutionBindingRequest& binding_request,
    const ExecutionBufferRequest& buffer_request,
    std::uint64_t arena_alignment = 256);

[[nodiscard]] EagerExecutionResult execute_compiled_serial_eager(
    const CompiledSerialExecution& compiled,
    const EagerExecutionRequest& request,
    EagerExecutionDispatcher& dispatcher);

}  // namespace strix
