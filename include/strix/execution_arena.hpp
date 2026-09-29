#pragma once

#include "strix/execution_buffers.hpp"
#include "strix/execution_io.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>
#include <vector>

namespace strix {

struct ExecutionArenaAllocation {
    ExecutionBufferUse use;
    std::uint32_t first_step = 0;
    std::uint32_t last_step = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

struct ExecutionArenaPlan {
    std::uint64_t alignment = 1;
    std::uint64_t total_bytes = 0;
    std::uint64_t naive_nonaliased_bytes = 0;
    std::vector<ExecutionArenaAllocation> allocations;
};

[[nodiscard]] ExecutionArenaPlan build_serial_execution_arena_plan(
    const ModelContract& model,
    const ExecutionBufferRequest& buffer_request,
    const SerialExecutionPlan& execution_plan,
    const SerialExecutionIoPlan& io_plan,
    std::uint64_t alignment = 256);

[[nodiscard]] const ExecutionArenaAllocation* find_arena_allocation(
    const ExecutionArenaPlan& plan,
    ExecutionBufferUse use) noexcept;

}  // namespace strix
