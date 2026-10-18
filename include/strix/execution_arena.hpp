#pragma once

#include "strix/execution_buffers.hpp"
#include "strix/execution_io.hpp"
#include "strix/model_contract.hpp"

#include <array>
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

// Two, four or eight known inputs share one physical arena. Each row view retains the exact
// serial payload sizes and uses; only offsets change. Strides are indexed in
// the same order as physical.allocations, including lifetime-reused regions.
struct KnownWindowArena {
    ExecutionArenaPlan physical;
    std::vector<std::uint64_t> row_stride;
    std::vector<ExecutionArenaPlan> row_view;
};

[[nodiscard]] KnownWindowArena build_known_window_arena(
    const ExecutionArenaPlan& serial, std::uint64_t alignment = 256, std::uint32_t logical_rows = 2);
// Separate bounded prefill geometry. Does not widen the closed window API.
[[nodiscard]] KnownWindowArena build_known_prefill_arena(
    const ExecutionArenaPlan& serial, std::uint64_t alignment, std::uint32_t logical_rows);

[[nodiscard]] ExecutionArenaPlan build_serial_execution_arena_plan(
    const ModelContract& model,
    const ExecutionBufferRequest& buffer_request,
    const SerialExecutionPlan& execution_plan,
    const SerialExecutionIoPlan& io_plan,
    std::uint64_t alignment = 256);

[[nodiscard]] const ExecutionArenaAllocation* find_arena_allocation(
    const ExecutionArenaPlan& plan,
    ExecutionBufferUse use) noexcept;

// While one is alive, find_arena_allocation on this thread indexes each plan it is asked about once
// (first allocation of every use, as the linear search returns) and answers later lookups of that
// plan from the index. The owner must keep the plans it looks up alive and unmodified for the
// scope's lifetime (one admitted prefill block's launch: about 2000 lookups per row).
class ArenaLookupScope {
public:
    ArenaLookupScope();
    ~ArenaLookupScope();
    ArenaLookupScope(const ArenaLookupScope&) = delete;
    ArenaLookupScope& operator=(const ArenaLookupScope&) = delete;

private:
    void* previous_;
};

}  // namespace strix
