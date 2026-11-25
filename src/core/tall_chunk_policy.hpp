#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

namespace strix::core::tall {

struct ChunkClass {
    std::uint32_t rows;
    std::uint64_t scratch_bytes;  // Peak incremental bytes for this full class, including padding.
};

struct ChunkPolicy {
    std::uint32_t target_rows_per_expert;
    std::uint32_t active_experts;  // Caller estimate for the selected workload/regime.
    std::uint32_t routes_per_row;
    std::uint64_t scratch_budget_bytes;  // Available after persistent allocations and reserves.
};

struct ChunkPlan {
    std::uint32_t rows = 0;      // Rows to consume now.
    std::uint32_t capacity = 0;  // Program class. Rows in [rows, capacity) are masked.
};

// Pick the smallest affordable class that reaches the estimated occupancy
// target or contains the remaining prompt. If none reaches the target, use the
// largest affordable class. Resolve again with the remainder for a masked tail.
// An empty result means no rows remain, invalid policy, or no class fits.
[[nodiscard]] constexpr ChunkPlan resolve_chunk(std::uint64_t remaining, ChunkPolicy policy,
                                                 std::span<const ChunkClass> classes) {
    if (remaining == 0 || policy.target_rows_per_expert == 0 || policy.routes_per_row == 0 ||
        policy.active_experts < policy.routes_per_row) return {};

    const auto assignments = std::uint64_t{policy.target_rows_per_expert} * policy.active_experts;
    const auto target = assignments / policy.routes_per_row + (assignments % policy.routes_per_row != 0);
    const auto wanted = std::min(remaining, target);
    std::uint32_t adequate = 0;
    std::uint32_t largest = 0;
    for (const auto& choice : classes) {
        if (choice.scratch_bytes > policy.scratch_budget_bytes) continue;
        largest = std::max(largest, choice.rows);
        if (choice.rows >= wanted && (adequate == 0 || choice.rows < adequate)) adequate = choice.rows;
    }
    const auto capacity = adequate == 0 ? largest : adequate;
    return {static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, capacity)), capacity};
}

}  // namespace strix::core::tall
