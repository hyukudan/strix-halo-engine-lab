// Host-only scheduling; never changes arithmetic or allocated workspace capacity.
#pragma once
#include "strix/prefill_piece_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strix::cuda::qwen_prefill {
// Some row-independent kernels use padded rows in CUDA grid y (limit 65535).
inline constexpr std::uint32_t kMaxBlockRows = 65520;
inline void validate_blocks(const std::vector<std::uint32_t>& blocks) {
    if (blocks.empty()) throw std::invalid_argument("empty block schedule");
    for (auto n : blocks) if (!n || n > kMaxBlockRows)
        throw std::invalid_argument("block rows must be in [1,65520]");
}
// Block plan on the shared prompt piece planner (strix/prefill_piece_plan.hpp): greedy over the schedule, then the
// tail_first and reserve_final traits. Only the block limits above are this engine's own.
inline std::vector<std::uint32_t> plan_blocks(std::uint64_t position,
    const std::vector<std::uint32_t>& blocks, bool tail_large = false, std::uint32_t final_rows = 0) {
    validate_blocks(blocks);
    const auto cap = *std::max_element(blocks.begin(), blocks.end());
    if (final_rows > cap) throw std::invalid_argument("final_block_rows exceeds loaded block capacity");
    auto plan = ::strix::prefill::greedy_pieces(position, blocks);
    if (tail_large) ::strix::prefill::tail_first(plan);
    ::strix::prefill::reserve_final(plan, position, final_rows, cap);
    return plan;
}
} // namespace strix::cuda::qwen_prefill
