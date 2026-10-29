// Host-only scheduling; never changes arithmetic or allocated workspace capacity.
#pragma once
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
inline std::vector<std::uint32_t> plan_blocks(std::uint64_t position,
    const std::vector<std::uint32_t>& blocks, bool tail_large = false, std::uint32_t final_rows = 0) {
    validate_blocks(blocks);
    const auto cap = *std::max_element(blocks.begin(), blocks.end());
    if (final_rows > cap) throw std::invalid_argument("final_block_rows exceeds loaded block capacity");
    std::vector<std::uint32_t> plan;
    for (std::uint64_t at = 0; at < position;) {
        auto n = static_cast<std::uint32_t>(std::min<std::uint64_t>(position - at,
            blocks[std::min(plan.size(), blocks.size()-1)]));
        plan.push_back(n); at += n;
    }
    // Move the final remainder to the front, preserving count, sum and maximum.
    if (tail_large && plan.size() > 1 && plan.back() < plan[plan.size()-2]) std::rotate(plan.begin(), plan.end()-1, plan.end());
    if (final_rows && plan.size() > 1) {
        // Reserve the requested tail, retaining at least one real row in each existing block.
        const auto target = static_cast<std::uint32_t>(std::min<std::uint64_t>(final_rows, position-plan.size()+1));
        if (target > plan.back()) {
            auto need = target-plan.back();
            for (std::size_t i = plan.size()-1; i-- > 0 && need;) {
                auto take = std::min(need, plan[i]-1); plan[i] -= take; plan.back() += take; need -= take;
            }
        } else if (target < plan.back()) {
            auto extra = plan.back()-target;
            for (std::size_t i = 0; i+1 < plan.size() && extra; ++i) {
                auto take = std::min(extra, cap-plan[i]); plan[i] += take; plan.back() -= take; extra -= take;
            }
        }
    }
    return plan;
}
} // namespace strix::cuda::qwen_prefill
