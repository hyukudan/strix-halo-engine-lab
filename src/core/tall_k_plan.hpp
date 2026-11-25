#pragma once

#include "tall_bf16_plan.hpp"
#include "tall_family.hpp"

#include <cstddef>
#include <vector>

namespace strix::core::tall {

using ::strix::cuda::core::tall::Reduction;

struct KPlan {
    std::vector<int> blocks;
    std::vector<int> ends;
};

// The launcher's shape admission supplies whole K32 blocks, whole K128
// chunks for NVFP4 and an even segment length for two-block pipeline units.
// Tables contain K32 indices even when each load consumes two blocks.
inline KPlan k_plan(Reduction reduction, std::uint32_t in, std::uint32_t splits, int blocks_k) {
    KPlan plan;
    if (reduction == Reduction::nvfp4_split) {
        const std::uint32_t chunks = in >> 7;
        for (std::uint32_t segment = 0; segment < splits; ++segment) {
            const auto begin = static_cast<std::uint32_t>(static_cast<std::uint64_t>(segment) * chunks / splits);
            const auto end = static_cast<std::uint32_t>(static_cast<std::uint64_t>(segment + 1) * chunks / splits);
            for (std::uint32_t block = 4 * begin; block < 4 * end; ++block)
                plan.blocks.push_back(static_cast<int>(block));
            plan.ends.push_back(static_cast<int>(plan.blocks.size()));
        }
    } else {
        const std::uint32_t blocks = in >> 5;
        using strix::core::tall::Bf16ExactPlan;
        const std::uint32_t unit = reduction == Reduction::warp_chain ? Bf16ExactPlan::blocks_per_group : 8U;
        strix::core::tall::visit_warp_k_segments(blocks, unit,
            [&](std::uint32_t block) { plan.blocks.push_back(static_cast<int>(block)); },
            [&](std::uint32_t) { plan.ends.push_back(static_cast<int>(plan.blocks.size())); });
    }
    if (blocks_k > 1) {
        const auto step = static_cast<std::size_t>(blocks_k);
        for (std::size_t i = 0; i < plan.blocks.size(); i += step)
            plan.blocks[i / step] = plan.blocks[i];
        plan.blocks.resize(plan.blocks.size() / step);
        for (int& end : plan.ends) end /= blocks_k;
    }
    return plan;
}

}  // namespace strix::core::tall
