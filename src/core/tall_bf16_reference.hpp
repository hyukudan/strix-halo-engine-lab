#pragma once

#include "tall_bf16_plan.hpp"

namespace strix::core::tall {

// Reference for the accumulation schedule of one output. The caller supplies
// the K16 operation, including its instruction-specific rounding. Planes are
// indexed in storage order (high, middle, low). The callback receives them in
// accumulation order. Input K is a positive multiple of block_k.
template <class Mma>
float bf16_exact_reference(std::uint32_t input_width, Mma mma) {
    using Plan = Bf16ExactPlan;
    float result = 0.0F;
    float acc = 0.0F;
    visit_warp_k_segments(input_width / Plan::block_k, Plan::blocks_per_group,
        [&](std::uint32_t block) {
            for (std::uint32_t step = 0; step < Plan::block_k; step += Plan::step_k) {
                const auto k = block * Plan::block_k + step;
                for (int plane = Plan::planes - 1; plane >= 0; --plane) acc = mma(acc, k, plane);
            }
        },
        [&](std::uint32_t segment) {
            result = segment == 0 ? acc : result + acc;
            acc = 0.0F;
        });
    return result;
}

}  // namespace strix::core::tall
