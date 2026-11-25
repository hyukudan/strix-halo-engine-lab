#pragma once

#include <algorithm>
#include <cstdint>

namespace strix::core::tall {

struct Bf16ExactPlan {
    static constexpr int rows = 64;
    static constexpr int outputs = 64;
    static constexpr int block_k = 32;
    static constexpr int step_k = 16;
    static constexpr int planes = 3;
    static constexpr std::uint32_t blocks_per_group = 4;
};

// Groups cycle through the incumbent's eight K segments. Inside a segment,
// whole K32 blocks are visited in increasing order, including a short group.
// Finish every segment, even when it contributed no blocks. Callers supply a
// positive group size and a block count derived from a 32-bit K width.
template <class Block, class Finish>
constexpr void visit_warp_k_segments(std::uint32_t blocks, std::uint32_t blocks_per_group,
                                     Block visit, Finish finish) {
    const auto groups = blocks / blocks_per_group + (blocks % blocks_per_group != 0);
    constexpr std::uint32_t segments = 8;
    for (std::uint32_t segment = 0; segment < segments; ++segment) {
        for (auto group = segment; group < groups; group += segments) {
            const auto first = group * blocks_per_group;
            const auto count = std::min(blocks - first, blocks_per_group);
            for (std::uint32_t offset = 0; offset < count; ++offset) visit(first + offset);
        }
        finish(segment);
    }
}

}  // namespace strix::core::tall
