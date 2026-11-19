#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "strix/session/mark_plan.hpp"

namespace strix::cuda::qwen_prefix {

inline void validate_block(std::uint32_t block) {
    if (!block || block % 4) throw std::invalid_argument("--prefix-cache-block must be a positive multiple of 4");
}

inline std::uint64_t lookup_limit(std::uint64_t tokens) { return session::mark_limit(tokens, 2); }

// The shared mark plan (strix::session::plan_marks): the stride grid, the root mark (the repeated system + tools prefix,
// 0: none) and the last block boundary a restore of this prompt may use.
inline std::vector<std::uint64_t> capture_points(std::uint64_t tokens, std::uint32_t block,
                                                std::uint64_t restored, std::uint64_t stride, std::uint64_t root = 0) {
    validate_block(block);
    if (stride && stride % block) throw std::invalid_argument("--prefix-cache-stride must be 0 or a multiple of --prefix-cache-block");
    session::MarkPlanRequest request;
    request.tokens = tokens;
    request.restored = restored;
    request.block = block;
    request.stride = stride;
    request.root = root;
    request.tail = 2;
    return session::plan_marks(request);
}

inline std::uint32_t clip_chunk(std::uint64_t prefilled, std::uint32_t rows,
                                const std::vector<std::uint64_t>& points) {
    const auto next = std::upper_bound(points.begin(), points.end(), prefilled);
    return next == points.end() ? rows : static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, *next - prefilled));
}

}  // namespace strix::cuda::qwen_prefix
