#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::spec {

// Entry n-1 caps depth at n live lanes. The last entry also covers wider
// cohorts. An empty table leaves the existing depth limit in charge.
[[nodiscard]] inline std::uint32_t draft_depth_for_lanes(
    std::span<const std::uint32_t> limits, std::uint32_t lanes, std::uint32_t cap) noexcept {
    if (limits.empty()) return cap;
    const std::size_t index = lanes == 0U ? 0U : std::min<std::size_t>(lanes - 1U, limits.size() - 1U);
    return std::min(cap, limits[index]);
}

}  // namespace strix::spec
