#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::spec {

// Candidate i is compared with the target selection preceding its input row.
// The first nonmatching selection remains pending, as does the bonus
// selection after a fully matching window.
inline std::size_t greedy_matching_prefix(std::span<const std::uint32_t> candidates,
                                         std::span<const std::uint32_t> target) {
    const auto mismatch = std::mismatch(candidates.begin(), candidates.end(), target.begin(), target.end());
    return static_cast<std::size_t>(mismatch.first - candidates.begin());
}

}  // namespace strix::spec
