#pragma once

#include "inkling2_mini_config.hpp"

#include <array>
#include <cstddef>

namespace strix::inkling2 {

// Position-major [position][tap][hidden], shared by the host and resident paths.
struct MiniDSparkTapPlan {
    static constexpr std::array<unsigned, 4> visits{5U, 23U, 41U, 48U};
    static constexpr auto count = static_cast<unsigned>(visits.size());
    static constexpr unsigned width = count * kHiddenWidth;

    static constexpr unsigned index(unsigned visit) {
        for (unsigned i = 0; i < count; ++i) {
            if (visits[i] == visit) return i;
        }
        return count;
    }

    static constexpr std::size_t elements(unsigned positions) { return std::size_t{positions} * width; }
};

}  // namespace strix::inkling2
