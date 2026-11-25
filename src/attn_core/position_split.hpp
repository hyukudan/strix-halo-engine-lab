#pragma once

#include "masks.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_POSITION_SPLIT_HD __host__ __device__
#else
#define STRIX_POSITION_SPLIT_HD
#endif

namespace strix::attn_core {

// Logical position determines the chunk. Window clipping and physical cache
// placement leave the chunk origin, including its denominator lane order, fixed.
inline constexpr unsigned position_split_keys = 256;

[[nodiscard]] STRIX_POSITION_SPLIT_HD constexpr unsigned position_split(unsigned key) {
    return key / position_split_keys;
}

[[nodiscard]] STRIX_POSITION_SPLIT_HD constexpr unsigned position_split_begin(unsigned split) {
    return split * position_split_keys;
}

struct PositionSplitPlan {
    unsigned context;
    unsigned window;

    [[nodiscard]] STRIX_POSITION_SPLIT_HD constexpr unsigned split_count() const {
        return context == 0 ? 0 : position_split(context - 1U) + 1U;
    }

    [[nodiscard]] STRIX_POSITION_SPLIT_HD unsigned first_key(unsigned position) const {
        return mask::window_lo(position, window);
    }

    [[nodiscard]] STRIX_POSITION_SPLIT_HD mask::Range row_keys(unsigned position) const {
        return position < context ? mask::Range{first_key(position), position + 1U} : mask::Range{0, 0};
    }
};

}  // namespace strix::attn_core

#undef STRIX_POSITION_SPLIT_HD
