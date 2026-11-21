#pragma once

#include "masks.hpp"

#include <algorithm>

namespace strix::attn_core::dense {

inline constexpr unsigned kKeySplitMaxRows = 16U;
inline constexpr unsigned kKeySplitTargetKeys = 128U;
inline constexpr unsigned kKeySplitMaxChunks = 16U;

// For valid consecutive query positions, use the visible union across the rows.
// Dense flash distributes its 16-key tiles of that union across these chunks.
inline unsigned small_row_key_splits(unsigned first_position, unsigned rows, unsigned window) {
    if (rows == 0U || rows > kKeySplitMaxRows) return 1U;
    const unsigned keys = first_position - mask::window_lo(first_position, window) + rows;
    return std::min(kKeySplitMaxChunks, 1U + (keys - 1U) / kKeySplitTargetKeys);
}

}  // namespace strix::attn_core::dense
