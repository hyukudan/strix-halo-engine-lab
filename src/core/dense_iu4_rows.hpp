#pragma once
// Row-block plan of the EXACT IU4 dense core over wide prefill windows (core/dense_linear_iu4.hpp run(), rows > 16).
// Host and device (constexpr): the kernel and the host test share it.
// One block of the wide grid = kWideRows rows of one 64-column block: the rows that share one weight decode (the
// core's row capacity per pass: 8 rows of activation words in SGPRs, w4a16.hpp UniformX). Blocks of the same column
// block are adjacent in dispatch order (row chunk = blockIdx.x) so they read its weight slice while it is cached.
// A block computes all kWideRows rows (pad rows read from the caller's FP16 scratch, never stored): the caller's
// activation scratch holds xh_rows(rows) rows.
#include <cstddef>

namespace strix::core::dense_iu4_rows {

inline constexpr unsigned kMaxRows = 16U;   // decode / verify buckets (1 / 2 / 4 / 8 / 16), one 1-D launch
inline constexpr unsigned kWideRows = 8U;   // rows per block of the wide grid

struct Block {
    unsigned row0;  // first window row of the block
    unsigned rows;  // rows stored (1..kWideRows)
};
constexpr unsigned blocks(unsigned rows) { return (rows + kWideRows - 1U) / kWideRows; }
constexpr Block block(unsigned index, unsigned rows) {
    const unsigned row0 = index * kWideRows;
    return Block{row0, rows - row0 < kWideRows ? rows - row0 : kWideRows};
}
constexpr std::size_t xh_rows(std::size_t rows) { return (rows + kWideRows - 1U) / kWideRows * kWideRows; }
constexpr bool wide(unsigned rows) { return rows > kMaxRows; }

}  // namespace strix::core::dense_iu4_rows
