#pragma once
// Rows form of the head-family dense kernel (src/hip_lm_head_launcher.hip): one block computes kOut outputs x kRows rows.
// Every (row, output) keeps the single-row reduction of lm_head_kernel exactly: kLanes per-lane chains over the features
// lane + kLanes * k in k order, then the pairwise tree at distances kLanes / 2 .. 1. The tile only decides which block
// computes a value and how many values share one weight / activation load; it never changes the bits.
// Host-visible so the tile map and the tree index map are tested on the CPU (tests/dense_rows_tile_tests.cpp).

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_DRT_HD __host__ __device__
#else
#define STRIX_DRT_HD
#endif

namespace strix::dense_rows_tile {

inline constexpr std::uint32_t kLanes = 256;       // == the single-row kernel's block size (lm_head kThreads)
inline constexpr std::uint32_t kLaneShift = 8;     // log2(kLanes)
inline constexpr std::uint32_t kOut = 8;           // outputs per block (weight rows sharing one activation load)
inline constexpr std::uint32_t kRows = 8;          // rows per block (activation rows sharing one weight load)
inline constexpr std::uint32_t kValues = kOut * kRows;
inline constexpr std::uint32_t kTreeValues = 32;   // values reduced per LDS pass: 32 x 256 x 4 B = 32 KiB
static_assert((kLanes >> kLaneShift) == 1U && (kLanes & (kLanes - 1U)) == 0U);
static_assert(kValues % kTreeValues == 0U);
static_assert(kTreeValues <= kLanes);

struct Grid {
    std::uint32_t x = 0, y = 0;  // x = output tiles, y = row tiles
};
STRIX_DRT_HD constexpr Grid grid(std::uint32_t rows, std::uint32_t outputs) {
    return {(outputs + kOut - 1U) / kOut, (rows + kRows - 1U) / kRows};
}

// Tile value v = o * kRows + r of block (bx, by).
STRIX_DRT_HD constexpr std::uint32_t value_output(std::uint32_t bx, std::uint32_t v) { return bx * kOut + v / kRows; }
STRIX_DRT_HD constexpr std::uint32_t value_row(std::uint32_t by, std::uint32_t v) { return by * kRows + v % kRows; }
// Load index: out-of-range outputs / rows read the last valid one (in bounds); their results are never stored.
STRIX_DRT_HD constexpr std::uint32_t clamp_index(std::uint32_t i, std::uint32_t n) { return i < n ? i : n - 1U; }

// LDS tree pass over kTreeValues values stored as scratch[value * kLanes + lane]. At distance 2^shift, work item
// i in [0, kTreeValues << shift) is (value = i >> shift, lane = i & (2^shift - 1)) and performs scratch[dst] += scratch[src],
// the single-row step `scratch[lane] += scratch[lane + distance]` of that value.
struct TreeItem {
    std::uint32_t dst = 0, src = 0;
};
STRIX_DRT_HD constexpr std::uint32_t tree_items(std::uint32_t shift) { return kTreeValues << shift; }
STRIX_DRT_HD constexpr TreeItem tree_item(std::uint32_t item, std::uint32_t shift) {
    const std::uint32_t value = item >> shift, lane = item & ((1U << shift) - 1U);
    return {(value << kLaneShift) + lane, (value << kLaneShift) + lane + (1U << shift)};
}

}  // namespace strix::dense_rows_tile
