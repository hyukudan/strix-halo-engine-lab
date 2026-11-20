#pragma once
// LDS placement of a WMMA row tile read one row per lane as a 16 x 16 operand (32-B rows = two 16-B halves) and
// filled by row-per-lane 16-B stores: the dense flash core's V^T tile. Pure index arithmetic (host-tested).
//
// LDS model (gfx11 wave32, 32 banks x 4 B): a 16-B access serves 8 lanes per cycle; the lanes of one cycle conflict
// when their 16-B pieces fall in the same 4-bank group (16-B slot mod 8) at different addresses.
//   plain 32-B rows: rows r and r + 4 share a group -> every fragment read (rows R .. R + 7) 2-way (the dense core's
//     29-32 % conflict cycles at Qwen 27B / E4B 8K).
//   slot32 (half h of row r at 16-B slot 2 r + (h ^ ((r / 4) % 2))): fragment reads hit 8 groups, and 4 consecutive
//     rows (both halves) of the row stores stay one permuted 128-B line.
// slot32(r + 16, h) = slot32(r, h) + 32: an unrolled walk over 16-row fragments keeps one per-lane base address per half
// plus immediate offsets (an XOR over a 32-row period cost address registers and scratch on gfx1151).
// Only the placement changes: every lane reads the same bytes, results are bit-identical.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_LDS_ROWS_HD __host__ __device__
#else
#define STRIX_LDS_ROWS_HD
#endif

namespace strix::core::lds_rows {

// 16-B slot of half `half` (0: elements 0..7, 1: elements 8..15) of row `row` in a slot32 tile (2 x rows slots).
STRIX_LDS_ROWS_HD constexpr unsigned slot32(unsigned row, unsigned half) { return 2U * row + (half ^ ((row >> 2U) & 1U)); }

}  // namespace strix::core::lds_rows
