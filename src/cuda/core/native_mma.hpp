// Shared contract of the tensor-core GSQ expert route (arithmetic id gsq-iq-bf16mma-cuda-v1, FAST class), host and device.
//
// This is a separate, explicitly selected arithmetic. It never replaces cuda-native-iq-fp32@1 (native_reuse.hpp), which
// stays the default. Everything the kernels in src/cuda/core/moe_a8.cuh (native_mma_project_kernel / native_mma_down_kernel)
// need to agree on with the CPU oracle lives here as executable code: the fragment geometry, the logical-k placement,
// the exact code -> BF16 conversion and the FP32 -> BF16 hi/lo split.
//
// Operands. A GSQ weight value is scale(group[, half]) * code with an integer code |code| <= 127 (GsqDecode<T>::code) and
// an FP32 group scale (GsqDecode<T>::scales; IQ2_S has one scale per 16-value half). The code is exact in BF16. The
// project activations are the BF16 rows that the exact route reads (no quantization). The down activation h (FP32) is
// carried as hi = bf16_rn(h), lo = bf16_rn(h - hi), so |h - (hi + lo)| <= 2^-17 |h| for finite normal h.
//
// Per (assignment, output) and 32-value group g, MMA half k16 h (logical k = 16 h .. 16 h + 15):
//   half_sum(h) = sum over the 16 k of code * x          (one m16n8k16 BF16 MMA, products exact, FP32 accumulate)
//   one-scale types : acc = fma(half_sum(0) + half_sum(1) [chained in the MMA accumulator], scale, acc)
//   two-scale types : acc = fma(half_sum(1), scale_b, fma(half_sum(0), scale_a, acc))
//   down            : each half_sum is the chained MMA of hi then lo against the same code fragment.
// Groups run in ascending order from acc = +0. The sum order inside one MMA is the hardware's; this is the only
// arithmetic freedom of the contract besides the hi/lo truncation, and why it is FAST and not bitwise.
#pragma once

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define STRIX_NATIVE_MMA_HD __host__ __device__ __forceinline__
#else
#define STRIX_NATIVE_MMA_HD inline
#endif

namespace strix::cuda::core::native_mma {

inline constexpr const char* arithmetic_id = "gsq-iq-bf16mma-cuda-v1";

// Fragment geometry of mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (PTX ISA), g = lane / 4, q = lane % 4.
// A register r (0..3) holds two BF16 of one row: row g + 8 (r & 1), physical k 2 q + 8 (r >> 1) + e (e = 0 low half).
// B register r (0..1) holds two BF16 of column g: physical k 2 q + 8 r + e.
// C element i (0..3): row g + 8 (i >> 1), column 2 q + (i & 1).
STRIX_NATIVE_MMA_HD unsigned a_row(unsigned lane, unsigned r) { return lane / 4U + 8U * (r & 1U); }
STRIX_NATIVE_MMA_HD unsigned a_slot(unsigned lane, unsigned r, unsigned e) { return 2U * (lane % 4U) + 8U * (r >> 1U) + e; }
STRIX_NATIVE_MMA_HD unsigned b_col(unsigned lane) { return lane / 4U; }
STRIX_NATIVE_MMA_HD unsigned b_slot(unsigned lane, unsigned r, unsigned e) { return 2U * (lane % 4U) + 8U * r + e; }
STRIX_NATIVE_MMA_HD unsigned c_row(unsigned lane, unsigned i) { return lane / 4U + 8U * (i >> 1U); }
STRIX_NATIVE_MMA_HD unsigned c_col(unsigned lane, unsigned i) { return 2U * (lane % 4U) + (i & 1U); }

// Logical k (0..31 inside the group) of physical slot p (0..15) of half h. Lane q's slots {2q, 2q+1, 2q+8, 2q+9} carry
// the four consecutive logical values 16 h + 4 q .. + 3. A is then one 8-byte load per row, and B is the packed code
// word 4 q + 16 h .. + 3 (GsqDecode<T>::word(row, g, q + 4 h): byte j = code(4 index + j)).
STRIX_NATIVE_MMA_HD unsigned logical_k(unsigned h, unsigned p) {
    return 16U * h + 4U * ((p & 7U) >> 1U) + 2U * (p >> 3U) + (p & 1U);
}
// Word index that lane q decodes for half h.
STRIX_NATIVE_MMA_HD unsigned word_index(unsigned lane, unsigned h) { return lane % 4U + 4U * h; }
// A registers from the two 8-byte loads of rows g and g + 8 (logical k 4 q .. 4 q + 3 of the half, BF16 pairs low first).
STRIX_NATIVE_MMA_HD void a_regs(std::uint32_t r0x, std::uint32_t r0y, std::uint32_t r1x, std::uint32_t r1y, std::uint32_t (&a)[4]) {
    a[0] = r0x; a[1] = r1x; a[2] = r0y; a[3] = r1y;
}
// The lane whose B column is the C column of element i of this lane (its weight row owns the scale).
STRIX_NATIVE_MMA_HD unsigned scale_lane(unsigned lane, unsigned i) { return 8U * (lane % 4U) + 4U * (i & 1U); }

// Block geometry (kernels and oracle): 8 warps; a tile has <= 64 rows = 4 m16 blocks.
inline constexpr unsigned kThreads = 256U, kProjectOutputs = 64U, kDownOutputs = 128U, kProjectK = 128U, kDownK = 64U;
STRIX_NATIVE_MMA_HD unsigned m_blocks(unsigned count) { return (count + 15U) >> 4U; }
// Decode-tile form of the tensor-core expert kernels. A tile holds one expert's rows of the step and a row routes to an
// expert at most once (top-k experts are distinct), so a step of rows <= 16 gives every tile <= 16 rows = one m16
// block. Those launches bind kDecodeMBlocks accumulators (8 floats instead of 32) and spend the freed registers on a
// kDecodeAhead-deep ring of weight fields: the loads of the next kDecodeAhead groups are in flight while a group is
// decoded and multiplied (the tile form keeps a one-group lookahead). Each output's MMA and scale-FMA sequence over the
// groups is unchanged, so both forms are bitwise the same arithmetic; only the load schedule and the tile capacity differ.
inline constexpr unsigned kTileMBlocks = 4U, kDecodeMBlocks = 1U, kTileAhead = 1U, kDecodeAhead = 3U;
STRIX_NATIVE_MMA_HD bool decode_form(unsigned rows) { return rows <= 16U * kDecodeMBlocks; }
// Wide-tile form (lever row expert_wide_tile). A tile holds up to kWideRows = 128 rows of one expert (kWideMBlocks m16
// blocks) in a block of kWideThreads = 512 threads (16 warps, one block per SM: the warps per SM of two tile-form blocks).
// Warp w owns column block w % 8 and ONE weight row per lane (project: matrix w / 8, gate or up; down: n8 block w / 8),
// so it keeps 8 x 4 = 32 accumulators like the tile form, and every weight code is still decoded once per tile: a run of
// 128 rows of an expert is decoded once instead of twice. CSR tiles are the expert's runs cut at multiples of the tile
// rows, so a wide tile is exactly two consecutive 64-row tiles and every row keeps its m16 block position (64 and 128 are
// multiples of 16): each output's MMA and scale-FMA sequence is unchanged, bitwise the tile form.
inline constexpr unsigned kWideMBlocks = 8U, kWideThreads = 512U, kWideRows = 16U * kWideMBlocks, kTileRows = 64U;
STRIX_NATIVE_MMA_HD constexpr bool wide(unsigned mb) { return mb > kTileMBlocks; }
STRIX_NATIVE_MMA_HD constexpr unsigned block_threads(unsigned mb) { return wide(mb) ? kWideThreads : kThreads; }
STRIX_NATIVE_MMA_HD constexpr unsigned min_blocks(unsigned mb) { return wide(mb) ? 1U : 2U; }
STRIX_NATIVE_MMA_HD constexpr unsigned tile_capacity(unsigned mb) { return wide(mb) ? kWideRows : kTileRows; }
// Warp map: column block of warp w (project: 8 outputs of project_output; down: 16 of down_output), the weight rows per lane
// (parts: gate + up / two n8 blocks in the tile forms, one in the wide form) and the first part the warp owns.
STRIX_NATIVE_MMA_HD constexpr unsigned column_block(unsigned warp) { return warp % 8U; }
STRIX_NATIVE_MMA_HD constexpr unsigned warp_parts(unsigned mb) { return wide(mb) ? 1U : 2U; }
STRIX_NATIVE_MMA_HD constexpr unsigned first_part(unsigned mb, unsigned warp) { return wide(mb) ? warp / 8U : 0U; }
static_assert(kWideRows % kTileRows == 0U && kTileRows % 16U == 0U, "a wide tile is whole 64-row tiles of whole m16 blocks");
static_assert(kWideMBlocks * warp_parts(kWideMBlocks) == kTileMBlocks * warp_parts(kTileMBlocks),
              "the wide form keeps the tile form's 32 accumulators per thread");
static_assert(kWideThreads / 32U == 8U * warp_parts(kTileMBlocks), "16 warps: 8 column blocks x the 2 parts of a tile-form warp");
// Launch rule (backend-neutral): the wide form when the launch routes at least min_avg rows per expert on average
// (min_avg = 0: off). min_avg is a planner parameter: the lever value `--expert-wide-tile N` (N rows per expert), taken
// from the no-model tile bench of the device (its TILEWIDE line: the smallest rows-per-expert case from which the wide
// form is faster at every larger case). kWideMinAvgRows is only the value of a bare `on`: the TILEWIDE of the RTX PRO 6000
// Blackwell (sm_120, 188 SMs; tile bench 11-19, 11/11 cases bits EQUAL): wide / tile time 0.79-0.96 at every case from 80
// to 320 rows per expert, 1.05-1.07 at 48-64. Another device takes its own TILEWIDE as `--expert-wide-tile N`.
inline constexpr unsigned kWideMinAvgRows = 80U;
STRIX_NATIVE_MMA_HD bool wide_form(std::uint64_t assignments, unsigned experts, unsigned min_avg) {
    return min_avg != 0U && assignments >= std::uint64_t{min_avg} * experts;
}
// Upper bound of the CSR tiles of a launch (the kernels' grid x): one partial tile per expert plus the full ones.
STRIX_NATIVE_MMA_HD unsigned tile_bound(unsigned assignments, unsigned experts, unsigned rows_per_tile) {
    const unsigned x = experts + (assignments + rows_per_tile - 1U) / rows_per_tile;
    return x < assignments ? x : assignments;
}
// Group held by ring slot `slot` after group `group` was taken (slot 0 = the next group to consume; the kernels load
// ring_group(group, ahead, groups) into the last slot). Past the last group a slot re-loads `group` itself (never
// consumed, never past the weight row): with ahead = 1 this is the tile form's `next` expression.
STRIX_NATIVE_MMA_HD unsigned ring_group(unsigned group, unsigned slot, unsigned groups) {
    return group + slot < groups ? group + slot : group;
}
// Output of C column col: project warp w owns outputs 8 w .. of both matrices; down warp w owns 16 w .. in two n8 blocks.
STRIX_NATIVE_MMA_HD unsigned project_output(unsigned block_y, unsigned warp, unsigned col) { return block_y * kProjectOutputs + 8U * warp + col; }
STRIX_NATIVE_MMA_HD unsigned down_output(unsigned block_y, unsigned warp, unsigned n, unsigned col) {
    return block_y * kDownOutputs + 16U * warp + 8U * n + col;
}
// Tile-major raster. The kernels are launched on grid (tiles, slices) and blocks are dispatched in linear order
// L = x + grid_x * y. Reading L as (tile = L / slices, slice = L % slices) makes the output slices of one tile consecutive,
// so they run together and share the tile's activations (project X, down h) in L2 instead of re-reading them once per
// full sweep over all tiles. A bijection of [0, grid_x) x [0, slices); the arithmetic of every output is unchanged.
struct TileSlice { unsigned tile, slice; };
STRIX_NATIVE_MMA_HD TileSlice tile_major(unsigned block_x, unsigned block_y, unsigned grid_x, unsigned slices) {
    const unsigned l = block_x + grid_x * block_y;
    return TileSlice{l / slices, l % slices};
}

STRIX_NATIVE_MMA_HD std::uint32_t f32_bits(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return u; }
STRIX_NATIVE_MMA_HD float bits_f32(std::uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }

// Exact FP32 of a small integer by the magic-number subtraction (both operands exact, |c| < 2^22): +0 for c = 0.
STRIX_NATIVE_MMA_HD float code_float(int c) { return bits_f32(0x4B400000U + static_cast<std::uint32_t>(c)) - 12582912.0F; }
// BF16 bits of an integer code, |c| <= 256: the FP32 value has at most 8 significant bits, so truncation is exact.
STRIX_NATIVE_MMA_HD std::uint32_t code_bf16(int c) { return f32_bits(code_float(c)) >> 16U; }
// The two B registers of one half from a packed code word (byte j = code of logical k 4 q + j, two's complement).
STRIX_NATIVE_MMA_HD void code_pairs(std::uint32_t word, std::uint32_t& b0, std::uint32_t& b1) {
    const int c0 = static_cast<std::int8_t>(word & 255U), c1 = static_cast<std::int8_t>((word >> 8U) & 255U);
    const int c2 = static_cast<std::int8_t>((word >> 16U) & 255U), c3 = static_cast<std::int8_t>(word >> 24U);
    b0 = code_bf16(c0) | (code_bf16(c1) << 16U);
    b1 = code_bf16(c2) | (code_bf16(c3) << 16U);
}

// FP32 -> BF16 round to nearest even, bit form (equals cvt.rn.bf16.f32 for every non-NaN input, subnormals included).
STRIX_NATIVE_MMA_HD std::uint32_t bf16_rn(float x) {
    const std::uint32_t u = f32_bits(x);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) return (u >> 16U) | 0x40U;  // NaN stays NaN (payload not part of the contract)
    return (u + 0x7FFFU + ((u >> 16U) & 1U)) >> 16U;
}
STRIX_NATIVE_MMA_HD float bf16_float(std::uint32_t b) { return bits_f32(b << 16U); }
// hi/lo split of the down activation: x - hi is exact in FP32 (|x - hi| <= half a BF16 ulp of x).
STRIX_NATIVE_MMA_HD void split(float x, std::uint32_t& hi, std::uint32_t& lo) {
    hi = bf16_rn(x);
    lo = bf16_rn(x - bf16_float(hi));
}

}  // namespace strix::cuda::core::native_mma
