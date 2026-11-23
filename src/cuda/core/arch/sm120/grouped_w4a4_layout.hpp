#pragma once
// Stage layout and coordinate maps of the sm_120 grouped W4A4 expert tile (contract: moe_core/grouped_w4a4.hpp). Host and device:
// the kernel (grouped_w4a4.cuh) and the host fragment simulation (tests/moe_core/grouped_w4a4_host_test.cpp) use the same maps.
//
// CTA = 128 outputs (mma m, weights) x 64 routed rows (mma n, activations), 256 threads = 8 warps as 4 (outputs, 32 each = 2 m16)
// x 2 (rows, 32 each = 4 n8); per 64-input block a warp issues 2 x 4 m16n8k64 block-scaled MMAs (kind::mxf4nvf4, scale_vec 4X).
// Stage = 128 inputs; a stage row holds the row's 64 code bytes, then its 8 E4M3 scale bytes, then 8 pad bytes (80 B pitch:
// the 8 rows g of one fragment read 4-byte words at banks 20 g + t, all distinct). 3 stages in a cp.async ring, one barrier per
// stage (the incumbent BF16 tile: one per 16 inputs). Work per MAC from shared memory: 22 LDS.32 per 8 MMAs per warp
// = 0.043 B/MAC (88 B/clk/SM at the FP4 tensor rate, under 128 B/clk).
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define STRIX_GW4_HD __host__ __device__ __forceinline__
#else
#define STRIX_GW4_HD inline
#endif

namespace strix::cuda::core::grouped_w4a4::sm120 {

inline constexpr unsigned kThreads = 256U, kStages = 3U;
inline constexpr unsigned kOutputs = 128U, kRows = 64U, kStageInputs = 128U;
inline constexpr unsigned kRowBytes = 80U, kCodeBytes = 64U, kScaleOffset = 64U, kScaleBytes = 8U;
inline constexpr unsigned kWBytes = kOutputs * kRowBytes, kXBytes = kRows * kRowBytes;
inline constexpr unsigned kStageBytes = kWBytes + kXBytes;
inline constexpr unsigned kSmem = kStages * kStageBytes;  // 46,080 B (static limit 48 KiB)
static_assert(kSmem <= 48U * 1024U && kRowBytes % 16U == 0U && kScaleOffset % 16U == 0U);

// Copy plan of one stage: 768 16-byte code chunks (thread c, c + 256, c + 512: weight rows 0..63, 64..127, activation rows
// 0..63; chunk part = c % 4) and 192 8-byte scale chunks (thread < 128: weight row, < 192: activation row thread - 128).
inline constexpr unsigned kCodeChunksPerThread = 3U;
struct Chunk {
    bool activation;
    unsigned row;
    unsigned part;  // 16-byte part of the row's 64 code bytes
};
STRIX_GW4_HD Chunk code_chunk(unsigned thread, unsigned j) {
    const unsigned c = thread + kThreads * j;
    return c < 2U * kThreads ? Chunk{false, c / 4U, c % 4U} : Chunk{true, (c - 2U * kThreads) / 4U, c % 4U};
}
STRIX_GW4_HD bool has_scale_chunk(unsigned thread) { return thread < kOutputs + kRows; }
STRIX_GW4_HD Chunk scale_chunk(unsigned thread) {
    return thread < kOutputs ? Chunk{false, thread, 0U} : Chunk{true, thread - kOutputs, 0U};
}
// Byte offset in a stage of (operand, row, byte of the row).
STRIX_GW4_HD unsigned stage_offset(bool activation, unsigned row, unsigned byte) {
    return (activation ? kWBytes : 0U) + row * kRowBytes + byte;
}

// Warp roles and fragment coordinates (g = lane / 4, t = lane % 4), block kb (0, 1) of the stage.
STRIX_GW4_HD unsigned warp_outputs(unsigned warp) { return (warp & 3U) * 32U; }
STRIX_GW4_HD unsigned warp_rows(unsigned warp) { return (warp >> 2U) * 32U; }
// A register i (0..3) of lane: weight row wo + 16 mb + g + 8 (i & 1), code byte 32 kb + 16 (i >> 1) + 4 t (8 inputs).
STRIX_GW4_HD unsigned a_row(unsigned warp, unsigned mb, unsigned lane, unsigned i) {
    return warp_outputs(warp) + 16U * mb + lane / 4U + 8U * (i & 1U);
}
STRIX_GW4_HD unsigned a_byte(unsigned kb, unsigned lane, unsigned i) { return 32U * kb + 16U * (i >> 1U) + 4U * (lane % 4U); }
// A scale word: row g + 8 (lane & 1) of the m16 block, the 4 scale bytes of block kb.
STRIX_GW4_HD unsigned a_scale_row(unsigned warp, unsigned mb, unsigned lane) {
    return warp_outputs(warp) + 16U * mb + lane / 4U + 8U * (lane & 1U);
}
// B register i (0, 1): activation row wr + 8 nb + g, code byte 32 kb + 16 i + 4 t; scale word of row wr + 8 nb + g.
STRIX_GW4_HD unsigned b_row(unsigned warp, unsigned nb, unsigned lane) { return warp_rows(warp) + 8U * nb + lane / 4U; }
STRIX_GW4_HD unsigned b_byte(unsigned kb, unsigned lane, unsigned i) { return 32U * kb + 16U * i + 4U * (lane % 4U); }
STRIX_GW4_HD unsigned scale_byte(unsigned kb) { return kScaleOffset + 4U * kb; }
// Accumulator z (0..3) of acc[mb][nb]: output (tile column) and routed row.
STRIX_GW4_HD unsigned acc_output(unsigned warp, unsigned mb, unsigned lane, unsigned z) {
    return warp_outputs(warp) + 16U * mb + lane / 4U + 8U * (z >> 1U);
}
STRIX_GW4_HD unsigned acc_row(unsigned warp, unsigned nb, unsigned lane, unsigned z) {
    return warp_rows(warp) + 8U * nb + 2U * (lane % 4U) + (z & 1U);
}

}  // namespace strix::cuda::core::grouped_w4a4::sm120

#undef STRIX_GW4_HD
