#pragma once
// Stage layout and coordinate maps of the sm_120 grouped W4A4 expert tile (contract: moe_core/grouped_w4a4.hpp). Host and device:
// the kernel (grouped_w4a4.cuh) and the host fragment simulation (tests/moe_core/grouped_w4a4_host_test.cpp) use the same maps.
//
// CTA = 128 outputs (mma m, weights) x 128 routed rows (mma n, activations), 256 threads = 8 warps as 2 (outputs, 64 each = 4 m16)
// x 4 (rows, 32 each = 4 n8); per 64-input block a warp issues 4 x 4 m16n8k64 block-scaled MMAs (kind::mxf4nvf4, scale_vec 4X).
// A plan tile of <= 128 rows is one CTA, so an expert's weights cross L2 -> shared memory once per tile (the 64-row CTA read them
// once per 64-row slice).
// Stage = 128 inputs; a stage row holds the row's 64 code bytes, then its 8 E4M3 scale bytes, then 8 pad bytes (80 B pitch).
// Fragments come from shared memory by ldmatrix (m8n8 x4 b16: lane l receives word l % 4 of the 16-byte row whose address lane
// 8 j + l / 4 supplied, for matrix j): an A fragment (rows g, g + 8 x two 16-byte halves of a 64-input block) is ONE x4, two B
// fragments (rows g of two n8 blocks x two halves) are one x4; the 8 row addresses of one matrix are 80 B apart, so its 16-byte
// rows fall on banks 20 r .. 20 r + 3 (mod 32), all distinct. The E4M3 scale words stay LDS.32.
// Per warp and 64-input block: 4 + 2 ldmatrix.x4 + 8 LDS.32 for 16 MMAs (the 64-row form: 22 LDS.32 for 8 MMAs); shared memory
// per MAC 0.031 B (was 0.043). 4 stages in a cp.async ring (80 KiB dynamic shared memory, one CTA per SM), one barrier per stage.
// The registers a lane holds and the MMA order per accumulator (stage ascending, 64-input block ascending) are those of the
// 64-row form, so every output element has the same bits (identity grouped-nvfp4-w4a4-f32-v1 unchanged).
// Rows: a warp whose n8 block starts at or past the tile's live rows issues neither its B loads nor its MMAs, and the stage copy
// skips activation rows past the live rows rounded up to 8 (never read); rows in a partly live n8 block are zero-filled.
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define STRIX_GW4_HD __host__ __device__ __forceinline__
#else
#define STRIX_GW4_HD inline
#endif

namespace strix::cuda::core::grouped_w4a4::sm120 {

inline constexpr unsigned kThreads = 256U, kStages = 4U;
inline constexpr unsigned kOutputs = 128U, kRows = 128U, kStageInputs = 128U;
inline constexpr unsigned kRowBytes = 80U, kCodeBytes = 64U, kScaleOffset = 64U, kScaleBytes = 8U;
inline constexpr unsigned kWBytes = kOutputs * kRowBytes, kXBytes = kRows * kRowBytes;
inline constexpr unsigned kStageBytes = kWBytes + kXBytes;
inline constexpr unsigned kSmem = kStages * kStageBytes;  // 81,920 B dynamic (sm_120: <= 99 KiB per CTA)
static_assert(kSmem <= 99U * 1024U && kRowBytes % 16U == 0U && kScaleOffset % 16U == 0U);

// Copy plan of one stage: 4 code chunks (16 B) per thread (j 0, 1: weight rows; j 2, 3: activation rows) and one scale chunk
// (8 B; threads 0..127 weight rows, 128..255 activation rows).
inline constexpr unsigned kCodeChunksPerThread = 4U;
struct Chunk {
    bool activation;
    unsigned row;
    unsigned part;  // 16-byte part of the row's 64 code bytes
};
STRIX_GW4_HD Chunk code_chunk(unsigned thread, unsigned j) {
    const unsigned c = thread + kThreads * j;
    return c < 4U * kOutputs ? Chunk{false, c / 4U, c % 4U} : Chunk{true, (c - 4U * kOutputs) / 4U, c % 4U};
}
STRIX_GW4_HD bool has_scale_chunk(unsigned thread) { return thread < kOutputs + kRows; }
STRIX_GW4_HD Chunk scale_chunk(unsigned thread) {
    return thread < kOutputs ? Chunk{false, thread, 0U} : Chunk{true, thread - kOutputs, 0U};
}
STRIX_GW4_HD unsigned stage_offset(bool activation, unsigned row, unsigned byte) {
    return (activation ? kWBytes : 0U) + row * kRowBytes + byte;
}

// Warp w: wm = w & 1 (64-output half), wn = w >> 1 (32-row quarter). mt = m16 block (4 per warp), nt = n8 block (4 per warp).
STRIX_GW4_HD unsigned warp_outputs(unsigned warp) { return (warp & 1U) * 64U; }
STRIX_GW4_HD unsigned warp_rows(unsigned warp) { return (warp >> 1U) * 32U; }
// ldmatrix row address supplied by `lane` (row, byte of 64-input block kb):
//   A x4 of block mt: matrix j = lane / 8 -> rows + 8 (j & 1), half j >> 1  => registers {g lo, g + 8 lo, g hi, g + 8 hi}.
//   B x4 of blocks 2 p, 2 p + 1: matrix j -> block 2 p + (j >> 1), half j & 1 => registers {2p lo, 2p hi, 2p+1 lo, 2p+1 hi}.
STRIX_GW4_HD unsigned ldsm_a_row(unsigned warp, unsigned mt, unsigned lane) {
    return warp_outputs(warp) + 16U * mt + 8U * ((lane >> 3U) & 1U) + (lane & 7U);
}
STRIX_GW4_HD unsigned ldsm_a_byte(unsigned kb, unsigned lane) { return 32U * kb + 16U * (lane >> 4U); }
STRIX_GW4_HD unsigned ldsm_b_row(unsigned warp, unsigned p, unsigned lane) {
    return warp_rows(warp) + 8U * (2U * p + (lane >> 4U)) + (lane & 7U);
}
STRIX_GW4_HD unsigned ldsm_b_byte(unsigned kb, unsigned lane) { return 32U * kb + 16U * ((lane >> 3U) & 1U); }
// E4M3 scale words (LDS.32): A row g + 8 (lane & 1) of block mt, B row g of block nt.
STRIX_GW4_HD unsigned a_scale_row(unsigned warp, unsigned mt, unsigned lane) {
    return warp_outputs(warp) + 16U * mt + lane / 4U + 8U * (lane & 1U);
}
STRIX_GW4_HD unsigned b_scale_row(unsigned warp, unsigned nt, unsigned lane) { return warp_rows(warp) + 8U * nt + lane / 4U; }
STRIX_GW4_HD unsigned scale_byte(unsigned kb) { return kScaleOffset + 4U * kb; }
// Accumulator acc[mt][nt][z] of lane (g, t): output g + 8 (z >> 1) of block mt, row 2 t + (z & 1) of block nt.
STRIX_GW4_HD unsigned acc_output(unsigned warp, unsigned mt, unsigned lane, unsigned z) {
    return warp_outputs(warp) + 16U * mt + lane / 4U + 8U * (z >> 1U);
}
STRIX_GW4_HD unsigned acc_row(unsigned warp, unsigned nt, unsigned lane, unsigned z) {
    return warp_rows(warp) + 8U * nt + 2U * (lane % 4U) + (z & 1U);
}
// n8 blocks of the warp that hold live rows (0..4): a warp-uniform bound for the B loads and the MMAs.
STRIX_GW4_HD unsigned live_blocks(unsigned warp, unsigned rows) {
    const unsigned first = warp_rows(warp);
    if (rows <= first) return 0U;
    const unsigned n = (rows - first + 7U) / 8U;
    return n < 4U ? n : 4U;
}

}  // namespace strix::cuda::core::grouped_w4a4::sm120

#undef STRIX_GW4_HD
