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
// Rows: a warp whose n8 block starts at or past the tile's live rows issues neither its B loads nor its MMAs. The stage copy
// (code_copy_rows / scale_copy_rows) writes activation code rows up to the live rows rounded up to 16 (the x4 B load reads
// n8 blocks in pairs) and scale rows up to the live rows rounded up to 8; rows past the live rows are zero-filled, rows past
// those extents are neither copied nor read.
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
// Two activation terms (one K sweep): a stage holds the weight block once and one activation block per term; three stages.
STRIX_GW4_HD constexpr unsigned stage_bytes(unsigned terms) { return kWBytes + terms * kXBytes; }
// Activation stage-copy extents of a slice with `rows` live rows (<= kRows), shared by the kernel and the host simulation: code
// rows r < code_copy_rows(rows) = rows rounded up to 16 are copied (one x4 B load covers two n8 blocks, so the partner block of
// the last live n8 block is read), scale rows r < scale_copy_rows(rows) = rows rounded up to 8 (scale words are read per live
// n8 block only).
STRIX_GW4_HD constexpr unsigned code_copy_rows(unsigned rows) { return (rows + 15U) & ~15U; }
STRIX_GW4_HD constexpr unsigned scale_copy_rows(unsigned rows) { return (rows + 7U) & ~7U; }
STRIX_GW4_HD constexpr unsigned stage_count(unsigned terms) { return terms == 1U ? kStages : 3U; }
STRIX_GW4_HD constexpr unsigned smem_bytes(unsigned terms) { return stage_count(terms) * stage_bytes(terms); }
static_assert(smem_bytes(1U) == kSmem && smem_bytes(2U) <= 99U * 1024U, "92,160 B for two terms");

// Copy plan of one stage: 2 + 2 terms code chunks (16 B) per thread (j 0, 1: weight rows; j >= 2: activation stage rows) and
// scale chunks (8 B) c = thread + 256 j < 128 + 128 terms (c < 128: weight row c; else activation stage row c - 128). Activation
// stage row 128 t + r is row r of term t: term t's block follows term t - 1's, so stage_offset needs no term. One term: 4 code
// chunks and one scale chunk per thread.
inline constexpr unsigned kCodeChunksPerThread = 4U;
STRIX_GW4_HD constexpr unsigned code_chunks(unsigned terms) { return 2U + 2U * terms; }
STRIX_GW4_HD constexpr unsigned scale_chunks(unsigned terms) { return (kOutputs + terms * kRows + kThreads - 1U) / kThreads; }
struct Chunk {
    bool activation;
    unsigned row;
    unsigned part;  // 16-byte part of the row's 64 code bytes
};
STRIX_GW4_HD Chunk code_chunk(unsigned thread, unsigned j) {
    const unsigned c = thread + kThreads * j;
    return c < 4U * kOutputs ? Chunk{false, c / 4U, c % 4U} : Chunk{true, (c - 4U * kOutputs) / 4U, c % 4U};
}
STRIX_GW4_HD bool has_scale_chunk(unsigned thread, unsigned j = 0U, unsigned terms = 1U) {
    return thread + kThreads * j < kOutputs + terms * kRows;
}
STRIX_GW4_HD Chunk scale_chunk(unsigned thread, unsigned j = 0U) {
    const unsigned c = thread + kThreads * j;
    return c < kOutputs ? Chunk{false, c, 0U} : Chunk{true, c - kOutputs, 0U};
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
