#pragma once
// Chunked gated-delta-rule PREFILL (versioned FAST contract `gdn-chunk16-f32-v1`), shared contract (vendor-free). The CUDA kernels
// (src/cuda/core/recur_chunk.cu), the host oracle (tests/gdn_chunk) and the engines use these constants and layout functions.
//
// Same recurrence as gdn.hpp (per value head h, key head h % key_heads, state S[k][v], q pre-scaled 1/sqrt(dk), k/q l2-normalised):
//   S_r = a_r S_{r-1} + k_r u_r^T,  u_r = beta_r (v_r - a_r S_{r-1}^T k_r),  o_r = S_r^T q_r,  a_r = exp(g_r),
//   g_r = ssm_a * softplus(a_raw_r + dt_bias), beta_r = sigmoid(b_raw_r)
// evaluated per chunk of C = 16 rows from the chunk-start state S0 (G_i = sum_{l <= i} g_l inside the chunk, i = 0..C-1):
//   A_ij = beta_i exp(G_i - G_j) k_i.k_j (j < i),  T = (I + A)^-1 (unit lower triangular, forward substitution)
//   W = T diag(beta exp(G)) K,  U~ = T diag(beta) V                  (K1, all chunks in parallel)
//   U = U~ - W S0,  O = diag(exp(G)) Q S0 + P U,  P_ij = exp(G_i - G_j) q_i.k_j (j <= i)
//   S_C = exp(G_{C-1}) S0 + K~^T U,  K~_j = exp(G_{C-1} - G_j) k_j     (K2, chunks sequential per (head, 32-column value tile))
// then the gated RMSNorm of gdn.hpp on o (K3). Algebraically equal to the serial recurrence; the summation order and the
// triangular solve differ, so bits differ: FAST class, engine identity suffix `-gdnc16`, opt-in until its logit gate passes.
// All arithmetic is FP32 (FFMA): no operand rounding. Padding rows (>= rows) carry beta = 0, g = 0, q = k = v = 0: u = 0, no
// state change, no output.
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_GDN_CHUNK_HD __host__ __device__
#else
#define STRIX_GDN_CHUNK_HD
#endif

namespace strix::gdn_chunk {

inline constexpr const char* kIdentity = "gdn-chunk16-f32-v1";
inline constexpr std::uint32_t kChunk = 16;   // rows per chunk (C)
inline constexpr std::uint32_t kDim = 128;    // key_dim == value_dim (gdn.hpp supports 128 only)
inline constexpr std::uint32_t kVTile = 32;   // value columns per K2 block (grid = value_heads x 4)

[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr std::uint32_t chunks(std::uint32_t rows) noexcept { return (rows + kChunk - 1) / kChunk; }

// Workspace written by K1 and read by K2, per (chunk c, value head h), floats:
//   Wt [kDim k][kChunk i]  W transposed (K2 reads W[i..i+3][k] as one float4)
//   U  [kChunk i][kDim v]  U~ row major (K2 reads its 32-column tile)
//   Pt [kChunk j][kChunk i] P transposed, zero above the diagonal
//   G  [kChunk]            cumulative log decay inside the chunk
struct Layout {
    static constexpr std::size_t wt = 0;
    static constexpr std::size_t u = wt + std::size_t{kDim} * kChunk;
    static constexpr std::size_t pt = u + std::size_t{kChunk} * kDim;
    static constexpr std::size_t g = pt + std::size_t{kChunk} * kChunk;
    static constexpr std::size_t block = g + kChunk;  // floats per (chunk, head): 4,368
};
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr std::size_t block_offset(std::uint32_t c, std::uint32_t h, std::uint32_t value_heads) noexcept {
    return (static_cast<std::size_t>(c) * value_heads + h) * Layout::block;
}
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr std::size_t workspace_floats(std::uint32_t value_heads, std::uint32_t rows) noexcept {
    return static_cast<std::size_t>(chunks(rows)) * value_heads * Layout::block;
}

// Scan trait of the chunk family (K1 and its workspace are shared by both):
//   f32  = gdn-chunk16-f32-v1: K2 on FP32 FFMA, state tile in shared memory (grid value_heads x 4, 128 threads).
//   tf32 = gdn-chunk16-tf32-v2: K2 on mma.m16n8k8.tf32 with FP32 accumulation; the FP32 state S^T lives in the accumulator
//          fragments for the whole launch (grid value_heads x 2, 4 warps x 16 value columns). Every product runs in split TF32
//          (x = hi + lo, hi = rna(x), lo = rna(x - hi), cvt.rna keeps 10 mantissa bits; A B ~ A_lo B_hi + A_hi B_lo + A_hi B_hi,
//          the lo * lo term (relative 2^-22) dropped) with FP32 accumulation; sums, decays and the state stay FP32. Result:
//          FP32-class error (the f32 trait's bounds hold). v1 rounded every operand once (relative 2^-11) and lost bits where
//          the recurrence carries or cancels them: the state write U'^T K put a 2^-11 error into every carried increment, the
//          delta read W S0 feeds U = U~ - W S0 (a small residual), and P U / Q S0 rounded the output (v1 state and output
//          errors ~1,000-2,000x the f32 trait in the host emulation). 300 mma per chunk and warp (v1: 100). Same algebra as
//          f32, other bits: FAST class, engine identity suffix -gdnc16t2, gated on its own.
enum class Scan : std::uint8_t { f32 = 0, tf32 = 1 };
inline constexpr const char* kIdentityTf32 = "gdn-chunk16-tf32-v2";
[[nodiscard]] constexpr const char* identity(Scan s) noexcept { return s == Scan::tf32 ? kIdentityTf32 : kIdentity; }

}  // namespace strix::gdn_chunk
