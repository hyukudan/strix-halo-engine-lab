#pragma once
// Chunked gated-delta-rule PREFILL (versioned FAST contract `gdn-chunk16-f32-v1`), shared CUDA core. Vendor-free: the CUDA kernels
// (src/cuda_core/gdn/gdn_chunk.cu), the host oracle (tests/gdn_chunk) and the engines use these constants and layout functions.
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

#if defined(__CUDACC__)
#define STRIX_GDN_CHUNK_HD __host__ __device__
#else
#define STRIX_GDN_CHUNK_HD
#endif

namespace strix::cuda_core::gdn_chunk {

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
//   tf32 = gdn-chunk16-tf32-v1: K2 on mma.m16n8k8.tf32 with FP32 accumulation; the FP32 state S^T lives in the accumulator
//          fragments for the whole launch (grid value_heads x 2, 4 warps x 16 value columns). Every product operand is rounded
//          to TF32 (cvt.rna: 10-bit mantissa, relative 2^-11); sums, decays and the state stay FP32. Same algebra as above,
//          other bits: FAST class, engine identity suffix -gdnc16t, gated on its own.
enum class Scan : std::uint8_t { f32 = 0, tf32 = 1 };
inline constexpr const char* kIdentityTf32 = "gdn-chunk16-tf32-v1";
[[nodiscard]] constexpr const char* identity(Scan s) noexcept { return s == Scan::tf32 ? kIdentityTf32 : kIdentity; }

namespace tc {
inline constexpr std::uint32_t kWarps = 4;                  // warps per CTA
inline constexpr std::uint32_t kVCols = 16;                 // value columns per warp (the MMA M dimension of S^T)
inline constexpr std::uint32_t kCtaCols = kWarps * kVCols;  // 64 value columns per CTA: grid = value_heads x (kDim / 64)
// Shared-memory row strides (floats) of one operand stage; each makes its fragment load bank-conflict free (32-bit loads:
// 32 distinct banks; the 64-bit Q load: 32 distinct banks per half warp). The host test checks every one.
inline constexpr std::uint32_t kWtStride = 20;   // Wt[k][i]: B of X^T = S^T W^T, rows k = 2t, 2t + 1, column i = g
inline constexpr std::uint32_t kQStride = 136;   // Q[i][k]:  B of Y^T = S^T Q^T, one float2 at row i = g, keys 2t, 2t + 1
inline constexpr std::uint32_t kKStride = 132;   // K[j][k]:  B of S^T += U'^T K, rows j = 2t, 2t + 1, column k = g
inline constexpr std::uint32_t kUStride = 68;    // U~[i][v]: read at the C fragment, rows i = 2t, 2t + 1, column v = g
inline constexpr std::uint32_t kPtStride = 20;   // Pt[j][i]: B of O^T += U^T P^T, rows j = 2t, 2t + 1, column i = g
struct Stage {  // floats of one operand stage (two stages, double buffered by cp.async)
    static constexpr std::size_t wt = 0;
    static constexpr std::size_t q = wt + std::size_t{kDim} * kWtStride;
    static constexpr std::size_t k = q + std::size_t{kChunk} * kQStride;
    static constexpr std::size_t u = k + std::size_t{kChunk} * kKStride;
    static constexpr std::size_t pt = u + std::size_t{kChunk} * kUStride;
    static constexpr std::size_t g = pt + std::size_t{kChunk} * kPtStride;
    static constexpr std::size_t floats = g + kChunk;  // 8,272 floats = 33,088 B
};
inline constexpr std::size_t kSmemBytes = 2 * Stage::floats * sizeof(float);  // 66,176 B (dynamic, opt-in above 48 KB)

// Fragment ownership of mma.m16n8k8 (PTX ISA), lane = 0..31, g = lane >> 2, t = lane & 3:
//   C/D element e = 0..3: row g + 8 (e >> 1), column 2t + (e & 1)
//   A element e = 0..3:   row g + 8 (e & 1),  k-slot t + 4 (e >> 1)
//   B element e = 0..1:   k-slot t + 4 e,      column g
// The hardware sums the 8 k-slots of a step in its own order, so the k-slots may stand for the 8 k indices of a block in any
// permutation, as long as A and B use the same one. With perm(s) = 2 (s & 3) + (s >> 2) the C fragment of a product whose
// COLUMNS are the next product's k index is that product's A fragment as is: a[e] = c[a_from_c(e)] (no shuffle, no smem).
// The scan uses it three times: S^T (columns = keys) -> X^T and Y^T; U^T (columns = rows j) -> O^T and the S^T update.
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr int c_row(int lane, int e) noexcept { return (lane >> 2) + 8 * (e >> 1); }
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr int c_col(int lane, int e) noexcept { return 2 * (lane & 3) + (e & 1); }
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr int a_row(int lane, int e) noexcept { return (lane >> 2) + 8 * (e & 1); }
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr int a_kslot(int lane, int e) noexcept { return (lane & 3) + 4 * (e >> 1); }
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr int b_kslot(int lane, int e) noexcept { return (lane & 3) + 4 * e; }
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr int b_col(int lane) noexcept { return lane >> 2; }
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr int perm(int kslot) noexcept { return 2 * (kslot & 3) + (kslot >> 2); }
[[nodiscard]] STRIX_GDN_CHUNK_HD constexpr int a_from_c(int e) noexcept { return ((e & 1) << 1) | (e >> 1); }
}  // namespace tc

}  // namespace strix::cuda_core::gdn_chunk
