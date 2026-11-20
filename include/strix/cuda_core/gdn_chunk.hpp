// Forwarding header: the chunked GDN prefill contract (gdn-chunk16-f32-v1, vendor-free) moved to strix/gdn_chunk.hpp
// (namespace strix::gdn_chunk). Kept until the CUDA core includes the new path; it also holds the CUDA-only TF32 fragment
// map of the scan trait (below).
#pragma once

#include "strix/gdn_chunk.hpp"

namespace strix::cuda_core {
namespace gdn_chunk = ::strix::gdn_chunk;
}  // namespace strix::cuda_core

// CUDA mma.m16n8k8 fragment map of the TF32 scan trait (strix::gdn_chunk::Scan::tf32): CUDA-only, so it stays in the
// CUDA core header and extends the shared contract namespace.
namespace strix::gdn_chunk {

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

}  // namespace strix::gdn_chunk
