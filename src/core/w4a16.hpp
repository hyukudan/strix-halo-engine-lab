#pragma once
// W4A16 small-M core over the IU4 g64 pack (FAST_I4 decode / verify, lossy; core/iu4.hpp layout):
//   W tiles [K/16][N/16][column within tile][K within tile / 2] signed nibbles (low nibble = even k), per column and
//   K64 group FP16 scale s, w = q * s. Activations FP16 (the caller's transformed rows, typically in LDS), F32
//   accumulation: per group a v_dot2_f32_f16 chain of nibble * activation pairs, then acc = fmaf(s, group sum, acc).
//   (v_dot2 on gfx1151 is not an exactly rounded chain: a few F32 ulps off even on integer operands.)
// One wave covers 4 adjacent 16-column tiles (64 outputs). Lane l reads one 16-byte run per k-tile: tile l / 8,
// columns 2 (l % 8) and 2 (l % 8) + 1, all 16 k of each (no cross-lane reduction). Rows are a static bucket R; rows
// at or past nrows are skipped (their emitted values are 0) and the caller must not store them.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include "core/iu4_rows_plan.hpp"
#include "strix/decode_weight_sweep.hpp"
#include "strix/hip/legacy_intrinsics.hpp"  // STRIX_HIP_FDOT2

namespace strix::core::w4a16 {

using Half2 = _Float16 __attribute__((ext_vector_type(2)));
using Float2 = float __attribute__((ext_vector_type(2)));
// Volatile access through a generic pointer lowers to FLAT plus global-memory
// completion waits. Preserve the local address space explicitly for dot staging.
using LocalFloat2 = Float2 __attribute__((address_space(3)));
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));
// Constant address space: wave-uniform loads through it are scalar (s_load, lgkmcnt).
using ConstU32 = const std::uint32_t __attribute__((address_space(4)));

// Nibbles at bit s and 16 + s of w as FP16 pair (exact): ((n ^ 8) | 0x6400) = 1024 + n + 8 (two's complement n), - 1032.
__device__ __forceinline__ Half2 nibble_pair(std::uint32_t w, unsigned s) {
    const std::uint32_t t = (((w >> s) & 0x000F000FU) ^ 0x00080008U) | 0x64006400U;
    return __builtin_bit_cast(Half2, t) - Half2{static_cast<_Float16>(1032.0F), static_cast<_Float16>(1032.0F)};
}
// a[0..3]: activation words (x 2i, x 2i + 1) of 8 consecutive k; p[j] pairs k j with k j + 4 (the nibble_pair order).
__device__ __forceinline__ void pair_activations(const std::uint32_t (&a)[4], Half2 (&p)[4]) {
    p[0] = __builtin_bit_cast(Half2, (a[0] & 0xFFFFU) | (a[2] << 16U));
    p[1] = __builtin_bit_cast(Half2, (a[0] >> 16U) | (a[2] & 0xFFFF0000U));
    p[2] = __builtin_bit_cast(Half2, (a[1] & 0xFFFFU) | (a[3] << 16U));
    p[3] = __builtin_bit_cast(Half2, (a[1] >> 16U) | (a[3] & 0xFFFF0000U));
}
// Dot of 8 nibbles (one word, k b..b + 7) with the paired activations of those k.
__device__ __forceinline__ float dot8(std::uint32_t w, const Half2 (&p)[4], float acc) {
    acc = STRIX_HIP_FDOT2(nibble_pair(w, 0U), p[0], acc, false);
    acc = STRIX_HIP_FDOT2(nibble_pair(w, 4U), p[1], acc, false);
    acc = STRIX_HIP_FDOT2(nibble_pair(w, 8U), p[2], acc, false);
    return STRIX_HIP_FDOT2(nibble_pair(w, 12U), p[3], acc, false);
}

// wq: this matrix's packed nibbles (one expert), ws: its FP16 scales [N][K/64]. k-tiles [k0, k1) (multiples of 4);
// xs: FP16 rows, row r at xs + r * xs_stride, k-tile k at local offset (k - k0) * 16 (16-byte aligned rows not needed).
// emit(split, r, column, value) at every KPS k-tiles (KPS % 4 == 0), split = (k - k0) / KPS, column = the output
// index col0 * 16 + 0..63, value = the split's F32 sum over its groups (each group: fmaf(s, dot chain, acc), groups
// ascending, acc from +0).
// MaxG: compile-time bound of the groups in [k0, k1) (k0 / 4 even): the lane's two columns' FP16 scales of the whole
// range are read once up front with 4-byte loads (groups are contiguous per column) instead of 2 scattered loads per group.
// ChunkedRows bounds activation scheduling to four rows while reusing each
// decoded weight stage for every row. StoreDots optionally keeps group dots in
// caller-provided R*32 Float2 LDS elements per wave, preserving their FP32 bits.
// UniformX: xs is global memory in the pre-paired layout (core/iu4_x_pairs.hpp) that no wave writes during the kernel,
// and k0 / xs / xs_stride are wave-uniform (the caller passes a readfirstlane wave index). The activation words are then
// read with scalar loads (constant address space): every lane needs the same words, so they belong in SGPRs, they are
// already the dot2 operands (no per-row pairing), and their waits use lgkmcnt instead of vmcnt. On the vector path each row's broadcast global load sits behind the weight ring
// refills in the in-order vmcnt queue, so its s_waitcnt vmcnt(0) also waits for the DRAM refills of the next k-tiles
// (prefetch defeated once per row and k-tile). Same operands, same dot2 order: bitwise equal to UniformX = false.
template<int R, unsigned KPS, unsigned MaxG, bool ChunkedRows = false, bool StoreDots = false, bool UniformX = false, class Emit>
__device__ __forceinline__ void wave_rows(const std::uint8_t* wq, const std::uint16_t* ws, unsigned N, unsigned K, unsigned k0,
                                          unsigned k1, unsigned col0, const _Float16* xs, unsigned xs_stride, unsigned lane,
                                          unsigned nrows, Emit&& emit, Float2* staged_dot = nullptr) {
    static_assert(KPS % 4U == 0U && R >= 1, "split boundaries on K64 groups");
    static_assert(!StoreDots || ChunkedRows, "dot staging uses chunked activation scheduling");
    const unsigned ntiles = N / 16U, groups = K / 64U, tile = col0 + (lane >> 3U), pair = lane & 7U;
    const unsigned n0 = tile * 16U + 2U * pair;
    const auto* w4 = reinterpret_cast<const U4*>(wq);
    const auto* x32 = reinterpret_cast<const std::uint32_t*>(xs);
    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
    static_assert(MaxG >= 2U && MaxG % 2U == 0U, "scales preloaded in pairs of groups");
    const unsigned g0 = k0 / 4U, ng = (k1 - k0) / 4U;
    std::uint32_t sc[2][MaxG / 2U];
#pragma unroll
    for (unsigned c = 0; c < 2U; ++c) {
        const auto* sp = reinterpret_cast<const std::uint32_t*>(ws + std::size_t{n0 + c} * groups + g0);
#pragma unroll
        for (unsigned j = 0; j < MaxG / 2U; ++j) sc[c][j] = 2U * j < ng ? sp[j] : 0U;
    }
    // Four K16 stages: each wave issues one contiguous 512-byte run per stage.
    // Refill a consumed stage before its ALU work, overlapping the next group.
    U4 ring[4];
#pragma unroll
    for (unsigned t = 0; t < 4U; ++t)
        ring[t] = k0 + t < k1 ? w4[(std::size_t{k0 + t} * ntiles + tile) * 8U + pair] : U4{};
    for (unsigned gi = 0; gi < ng; ++gi) {
        const unsigned k = k0 + 4U * gi;
        const unsigned sh = 16U * (gi & 1U);
        const float s0 = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(sc[0][gi / 2U] >> sh)));
        const float s1 = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(sc[1][gi / 2U] >> sh)));
        // XRows: UniformX rows whose activation words are held in SGPRs at once (8 words per row; 8 rows = 64 SGPRs,
        // 16 would pass the 106-SGPR wave limit). Rows past XRows take another load + wait per k-tile; each row's
        // operands and dot2 order are unchanged, so every row is bitwise the same for any XRows. R <= 8: one chunk.
        constexpr int OperandRows = static_cast<int>(::strix::decode::kHipIu4ScalarX.operand_rows);
        constexpr int XRows = UniformX && R > OperandRows ? OperandRows : R;
        static_assert(R % XRows == 0, "activation chunks cover the rows bucket");
        constexpr int Chunk = ChunkedRows ? 4 : XRows;
        // UniformX indexes xw by the row inside its activation chunk: row chunks must nest inside it. (The vector form
        // keeps its previous loop: one activation chunk, row chunks of 4 may pass a bucket under 4 rows.)
        static_assert(!UniformX || XRows % Chunk == 0, "row chunks nest inside activation chunks");
        float dot[StoreDots ? Chunk : R][2] = {};
#pragma unroll
        for (unsigned t = 0; t < 4U; ++t) {
            const U4 wv = ring[t];
            if (k + t + 4U < k1) ring[t] = w4[(std::size_t{k + t + 4U} * ntiles + tile) * 8U + pair];
            Half2 wn[4][4];
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q)
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) wn[q][j] = nibble_pair(wv[q], 4U * j);
#pragma unroll
            for (int xb = 0; xb < R; xb += XRows) {
            // UniformX: every row's words of this activation chunk are requested before the first is used (scalar loads
            // complete out of order, so each use waits lgkmcnt(0): one exposed latency per chunk and k-tile, not per row).
            [[maybe_unused]] std::uint32_t xw[UniformX ? XRows : 1][8];
            if constexpr (UniformX) {
#pragma unroll
                for (int i = 0; i < XRows; ++i) {
                    const int r = xb + i;
                    if (static_cast<unsigned>(r) >= nrows) break;
                    const auto* xr = (const ConstU32*)(x32) +  // address-space cast (C-style only)
                                     (static_cast<std::size_t>(r) * xs_stride + (k + t - k0) * 16U) / 2U;
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j) xw[i][j] = xr[j];
                }
            }
#pragma unroll
            for (int h = xb; h < xb + XRows; h += Chunk) {
#pragma unroll
            for (int r = 0; r < Chunk; ++r) {
                const unsigned row = static_cast<unsigned>(h + r);
                const unsigned d = StoreDots ? static_cast<unsigned>(r) : row;
                if (row >= nrows) break;
                if constexpr (StoreDots) {
                    if (t) {
                        const Float2 previous = ((volatile LocalFloat2*)staged_dot)[row * 32U + lane];
                        dot[d][0] = previous[0]; dot[d][1] = previous[1];
                    } else dot[d][0] = dot[d][1] = 0.0F;
                }
                Half2 pl[4], ph[4];
                if constexpr (UniformX) {  // pre-paired words (core/iu4_x_pairs.hpp): already the dot2 operands
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j) {
                        pl[j] = __builtin_bit_cast(Half2, xw[row - static_cast<unsigned>(xb)][j]);
                        ph[j] = __builtin_bit_cast(Half2, xw[row - static_cast<unsigned>(xb)][4U + j]);
                    }
                } else {
                    const auto* xr = x32 + (static_cast<std::size_t>(row) * xs_stride + (k + t - k0) * 16U) / 2U;
                    const std::uint32_t lo[4] = {xr[0], xr[1], xr[2], xr[3]}, hi[4] = {xr[4], xr[5], xr[6], xr[7]};
                    pair_activations(lo, pl);
                    pair_activations(hi, ph);
                }
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) {
                    dot[d][0] = STRIX_HIP_FDOT2(wn[0][j], pl[j], dot[d][0], false);
                    dot[d][1] = STRIX_HIP_FDOT2(wn[2][j], pl[j], dot[d][1], false);
                }
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) {
                    dot[d][0] = STRIX_HIP_FDOT2(wn[1][j], ph[j], dot[d][0], false);
                    dot[d][1] = STRIX_HIP_FDOT2(wn[3][j], ph[j], dot[d][1], false);
                }
                if constexpr (StoreDots) {
                    if (t < 3U) ((volatile LocalFloat2*)staged_dot)[row * 32U + lane] = Float2{dot[d][0], dot[d][1]};
                    else {
                        acc[row][0] = __builtin_fmaf(s0, dot[d][0], acc[row][0]);
                        acc[row][1] = __builtin_fmaf(s1, dot[d][1], acc[row][1]);
                    }
                }
            }
            if constexpr (ChunkedRows) {
                // Bound live activation chunks in both compiler and machine
                // scheduling; neither fence emits a device barrier or wait.
                asm volatile("" ::: "memory");
                __builtin_amdgcn_sched_barrier(0);
            }
            }
            }
        }
        if constexpr (!StoreDots) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (static_cast<unsigned>(r) >= nrows) break;
            acc[r][0] = __builtin_fmaf(s0, dot[r][0], acc[r][0]);
            acc[r][1] = __builtin_fmaf(s1, dot[r][1], acc[r][1]);
        }
        }
        if ((k + 4U - k0) % KPS == 0U) {
            const unsigned split = (k + 4U - k0) / KPS - 1U;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (static_cast<unsigned>(r) >= nrows) break;
                emit(split, static_cast<unsigned>(r), n0, acc[r][0]);
                emit(split, static_cast<unsigned>(r), n0 + 1U, acc[r][1]);
                acc[r][0] = acc[r][1] = 0.0F;
            }
        }
    }
}

}  // namespace strix::core::w4a16
#endif
