#pragma once
// Shared EXACT rows-selected latent attention, materialised form (F4 rows-selected, contract
// include/strix/attn/mla_contract.hpp steps (2)-(4) without rope and sink): per query row a list of n <= kListCap cache
// positions, one bf16 latent row of kLat values per position shared by every head (keys = values), the absorbed query
// qt [rows][H][kLat] (FP32), FP32 output ob [rows][H][kLat]. Three passes through a probability image
// e [rows][H][kEStride] (FP32):
//   scores   e[h][j] = (fmaf chain over l = 0 .. kLat-1 ascending from +0 of qt[h][l] * c_j[l]) * kScale
//   softmax  m = max_j e[h][j] (256-lane max, then a halving max tree); w_t = exp(e_t - m) (the engine's declared exp);
//            sum = halving tree over 256 lanes of the lane chains a = a + w_t, t = lane, lane + 256, ...; p_t = w_t / sum
//   o-bar    ob[h][l] = halving tree over 256 lanes of the lane chains fmaf(p_t, c_t[l], a), t = lane, lane + 256, ...
// Identity (EXACT): every form below computes these chains and trees in this order, so they give the same bits for
// every (row, head), whatever the CTA shape or staging. The FAST flash form of the same contract is
// sparse_flash_mla.hpp (online softmax, WMMA sums; other bits).
//
// Spec S (engine data, no behaviour): static constexpr unsigned kLat (multiple of 128), kListCap (list stride per row,
// at most 2304: the lane-major layout holds kRounds <= 9 rounds of 256); static constexpr float kScale;
// using Cache (cache handle passed to the kernels by value); static const std::uint16_t* latent(const Cache&) ->
// rows [position][kLat] bf16; static float exp(float) (device-callable; the engine's declared exponential).
// Lists: [rows][kListCap] positions, counts n [rows]. Positions past n are never read for results (loads of entries
// past n use position 0, which every list holds).
//
// Forms (the caller launches them; grids in the comments):
//   scores_kernel<S, H>           H in {32, 64}: grid (ceil(kListCap / (kScoreWaves * 32 * (64 / H))), rows),
//                                 block 256; lane = 64 / H list entries, every head's chain in registers, q via SGPRs.
//   scores_paired_kernel<S, G>    H = 32, same chains with the inner step hand-paired as VOPD (generated
//                                 selected_exact_mla_scores_vopd.inc); G = 0: grid (kPairedTiles, rows); G > 0: 1-D grid
//                                 of groups of G rows x kPairedTiles tiles, rows fastest (ceil(C / G) * G * tiles CTAs).
//   softmax_kernel<S, kLane>      grid (heads, rows), block 256, in place; kLane: p stored lane-major, slot
//                                 kRounds * bitrev8(t % 256) + t / 256 (padding slots +0), for obar_lanes_kernel.
//   obar_lanes_kernel<S, Steps>   H = 32, lane-major p: grid (kLanesTiles, rows), block 256; the halving tree evaluated as
//                                 a binary-counter merge over k = bitrev8(lane) (left operand = the stored lower-k
//                                 subtree); latent staged in LDS as FP32, Steps k per batch, double buffered.
//   obar_tree_kernel<S>           any heads (multiple of 4), row-major p: grid (kLat / kTreeDims, rows), block 256; tree
//                                 levels 128 / 64 / 32 through LDS, 16 .. 1 by shuffles inside wave 0.
// Host statement of the layout and of the counter == tree identity: selected_exact_mla_rule.hpp.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "strix/hip/legacy_intrinsics.hpp"

namespace strix::attn_core::selected_exact {

template <class S> inline constexpr unsigned kRounds = (S::kListCap + 255) / 256;
template <class S> inline constexpr unsigned kEStride = kRounds<S> * 256;
// The forms stage 16 latent dims per entry step and 128 per lanes o-bar CTA; the lane-major image holds <= 9 rounds.
template <class S> inline constexpr bool kSpecOk = S::kLat % 128U == 0 && S::kListCap > 0 && kRounds<S> <= 9;

using ConstF = const __attribute__((address_space(4))) float;
typedef float SVec16 __attribute__((ext_vector_type(16)));
using ConstV16 = const __attribute__((address_space(4))) SVec16;

__device__ __forceinline__ float widen_bf16(std::uint16_t b) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U); }
// Halving tree over L lanes in LDS: s[l] = s[l] + s[l + h], h = L/2 .. 1; every lane returns the root.
template <unsigned L> __device__ __forceinline__ float lane_tree(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// ---------------------------------------------------------------- softmax (in place over e)
__device__ __forceinline__ unsigned bitrev8(unsigned x) { return __brev(x) >> 24; }
template <class S, bool kLane>
__global__ __launch_bounds__(256) void softmax_kernel(const std::uint32_t* n_all, unsigned heads, float* e_all) {
    static_assert(kSpecOk<S>, "selected_exact: kLat a multiple of 128, kListCap <= 2304");
    __shared__ float s[256];
    const unsigned h = blockIdx.x, r = blockIdx.y, l = threadIdx.x, n = n_all[r];
    float* e = e_all + (static_cast<std::size_t>(r) * heads + h) * kEStride<S>;
    float mx = -__builtin_huge_valf();
    for (unsigned t = l; t < n; t += 256) mx = e[t] > mx ? e[t] : mx;
    s[l] = mx;
    __syncthreads();
    for (unsigned hh = 128; hh > 0; hh >>= 1U) {
        if (l < hh) s[l] = s[l + hh] > s[l] ? s[l + hh] : s[l];
        __syncthreads();
    }
    const float m = s[0];
    __syncthreads();
    float a = 0.0F;
    for (unsigned t = l; t < n; t += 256) {
        e[t] = S::exp(e[t] - m);
        a = a + e[t];
    }
    const float sum = lane_tree<256>(s, a);
    if constexpr (!kLane) {
        for (unsigned t = l; t < n; t += 256) e[t] = e[t] / sum;
    } else {
        float q[kRounds<S>];
#pragma unroll
        for (unsigned i = 0; i < kRounds<S>; ++i) q[i] = l + 256 * i < n ? e[l + 256 * i] / sum : 0.0F;
        __syncthreads();  // every thread's reads of e are done
        float* lane = e + kRounds<S> * bitrev8(l);
#pragma unroll
        for (unsigned i = 0; i < kRounds<S>; ++i) lane[i] = q[i];
    }
}

// ---------------------------------------------------------------- scores, lane = 64 / H list entries
// Every head's chain in registers; q-tilde is row-uniform, read through scalar loads (16 dims per load, SGPR operand);
// the entries' latent comes from global memory, 16 bf16 per entry per step, prefetched one step ahead. No LDS.
constexpr unsigned kScoreStep = 16, kScoreWaves = 8;
template <class S, unsigned H>
__global__ __launch_bounds__(32 * kScoreWaves) void scores_kernel(typename S::Cache sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                                      const float* qt, float* e_all) {
    static_assert(kSpecOk<S>, "selected_exact: kLat a multiple of 128, kListCap <= 2304");
    constexpr unsigned E = 64 / H;  // list entries per lane: 2 (H = 32) or 1 (H = 64)
    const unsigned r = blockIdx.y, n = n_all[r];
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32))), lane = threadIdx.x % 32;
    const unsigned j0 = (blockIdx.x * kScoreWaves + wave) * 32 * E;
    if (j0 >= n) return;  // uniform per wave
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * S::kListCap;
    const uint4* lat[E];
#pragma unroll
    for (unsigned x = 0; x < E; ++x) {
        const unsigned j = j0 + lane + 32 * x;  // entries past n read position 0's latent (always cached) and store nothing
        lat[x] = reinterpret_cast<const uint4*>(S::latent(sl) + static_cast<std::size_t>(j < n ? tok[j] : 0U) * S::kLat);
    }
    const auto* qr = (ConstF*)(qt + static_cast<std::size_t>(r) * H * S::kLat);  // NOLINT: address-space cast
    float acc[E][H];
#pragma unroll
    for (unsigned x = 0; x < E; ++x)
#pragma unroll
        for (unsigned h = 0; h < H; ++h) acc[x][h] = 0.0F;
    uint4 raw[E][2];
#pragma unroll
    for (unsigned x = 0; x < E; ++x) {
        raw[x][0] = lat[x][0];
        raw[x][1] = lat[x][1];
    }
    for (unsigned l0 = 0; l0 < S::kLat; l0 += kScoreStep) {
        float c[E][kScoreStep];
#pragma unroll
        for (unsigned x = 0; x < E; ++x) {
            const std::uint32_t wv[8] = {raw[x][0].x, raw[x][0].y, raw[x][0].z, raw[x][0].w, raw[x][1].x, raw[x][1].y, raw[x][1].z, raw[x][1].w};
#pragma unroll
            for (unsigned i = 0; i < kScoreStep; ++i) c[x][i] = widen_bf16(static_cast<std::uint16_t>(wv[i / 2] >> (16 * (i % 2))));
        }
        if (l0 + kScoreStep < S::kLat)
#pragma unroll
            for (unsigned x = 0; x < E; ++x) {
                raw[x][0] = lat[x][(l0 + kScoreStep) / 8];
                raw[x][1] = lat[x][(l0 + kScoreStep) / 8 + 1];
            }
#pragma unroll
        for (unsigned h = 0; h < H; ++h) {
            const SVec16 q = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(h) * S::kLat + l0);
#pragma unroll
            for (unsigned i = 0; i < kScoreStep; ++i)
#pragma unroll
                for (unsigned x = 0; x < E; ++x) acc[x][h] = __builtin_fmaf(q[i], c[x][i], acc[x][h]);
        }
    }
    float* e = e_all + static_cast<std::size_t>(r) * H * kEStride<S>;
#pragma unroll
    for (unsigned x = 0; x < E; ++x) {
        const unsigned j = j0 + lane + 32 * x;
        if (j < n)
#pragma unroll
            for (unsigned h = 0; h < H; ++h) e[static_cast<std::size_t>(h) * kEStride<S> + j] = acc[x][h] * S::kScale;
    }
}

// ---------------------------------------------------------------- scores, H = 32, VOPD-paired inner step
// a[2 h + x] = scores_kernel's acc[x][h]: the same chains, two entries of a lane sharing each q SGPR, four heads per block.
#include "attn_core/selected_exact_mla_scores_vopd.inc"
template <class S, unsigned P = 0>
__device__ __forceinline__ void paired_step(float (&a)[64], const float (&c)[32], const ConstF* qr, unsigned l0) {
    const SVec16 qa = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(4 * P) * S::kLat + l0);
    const SVec16 qb = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(4 * P + 1) * S::kLat + l0);
    const SVec16 qc = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(4 * P + 2) * S::kLat + l0);
    const SVec16 qd = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(4 * P + 3) * S::kLat + l0);
    scores_vopd_heads<P>(a, c, qa, qb, qc, qd);
    if constexpr (P + 1 < 8) paired_step<S, P + 1>(a, c, qr, l0);
}
template <class S> inline constexpr unsigned kPairedTiles = (S::kListCap + 32 * kScoreWaves * 2 - 1) / (32 * kScoreWaves * 2);
template <class S, unsigned G>
__global__ __launch_bounds__(32 * kScoreWaves) void scores_paired_kernel(typename S::Cache sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                                      const float* qt, float* e_all, unsigned C) {
    static_assert(kSpecOk<S>, "selected_exact: kLat a multiple of 128, kListCap <= 2304");
    constexpr unsigned H = 32;
    const unsigned idx = blockIdx.x % (G ? G * kPairedTiles<S> : 1U);
    const unsigned r = G ? blockIdx.x / (G * kPairedTiles<S>) * G + idx % G : blockIdx.y, tile = G ? idx / G : blockIdx.x;
    if (r >= C) return;  // uniform per CTA (partial last group)
    const unsigned n = n_all[r];
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32))), lane = threadIdx.x % 32;
    const unsigned j0 = (tile * kScoreWaves + wave) * 64;
    if (j0 >= n) return;  // uniform per wave
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * S::kListCap;
    const uint4* lat[2];
#pragma unroll
    for (unsigned x = 0; x < 2; ++x) {
        const unsigned j = j0 + lane + 32 * x;  // entries past n read position 0's latent (always cached) and store nothing
        lat[x] = reinterpret_cast<const uint4*>(S::latent(sl) + static_cast<std::size_t>(j < n ? tok[j] : 0U) * S::kLat);
    }
    const auto* qr = (ConstF*)(qt + static_cast<std::size_t>(r) * H * S::kLat);  // NOLINT: address-space cast
    float a[64];
#pragma unroll
    for (unsigned i = 0; i < 64; ++i) a[i] = 0.0F;
    uint4 raw[2][2];
#pragma unroll
    for (unsigned x = 0; x < 2; ++x) {
        raw[x][0] = lat[x][0];
        raw[x][1] = lat[x][1];
    }
    for (unsigned l0 = 0; l0 < S::kLat; l0 += kScoreStep) {
        float c[32];  // c[2 i + x] = entry x, dim i
#pragma unroll
        for (unsigned x = 0; x < 2; ++x) {
            const std::uint32_t wv[8] = {raw[x][0].x, raw[x][0].y, raw[x][0].z, raw[x][0].w, raw[x][1].x, raw[x][1].y, raw[x][1].z, raw[x][1].w};
#pragma unroll
            for (unsigned i = 0; i < kScoreStep; ++i) c[2 * i + x] = widen_bf16(static_cast<std::uint16_t>(wv[i / 2] >> (16 * (i % 2))));
        }
        if (l0 + kScoreStep < S::kLat)
#pragma unroll
            for (unsigned x = 0; x < 2; ++x) {
                raw[x][0] = lat[x][(l0 + kScoreStep) / 8];
                raw[x][1] = lat[x][(l0 + kScoreStep) / 8 + 1];
            }
        paired_step<S>(a, c, qr, l0);
    }
    float* e = e_all + static_cast<std::size_t>(r) * H * kEStride<S>;
#pragma unroll
    for (unsigned x = 0; x < 2; ++x) {
        const unsigned j = j0 + lane + 32 * x;
        if (j < n)
#pragma unroll
            for (unsigned h = 0; h < H; ++h) e[static_cast<std::size_t>(h) * kEStride<S> + j] = a[2 * h + x] * S::kScale;
    }
}

// ---------------------------------------------------------------- o-bar, H = 32, lane-major p (softmax_kernel<S, true>)
// CTA = all 32 heads x kLanesDims latent dims of one row; wave w owns heads 4 w .. 4 w + 3, lane the dims d0 + 4 lane .. + 3.
// p is wave-uniform (scalar loads, kRounds contiguous slots per head and k).
constexpr unsigned kLanesDims = 128;  // latent dims per CTA of the lanes o-bar
template <class S> inline constexpr unsigned kLanesTiles = S::kLat / kLanesDims;
template <class S, unsigned Steps>
__global__ __launch_bounds__(256) void obar_lanes_kernel(typename S::Cache sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                          const float* e_all, float* ob) {
    static_assert(kSpecOk<S>, "selected_exact: kLat a multiple of 128, kListCap <= 2304");
    constexpr unsigned Sl = Steps * kRounds<S>, LV = Sl * kLanesDims / 8, LD = (LV + 255) / 256;
    __shared__ alignas(16) float lt[2][Sl][kLanesDims];  // Steps x 36 rounds x 512 B: 36 KB (4), 18 KB (2)
    const unsigned r = blockIdx.y, d0 = blockIdx.x * kLanesDims, n = n_all[r];
    const unsigned tid = threadIdx.x, lane = tid % 32, w = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(tid / 32)));
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * S::kListCap;
    const auto* pw = (ConstF*)(e_all + (static_cast<std::size_t>(r) * 32 + 4 * w) * kEStride<S>);  // NOLINT: address-space cast (heads 4 w ..)
    uint4 lr[LD];
    auto load = [&](unsigned b) {  // batch b (k = Steps b .. Steps b + Steps - 1) into registers
#pragma unroll
        for (unsigned i = 0; i < LD; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < LV) {
                const unsigned slot = x / (kLanesDims / 8), c8 = x % (kLanesDims / 8);
                const unsigned t = bitrev8(Steps * b + slot / kRounds<S>) + 256 * (slot % kRounds<S>);
                lr[i] = t < n ? *reinterpret_cast<const uint4*>(S::latent(sl) + static_cast<std::size_t>(tok[t]) * S::kLat + d0 + 8 * c8)
                              : uint4{0, 0, 0, 0};
            }
        }
    };
    auto store = [&](unsigned buf) {
#pragma unroll
        for (unsigned i = 0; i < LD; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < LV) {
                const unsigned slot = x / (kLanesDims / 8), c8 = x % (kLanesDims / 8);
                const std::uint32_t wv[4] = {lr[i].x, lr[i].y, lr[i].z, lr[i].w};
                float f[8];
#pragma unroll
                for (unsigned ii = 0; ii < 8; ++ii) f[ii] = widen_bf16(static_cast<std::uint16_t>(wv[ii / 2] >> (16 * (ii % 2))));
                float4* dst = reinterpret_cast<float4*>(&lt[buf][slot][8 * c8]);
                dst[0] = float4{f[0], f[1], f[2], f[3]};
                dst[1] = float4{f[4], f[5], f[6], f[7]};
            }
        }
    };
    float stk[4][4][8];  // binary-counter levels 0..7; the root (level 8) is formed by the last merge (k = 255) and stored
    load(0);
    store(0);
    __syncthreads();
    for (unsigned b = 0; b < 256 / Steps; ++b) {
        const unsigned cur = b & 1U;
        if (b + 1 < 256 / Steps) load(b + 1);
        for (unsigned j = 0; j < Steps; ++j) {
            const unsigned k = Steps * b + j, l = bitrev8(k);
            float p[4][kRounds<S>];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned m = 0; m < kRounds<S>; ++m) p[a][m] = pw[a * kEStride<S> + kRounds<S> * k + m];  // padding slots are +0
            float c[4][4];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) c[a][q] = 0.0F;
#pragma unroll
            for (unsigned m = 0; m < kRounds<S>; ++m) {
                if (l + 256 * m < n) {  // uniform; true for a prefix of m
                    const float4 cv4 = *reinterpret_cast<const float4*>(&lt[cur][kRounds<S> * j + m][4 * lane]);
                    const float cv[4] = {cv4.x, cv4.y, cv4.z, cv4.w};
#pragma unroll
                    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) c[a][q] = __builtin_fmaf(p[a][m], cv[q], c[a][q]);
                }
            }
            bool placed = false;
#pragma unroll
            for (unsigned L = 0; L < 8; ++L) {
                if (placed) continue;
                if ((k >> L) & 1U) {
#pragma unroll
                    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) c[a][q] = stk[a][q][L] + c[a][q];
                } else {
#pragma unroll
                    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) stk[a][q][L] = c[a][q];
                    placed = true;
                }
            }
            if (!placed)
#pragma unroll
                for (unsigned a = 0; a < 4; ++a)
                    *reinterpret_cast<float4*>(ob + (static_cast<std::size_t>(r) * 32 + 4 * w + a) * S::kLat + d0 + 4 * lane) =
                        float4{c[a][0], c[a][1], c[a][2], c[a][3]};
        }
        if (b + 1 < 256 / Steps) store(cur ^ 1U);
        __syncthreads();
    }
}

// ---------------------------------------------------------------- o-bar, any heads, row-major p (softmax_kernel<S, false>)
// The row's latent tile [n][kTreeDims] is staged in LDS once; per pass of kTreeHeads heads every lane accumulates its chains.
constexpr unsigned kTreeDims = 8, kTreeHeads = 4, kTreeVals = kTreeDims * kTreeHeads;
template <class S>
__global__ __launch_bounds__(256) void obar_tree_kernel(typename S::Cache sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all, unsigned heads,
                                                          const float* e_all, float* ob) {
    static_assert(kSpecOk<S>, "selected_exact: kLat a multiple of 128, kListCap <= 2304");
    __shared__ uint4 lt[S::kListCap];             // latent dims [lb, lb + 8) of each listed token (16 B)
    __shared__ float buf[kTreeVals][128];
    const unsigned lb = blockIdx.x * kTreeDims, r = blockIdx.y, j = threadIdx.x, n = n_all[r];
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * S::kListCap;
    for (unsigned t = j; t < n; t += 256) lt[t] = *reinterpret_cast<const uint4*>(S::latent(sl) + static_cast<std::size_t>(tok[t]) * S::kLat + lb);
    for (unsigned h0 = 0; h0 < heads; h0 += kTreeHeads) {
        float acc[kTreeHeads][kTreeDims];
#pragma unroll
        for (unsigned a = 0; a < kTreeHeads; ++a)
#pragma unroll
            for (unsigned c = 0; c < kTreeDims; ++c) acc[a][c] = 0.0F;
        __syncthreads();  // lt staged (first pass) / buf free (later passes)
        for (unsigned t = j; t < n; t += 256) {
            const uint4 v = lt[t];
            const std::uint32_t wv[4] = {v.x, v.y, v.z, v.w};
            float p[kTreeHeads];
#pragma unroll
            for (unsigned a = 0; a < kTreeHeads; ++a) p[a] = e_all[(static_cast<std::size_t>(r) * heads + h0 + a) * kEStride<S> + t];
#pragma unroll
            for (unsigned c = 0; c < kTreeDims; ++c) {
                const float cv = widen_bf16(static_cast<std::uint16_t>(wv[c / 2] >> (16 * (c % 2))));
#pragma unroll
                for (unsigned a = 0; a < kTreeHeads; ++a) acc[a][c] = __builtin_fmaf(p[a], cv, acc[a][c]);
            }
        }
        for (unsigned hh = 128; hh >= 32; hh >>= 1U) {  // cross-wave levels: lanes [hh, 2hh) hand their values to l - hh
            if (j >= hh && j < 2 * hh)
#pragma unroll
                for (unsigned a = 0; a < kTreeHeads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kTreeDims; ++c) buf[a * kTreeDims + c][j - hh] = acc[a][c];
            __syncthreads();
            if (j < hh)
#pragma unroll
                for (unsigned a = 0; a < kTreeHeads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kTreeDims; ++c) acc[a][c] = acc[a][c] + buf[a * kTreeDims + c][j];
            __syncthreads();
        }
        if (j < 32) {
#pragma unroll
            for (unsigned hh = 16; hh > 0; hh >>= 1U)
#pragma unroll
                for (unsigned a = 0; a < kTreeHeads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kTreeDims; ++c) acc[a][c] = acc[a][c] + __shfl_down(acc[a][c], hh, 32);
            if (j == 0)
#pragma unroll
                for (unsigned a = 0; a < kTreeHeads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kTreeDims; ++c) ob[(static_cast<std::size_t>(r) * heads + h0 + a) * S::kLat + lb + c] = acc[a][c];
        }
    }
}

}  // namespace strix::attn_core::selected_exact
