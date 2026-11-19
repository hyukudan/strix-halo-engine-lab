#pragma once
// Shared EXACT decode scan of the compressed group indexer (contract strix/attn/group_indexer.hpp, score (c)):
// for every visible pooled group key p of a row r: acc = +0; for h ascending: dot = fma chain over i ascending from +0;
// acc = fma(w_h, relu(scaled(dot)), acc); key = key_of(acc) (score_key order) written to keys[r * key_stride + p], plus
// the row's top-byte histogram hist0[r * 256 + key >> 24] that the radix select (select.hpp) starts from.
// This is group_indexer::score_group, i.e. bit-identical to the host oracle; the WMMA core (index_score_wmma.hpp) is
// the FAST prefill form. Lifted from GLM's decode family (rg1 / rg2 / rk1 / rk2); GLM binds it with fp32 keys,
// 32 heads, InsideRelu; or bf16 keys, 16 heads, InHeadWeight.
//
// Row view V (an engine adapter, all static):
//   using Rows;  using Key;  (float: fp32 keys | std::uint16_t: bf16 bits)   static constexpr unsigned kHeads, kDim;
//   unsigned rows(const Rows&); const Key* keys(const Rows&, r); unsigned pools(const Rows&, r)  (visible groups)
//   float weight(float raw)  (w_h from the raw head weight);  float scaled(float dot)  (inside the relu);
//   std::uint32_t key_of(float score).
// Variants (same bits, scheduling only): score_kernel (rg1: one row per CTA, query in LDS), score_rg_kernel<RG = 2>
// (two rows per CTA, pass 0 shares the first slot's key loads), score_rk_kernel<1 | 2> (register-blocked: lane owns two
// groups, query from uniform loads). The engine's launcher picks by its measured selector.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::group_scan {

constexpr unsigned kThreads = 64;

// Elements 4 i4 .. 4 i4 + 3 of one group key row, as float.
template <class Key> __device__ __forceinline__ float4 load4(const Key* row, unsigned i4) {
    if constexpr (sizeof(Key) == 4) {
        return reinterpret_cast<const float4*>(row)[i4];
    } else {
        static_assert(sizeof(Key) == 2, "group keys are fp32 or bf16 bits");
        const uint2 b = reinterpret_cast<const uint2*>(row)[i4];
        return float4{__builtin_bit_cast(float, b.x << 16U), __builtin_bit_cast(float, b.x & 0xFFFF0000U),
                      __builtin_bit_cast(float, b.y << 16U), __builtin_bit_cast(float, b.y & 0xFFFF0000U)};
    }
}

template <class V>
__global__ __launch_bounds__(kThreads) void score_kernel(typename V::Rows d, const float* qI, const float* wraw,
                                                         std::uint32_t* keys, unsigned key_stride, std::uint32_t* hist0) {
    __shared__ float4 qs[V::kHeads * V::kDim / 4];
    __shared__ float wh[V::kHeads];
    __shared__ unsigned lh[256];
    const unsigned r = blockIdx.y, l = threadIdx.x;
    const unsigned P = V::pools(d, r), p0 = blockIdx.x * kThreads;
    if (p0 >= P) return;
    const float4* q4 = reinterpret_cast<const float4*>(qI + static_cast<std::size_t>(r) * V::kHeads * V::kDim);
    for (unsigned e = l; e < V::kHeads * V::kDim / 4; e += kThreads) qs[e] = q4[e];
    if (l < V::kHeads) wh[l] = V::weight(wraw[static_cast<std::size_t>(r) * V::kHeads + l]);
    for (unsigned e = l; e < 256; e += kThreads) lh[e] = 0;
    __syncthreads();
    const unsigned p = p0 + l;
    if (p < P) {
        const typename V::Key* K = V::keys(d, r) + static_cast<std::size_t>(p) * V::kDim;
        float dot[V::kHeads];
#pragma unroll
        for (unsigned h = 0; h < V::kHeads; ++h) dot[h] = 0.0F;
        for (unsigned i4 = 0; i4 < V::kDim / 4; ++i4) {
            const float4 kv = load4(K, i4);
#pragma unroll
            for (unsigned h = 0; h < V::kHeads; ++h) {
                const float4 q = qs[h * (V::kDim / 4) + i4];
                dot[h] = __builtin_fmaf(q.x, kv.x, dot[h]);
                dot[h] = __builtin_fmaf(q.y, kv.y, dot[h]);
                dot[h] = __builtin_fmaf(q.z, kv.z, dot[h]);
                dot[h] = __builtin_fmaf(q.w, kv.w, dot[h]);
            }
        }
        float acc = 0.0F;
#pragma unroll
        for (unsigned h = 0; h < V::kHeads; ++h) {
            const float sc = V::scaled(dot[h]);
            acc = __builtin_fmaf(wh[h], sc > 0.0F ? sc : 0.0F, acc);
        }
        const std::uint32_t key = V::key_of(acc);
        keys[static_cast<std::size_t>(r) * key_stride + p] = key;
        atomicAdd(&lh[key >> 24U], 1U);
    }
    __syncthreads();
    for (unsigned e = l; e < 256; e += kThreads)
        if (lh[e]) atomicAdd(&hist0[r * 256 + e], lh[e]);
}

// RG=2 uses 35,072 bytes of LDS at 32 heads; RG=4 would exceed 64 KiB.
template <class V, unsigned RG>
__global__ __launch_bounds__(kThreads) void score_rg_kernel(typename V::Rows d, const float* qI, const float* wraw,
                                                            std::uint32_t* keys, unsigned key_stride, std::uint32_t* hist0) {
    static_assert(RG == 2, "indexer row grouping must fit in 64 KiB LDS");
    __shared__ float4 qs[RG][V::kHeads * V::kDim / 4];
    __shared__ float wh[RG][V::kHeads];
    __shared__ unsigned lh[RG][256];
    static_assert(sizeof(qs) + sizeof(wh) + sizeof(lh) <= 65536, "indexer LDS budget");
    const unsigned g0 = blockIdx.y * RG, l = threadIdx.x;
    const unsigned p = blockIdx.x * kThreads + l;
    const typename V::Key* pooled[RG];
    unsigned pools[RG];
#pragma unroll
    for (unsigned u = 0; u < RG; ++u) {
        const unsigned r = g0 + u;
        pooled[u] = r < V::rows(d) ? V::keys(d, r) : nullptr;
        pools[u] = r < V::rows(d) ? V::pools(d, r) : 0U;
        if (r < V::rows(d)) {
            const float4* q4 = reinterpret_cast<const float4*>(qI + static_cast<std::size_t>(r) * V::kHeads * V::kDim);
            for (unsigned e = l; e < V::kHeads * V::kDim / 4; e += kThreads) qs[u][e] = q4[e];
            if (l < V::kHeads) wh[u][l] = V::weight(wraw[static_cast<std::size_t>(r) * V::kHeads + l]);
        }
        for (unsigned e = l; e < 256; e += kThreads) lh[u][e] = 0;
    }
    __syncthreads();
    // Pass 0 shares the first slot's keys. Other slots retain their own key loads.
#pragma unroll
    for (unsigned pass = 0; pass < RG; ++pass) {
        bool active[RG];
        bool any = false;
#pragma unroll
        for (unsigned u = 0; u < RG; ++u) {
            active[u] = p < pools[u] &&
                        (pass == 0 ? pooled[u] == pooled[0] : u == pass && pooled[u] != pooled[0]);
            any = any || active[u];
        }
        if (any) {
            const typename V::Key* K = pooled[pass] + static_cast<std::size_t>(p) * V::kDim;
            float dot[RG][V::kHeads];
#pragma unroll
            for (unsigned u = 0; u < RG; ++u)
#pragma unroll
                for (unsigned h = 0; h < V::kHeads; ++h) dot[u][h] = 0.0F;
            for (unsigned i4 = 0; i4 < V::kDim / 4; ++i4) {
                const float4 kv = load4(K, i4);
#pragma unroll
                for (unsigned u = 0; u < RG; ++u) {
                    if (active[u]) {
#pragma unroll
                        for (unsigned h = 0; h < V::kHeads; ++h) {
                            const float4 q = qs[u][h * (V::kDim / 4) + i4];
                            dot[u][h] = __builtin_fmaf(q.x, kv.x, dot[u][h]);
                            dot[u][h] = __builtin_fmaf(q.y, kv.y, dot[u][h]);
                            dot[u][h] = __builtin_fmaf(q.z, kv.z, dot[u][h]);
                            dot[u][h] = __builtin_fmaf(q.w, kv.w, dot[u][h]);
                        }
                    }
                }
            }
#pragma unroll
            for (unsigned u = 0; u < RG; ++u) {
                if (active[u]) {
                    float acc = 0.0F;
#pragma unroll
                    for (unsigned h = 0; h < V::kHeads; ++h) {
                        const float sc = V::scaled(dot[u][h]);
                        acc = __builtin_fmaf(wh[u][h], sc > 0.0F ? sc : 0.0F, acc);
                    }
                    const std::uint32_t key = V::key_of(acc);
                    keys[static_cast<std::size_t>(g0 + u) * key_stride + p] = key;
                    atomicAdd(&lh[u][key >> 24U], 1U);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (unsigned u = 0; u < RG; ++u)
        if (g0 + u < V::rows(d))
            for (unsigned e = l; e < 256; e += kThreads)
                if (lh[u][e]) atomicAdd(&hist0[(g0 + u) * 256 + e], lh[u][e]);
}

// Register-blocked: RK_ROWS = 2 or 1. 64 lanes cover 128 groups: lane l owns p0+l and p0+l+64. Only row histograms use
// LDS. The two-row form has 4 x kHeads dot accumulators/lane (one-row: 2 x kHeads); review the ISA for scalar query
// loads, VGPR allocation and spills; RK_ROWS = 1 is the lower-register option.
template <class V, unsigned RK_ROWS>
__global__ __launch_bounds__(kThreads) void score_rk_kernel(typename V::Rows d, const float* qI, const float* wraw,
                                                            std::uint32_t* keys, unsigned key_stride, std::uint32_t* hist0) {
    static_assert(RK_ROWS == 1 || RK_ROWS == 2, "indexer register-blocked rows");
    static_assert(V::kHeads % 2 == 0, "two-head query microtiles");
    __shared__ unsigned lh[RK_ROWS][256];
    const unsigned g0 = blockIdx.y * RK_ROWS, l = threadIdx.x;
    const unsigned p0 = blockIdx.x * (2 * kThreads);
    const typename V::Key* pooled[RK_ROWS];
    unsigned pools[RK_ROWS];
    unsigned max_pools = 0;
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u) {
        const unsigned r = g0 + u;
        pooled[u] = r < V::rows(d) ? V::keys(d, r) : nullptr;
        pools[u] = r < V::rows(d) ? V::pools(d, r) : 0U;
        max_pools = pools[u] > max_pools ? pools[u] : max_pools;
    }
    if (p0 >= max_pools) return;  // CTA-uniform, including zero-pool rows
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u)
        for (unsigned e = l; e < 256; e += kThreads) lh[u][e] = 0;
    __syncthreads();
    float dot[RK_ROWS][2][V::kHeads];
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned h = 0; h < V::kHeads; ++h) dot[u][j][h] = 0.0F;
    // For a shared slot, the first row's key loads cover the union of both P bounds.
    unsigned first_pools = pools[0];
    if constexpr (RK_ROWS == 2)
        if (pooled[1] == pooled[0]) first_pools = max_pools;
    for (unsigned i4 = 0; i4 < V::kDim / 4; ++i4) {
        float4 kv[2];
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned p = p0 + l + j * kThreads;
            kv[j] = float4{};
            if (p < first_pools) kv[j] = load4(pooled[0] + static_cast<std::size_t>(p) * V::kDim, i4);
        }
#pragma unroll
        for (unsigned u = 0; u < RK_ROWS; ++u) {
            if (p0 < pools[u]) {  // uniform: never predicate query loads on a lane's key validity
                float4 row_kv[2];
#pragma unroll
                for (unsigned j = 0; j < 2; ++j) {
                    row_kv[j] = kv[j];
                    if (pooled[u] != pooled[0]) {
                        const unsigned p = p0 + l + j * kThreads;
                        row_kv[j] = float4{};
                        if (p < pools[u]) row_kv[j] = load4(pooled[u] + static_cast<std::size_t>(p) * V::kDim, i4);
                    }
                }
                // Uniform addresses permit s_load: no lane index or lane-dependent branch.
                // Two-head microtiles, double-buffered: prefetch the next pair before using
                // the current pair, then rotate. Each query float4 serves both keys.
                const float4* q4 = reinterpret_cast<const float4*>(qI + static_cast<std::size_t>(g0 + u) * V::kHeads * V::kDim);
                float4 qa = q4[i4], qb = q4[V::kDim / 4 + i4];
#pragma unroll
                for (unsigned h = 0; h < V::kHeads; h += 2) {
                    float4 next_a{}, next_b{};
                    if (h + 2 < V::kHeads) {
                        next_a = q4[(h + 2) * (V::kDim / 4) + i4];
                        next_b = q4[(h + 3) * (V::kDim / 4) + i4];
                    }
#pragma unroll
                    for (unsigned j = 0; j < 2; ++j) {
                        dot[u][j][h] = __builtin_fmaf(qa.x, row_kv[j].x, dot[u][j][h]);
                        dot[u][j][h] = __builtin_fmaf(qa.y, row_kv[j].y, dot[u][j][h]);
                        dot[u][j][h] = __builtin_fmaf(qa.z, row_kv[j].z, dot[u][j][h]);
                        dot[u][j][h] = __builtin_fmaf(qa.w, row_kv[j].w, dot[u][j][h]);
                        dot[u][j][h + 1] = __builtin_fmaf(qb.x, row_kv[j].x, dot[u][j][h + 1]);
                        dot[u][j][h + 1] = __builtin_fmaf(qb.y, row_kv[j].y, dot[u][j][h + 1]);
                        dot[u][j][h + 1] = __builtin_fmaf(qb.z, row_kv[j].z, dot[u][j][h + 1]);
                        dot[u][j][h + 1] = __builtin_fmaf(qb.w, row_kv[j].w, dot[u][j][h + 1]);
                    }
                    qa = next_a;
                    qb = next_b;
                }
            }
        }
    }
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u) {
        if (p0 < pools[u]) {
            float acc[2] = {0.0F, 0.0F};
#pragma unroll
            for (unsigned h = 0; h < V::kHeads; ++h) {
                const float wh = V::weight(wraw[static_cast<std::size_t>(g0 + u) * V::kHeads + h]);
#pragma unroll
                for (unsigned j = 0; j < 2; ++j) {
                    const float sc = V::scaled(dot[u][j][h]);
                    acc[j] = __builtin_fmaf(wh, sc > 0.0F ? sc : 0.0F, acc[j]);
                }
            }
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                const unsigned p = p0 + l + j * kThreads;
                if (p < pools[u]) {
                    const std::uint32_t key = V::key_of(acc[j]);
                    keys[static_cast<std::size_t>(g0 + u) * key_stride + p] = key;
                    atomicAdd(&lh[u][key >> 24U], 1U);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u)
        if (g0 + u < V::rows(d))
            for (unsigned e = l; e < 256; e += kThreads)
                if (lh[u][e]) atomicAdd(&hist0[(g0 + u) * 256 + e], lh[u][e]);
}

}  // namespace strix::attn_core::group_scan
