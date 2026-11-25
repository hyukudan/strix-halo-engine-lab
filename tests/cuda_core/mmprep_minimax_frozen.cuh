#pragma once

// Frozen MiniMax RTX prep oracle from strix-halo-engine-lab e123e646f.
// Source: src/cuda/models/minimax/mm_rtx_kernels.cu:22-26,46-52,79-151.
// Descriptor definitions are from mm_rtx_kernels.cuh and mm_rtx_select.hpp.

// Keep this implementation independent of the shared chain.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace mmprep_frozen {

constexpr unsigned kHeadDim = 128;
constexpr float kEps = 1e-6F;

struct SeqView {
    std::uint32_t row0, rows, cap;
    std::uint16_t *k, *v, *ik;
    std::uint32_t pos0 = 0;
};

struct RopeRow {
    float c[32], s[32];
};

struct MsaLayer {
    unsigned layer;
    const float *q_w, *k_w, *iq_w, *ik_w;  // 128 Gemma weights each
    std::size_t index_layer_stride; // BF16 elements in one layer of an index-key slot
};

__device__ __forceinline__ std::uint32_t f2u(float x) { return __float_as_uint(x); }
__device__ __forceinline__ std::uint16_t float_to_bf16(float x) {  // RNE (finite inputs), strix/minimax_msa.hpp
    const std::uint32_t b = f2u(x);
    return static_cast<std::uint16_t>((b + 0x7FFFU + ((b >> 16U) & 1U)) >> 16U);
}

__device__ __forceinline__ float shx(float v, int d) { return __shfl_xor_sync(0xffffffffU, v, d, 32); }
// Butterfly of strix/minimax_msa.hpp (v = v + v[lane ^ d], d = 16..1): every lane ends with lanes[0]'s value.
__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v = v + shx(v, d);
    return v;
}

// head_norm_rope on one 128 head held by a warp (lane l: dims 4l..4l+3). Returns the lane's 4 values.
__device__ void head_norm_rope_warp(float (&x)[4], const float* w, const RopeRow& rope, unsigned lane) {
    float acc = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = __fmaf_rn(x[i], x[i], acc);
    acc = butterfly(acc);
    const float r = 1.0F / sqrtf(acc / 128.0F + kEps);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = (x[i] * r) * (1.0F + w[4 * lane + i]);
    float partner[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) partner[i] = __shfl_sync(0xffffffffU, x[i], lane < 8 ? lane + 8 : (lane < 16 ? lane - 8 : lane), 32);
    if (lane < 8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned d = 4 * lane + i;
            x[i] = __fmaf_rn(-partner[i], rope.s[d], x[i] * rope.c[d]);  // a' = fmaf(-b, sin, a cos)
        }
    } else if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned d = 4 * (lane - 8) + i;
            x[i] = __fmaf_rn(partner[i], rope.s[d], x[i] * rope.c[d]);  // b' = fmaf(a, sin, b cos)
        }
    }
}

// Prep: grid (rows, 73 heads: 64 q, 4 k, 4 iq, 1 ik) + 4 v heads copied; one warp per block.
__global__ void msa_prep_kernel(const SeqView* seqs, const std::uint32_t* row_seq, MsaLayer L, float* q, const float* k, const float* v,
                                float* iq, const float* ik, const RopeRow* rope) {
    const std::size_t row = blockIdx.x;
    const unsigned head = blockIdx.y, lane = threadIdx.x;
    const SeqView sq = seqs[row_seq[row]];
    const std::uint64_t pos = row - sq.row0 + sq.pos0;
    const RopeRow& rr = rope[pos];
    float x[4];
    if (head < 64) {
        float* p = q + (row * 64 + head) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = p[i];
        head_norm_rope_warp(x, L.q_w, rr, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = x[i];
    } else if (head < 68) {
        const unsigned g = head - 64;
        const float* p = k + (row * 4 + g) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = p[i];
        head_norm_rope_warp(x, L.k_w, rr, lane);
        std::uint16_t* c = sq.k + ((static_cast<std::size_t>(L.layer) * sq.cap + pos) * 4 + g) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = float_to_bf16(x[i]);
        const float* pv = v + (row * 4 + g) * 128 + 4 * lane;
        std::uint16_t* cv = sq.v + ((static_cast<std::size_t>(L.layer) * sq.cap + pos) * 4 + g) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) cv[i] = float_to_bf16(pv[i]);
    } else if (head < 72) {
        float* p = iq + (row * 4 + (head - 68)) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = p[i];
        head_norm_rope_warp(x, L.iq_w, rr, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = x[i];
    } else {
        const float* p = ik + row * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = p[i];
        head_norm_rope_warp(x, L.ik_w, rr, lane);
        std::uint16_t* c = sq.ik + L.layer * L.index_layer_stride + pos * kHeadDim + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = float_to_bf16(x[i]);
    }
}

}  // namespace mmprep_frozen
