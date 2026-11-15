// Shared CUDA core: FP16 tensor-core tile mechanics for FAST forms (sm_80+, built for sm_120).
// A = activations (FP32 in global, FP16 in smem), B = quantised weights (int8 or int4 codes + one FP32 scale per 32 K),
// accumulation FP32 (mma.sync m16n8k16 f16). Codes are exact in FP16; the scale is applied to the FP32 block partial:
//   acc += scale * (sum over the 32-K block of act * code)     -> the weights are never rounded.
//
// K permutation "perm8" (A and B agree, no runtime cost). Inside every 32-wide K block (the scale block) and for MMA step
// j in {0,1}, the thread t (lane & 3) holds the actual K indices 8t + 4j + {0,1} in MMA slots 2t, 2t+1 and 8t + 4j + {2,3} in
// slots 8 + 2t, 9 + 2t. So one thread's 8 consecutive codes of a weight row (uint2 of int8 / one uint32 of nibbles)
// feed both k16 steps of the block, and A is stored to smem in permuted order by stage_k4().
//
// Fragment map (PTX m16n8k16, g = lane >> 2, t = lane & 3): a0 (row g, slots 2t..2t+1), a1 (row g+8), a2 (row g, slots 8+2t..),
// a3 (row g+8); b0 (slots 2t..2t+1, col g), b1 (slots 8+2t.., col g); c0,c1 (row g, cols 2t, 2t+1), c2,c3 (row g+8).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "ptx.cuh"

namespace strix::cuda::core::mma {

constexpr unsigned kPadHalves = 8;  // smem row pad (16 B): ldmatrix rows hit distinct banks when pitch/8 is odd

// FP32 x4 (actual K = 4m .. 4m+3 of a 32-block, m in 0..7) -> FP16, stored in perm8 order into the 32-half block at `blk`.
__device__ __forceinline__ void stage_k4(__half* blk, unsigned m, float4 v) {
    const unsigned t = m >> 1U, j = m & 1U;
    *reinterpret_cast<__half2*>(blk + j * 16U + 2U * t) = __floats2half2_rn(v.x, v.y);
    *reinterpret_cast<__half2*>(blk + j * 16U + 8U + 2U * t) = __floats2half2_rn(v.z, v.w);
}

// A 16x16 fragment (16 rows, 16 halves = one MMA step) at `tile` (row 0, first slot), row pitch in halves.
__device__ __forceinline__ void ldsm_a(unsigned (&a)[4], const __half* tile, unsigned pitch_halves, unsigned lane) {
    ptx::ldsm_x4(a, tile + (lane & 15U) * pitch_halves + (lane >> 4U) * 8U);
}

// Exact integer -> half2 unpack (magic-number trick: 0x6400 | u8 = 1024 + u8).
__device__ __forceinline__ unsigned i8_pair_to_half2(unsigned w, unsigned sel) {  // sel: 0x4140 (bytes 0,1) or 0x4342 (bytes 2,3)
    const unsigned v = __byte_perm(w ^ 0x80808080U, 0x64646464U, sel);  // {1024 + (c0^0x80), 1024 + (c1^0x80)}
    const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&v), __float2half2_rn(1152.0F));
    return *reinterpret_cast<const unsigned*>(&r);
}
__device__ __forceinline__ unsigned u4_pair_to_half2(unsigned w, unsigned shift) {  // nibbles at bit shift, shift + 4; offset 8
    const unsigned v = (((w >> shift) & 0xFU) | (((w >> (shift + 4U)) & 0xFU) << 16U)) | 0x64006400U;
    const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&v), __float2half2_rn(1032.0F));
    return *reinterpret_cast<const unsigned*>(&r);
}

// Code traits: W words hold the thread's 8 consecutive codes (k 8t .. 8t+7 of the 32-block) of one weight column n = g.
struct Int8Codes {  // signed int8 (Q8_0 style), 2 words, byte i of word 0 = k 8t + i, word 1 = k 8t + 4 + i
    static constexpr unsigned W = 2;
    __device__ static __forceinline__ void b_frag(const unsigned (&w)[W], unsigned step, unsigned& b0, unsigned& b1) {
        b0 = i8_pair_to_half2(w[step], 0x4140U);
        b1 = i8_pair_to_half2(w[step], 0x4342U);
    }
};
struct Int4Codes {  // unsigned nibbles with offset 8 (Q4_0 style value = nibble - 8), 1 word, nibble i = k 8t + i
    static constexpr unsigned W = 1;
    __device__ static __forceinline__ void b_frag(const unsigned (&w)[W], unsigned step, unsigned& b0, unsigned& b1) {
        b0 = u4_pair_to_half2(w[0], 16U * step);
        b1 = u4_pair_to_half2(w[0], 16U * step + 8U);
    }
};

// One 32-K block for MT m16 tiles x one n8 tile. a[mt][step] are the A fragments (ldsm_a of the 2 steps of the block),
// w the thread's codes of column g, s0/s1 the FP32 scales of the columns 2t and 2t+1 (c-fragment columns).
template <unsigned MT, class Codes>
__device__ __forceinline__ void warp_gemm_k32(float (&acc)[MT][4], const unsigned (&a)[MT][2][4], const unsigned (&w)[Codes::W],
    float s0, float s1) {
    unsigned b[2][2];
    Codes::b_frag(w, 0U, b[0][0], b[0][1]);
    Codes::b_frag(w, 1U, b[1][0], b[1][1]);
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt) {
        float blk[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        ptx::mma_f16(blk, a[mt][0][0], a[mt][0][1], a[mt][0][2], a[mt][0][3], b[0][0], b[0][1]);
        ptx::mma_f16(blk, a[mt][1][0], a[mt][1][1], a[mt][1][2], a[mt][1][3], b[1][0], b[1][1]);
        acc[mt][0] = __fmaf_rn(s0, blk[0], acc[mt][0]);
        acc[mt][1] = __fmaf_rn(s1, blk[1], acc[mt][1]);
        acc[mt][2] = __fmaf_rn(s0, blk[2], acc[mt][2]);
        acc[mt][3] = __fmaf_rn(s1, blk[3], acc[mt][3]);
    }
}

}  // namespace strix::cuda::core::mma
