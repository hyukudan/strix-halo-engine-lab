#pragma once
// THE NVFP4 activation quantiser of the sm_120 FAST4 family (Home: shared, backend core CUDA, arch specialisation sm_120a:
// cvt e2m1x2 / e4m3). One rule for every producer: the tile GEMM X records (nvfp4_tile.cuh) and the grouped expert image
// (grouped_w4a4.cuh) differ only in where they store the codes. Contract: strix/cuda/core/nvfp4_tile.hpp (activation
// quantisation) and moe_core/grouped_w4a4.hpp. Scale rules (default nearest@1; mse@1 opt-in per producer):
// moe_core/nvfp4_act_rule.hpp.
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "moe_core/nvfp4_act_rule.hpp"

namespace strix::cuda::core::nvfp4::tile::sm120 {

// THE activation quantiser of the family (one rule for every producer): 16 values, row scale ts -> e4m3 group scale (returned;
// a16 / 6 / ts, saturated at 448) and 16 e2m1 codes (v / (scale * ts), clamped to +-6, round to nearest even) in canonical byte
// order (word 0 = values 0..7, word 1 = values 8..15, low nibble = even value). Every producer (FP32 rows, BF16 rows, the norm
// folds, the grouped expert image of moe_core/grouped_w4a4.hpp) hands this core the group's 16 FP32 values; only the store differs.
__device__ __forceinline__ std::uint8_t encode_values(const float (&v)[16], float ts, std::uint32_t (&word)[2]) {
    float a16 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a16 = fmaxf(a16, fabsf(v[i]));
    __nv_fp8_e4m3 sc;
    sc.__x = 0;
    float scale = 0.f;
    if (ts > 0.f) {
        sc.__x = __nv_cvt_float_to_fp8(fminf(448.f, __fdiv_rn(__fdiv_rn(a16, 6.f), ts)), __NV_SATFINITE, __NV_E4M3);
        scale = __fmul_rn(static_cast<float>(sc), ts);
    }
    word[0] = word[1] = 0u;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        const float u = scale != 0.f ? fminf(6.f, fmaxf(-6.f, __fdiv_rn(v[2 * p], scale))) : 0.f;
        const float w = scale != 0.f ? fminf(6.f, fmaxf(-6.f, __fdiv_rn(v[2 * p + 1], scale))) : 0.f;
        const std::uint32_t b = __nv_cvt_float2_to_fp4x2(make_float2(u, w), __NV_E2M1, cudaRoundNearest);
        word[p >> 2] |= (b & 0xffu) << (8 * (p & 3));
    }
    return sc.__x;
}
// mse@1 scale rule (moe_core/nvfp4_act_rule.hpp): the nearest code of encode_values, then the candidate search; the codes
// follow the same rule as encode_values at the chosen scale. encode_values itself is left as it is, so every nearest@1 producer
// keeps its machine code.
__device__ __forceinline__ std::uint8_t encode_values_mse(const float (&v)[16], float ts, std::uint32_t (&word)[2]) {
    float a16 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a16 = fmaxf(a16, fabsf(v[i]));
    std::uint8_t code = 0u;
    float scale = 0.f;
    if (ts > 0.f) {
        const std::uint8_t c0 = __nv_cvt_float_to_fp8(fminf(448.f, __fdiv_rn(__fdiv_rn(a16, 6.f), ts)), __NV_SATFINITE, __NV_E4M3);
        code = ::strix::moe_core::nvfp4_act::search_scale_code(v, ts, c0);
        __nv_fp8_e4m3 sc;
        sc.__x = code;
        scale = __fmul_rn(static_cast<float>(sc), ts);
    }
    word[0] = word[1] = 0u;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        const float u = scale != 0.f ? fminf(6.f, fmaxf(-6.f, __fdiv_rn(v[2 * p], scale))) : 0.f;
        const float w = scale != 0.f ? fminf(6.f, fmaxf(-6.f, __fdiv_rn(v[2 * p + 1], scale))) : 0.f;
        const std::uint32_t b = __nv_cvt_float2_to_fp4x2(make_float2(u, w), __NV_E2M1, cudaRoundNearest);
        word[p >> 2] |= (b & 0xffu) << (8 * (p & 3));
    }
    return code;
}
__device__ __forceinline__ void load_group(const float4* g4, float (&v)[16]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 f = g4[i];
        v[4 * i] = f.x;
        v[4 * i + 1] = f.y;
        v[4 * i + 2] = f.z;
        v[4 * i + 3] = f.w;
    }
}

// Row element types of quantize_kernel: FP32, or BF16 bit patterns widened exactly (bf16 b -> float bits b << 16), so a BF16 row
// quantises bit for bit as its FP32 widening. Both load 16 bytes per access (4 FP32 / 8 BF16 values).
__device__ __forceinline__ float bf16_lo(std::uint32_t w) { return __uint_as_float(w << 16); }          // element 2i (little endian)
__device__ __forceinline__ float bf16_hi(std::uint32_t w) { return __uint_as_float(w & 0xffff0000u); }  // element 2i + 1
__device__ __forceinline__ float abs_max16(const float* row, std::uint32_t i) {  // |max| of 16-byte vector i
    const float4 v = reinterpret_cast<const float4*>(row)[i];
    return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}
__device__ __forceinline__ float abs_max16(const std::uint16_t* row, std::uint32_t i) {
    const uint4 u = reinterpret_cast<const uint4*>(row)[i];
    const float a = fmaxf(fmaxf(fabsf(bf16_lo(u.x)), fabsf(bf16_hi(u.x))), fmaxf(fabsf(bf16_lo(u.y)), fabsf(bf16_hi(u.y))));
    const float b = fmaxf(fmaxf(fabsf(bf16_lo(u.z)), fabsf(bf16_hi(u.z))), fmaxf(fabsf(bf16_lo(u.w)), fabsf(bf16_hi(u.w))));
    return fmaxf(a, b);
}
// The 16 values of group q of a row (FP32, or BF16 widened exactly).
__device__ __forceinline__ void load_row_group(const float* row, std::uint32_t q, float (&v)[16]) {
    load_group(reinterpret_cast<const float4*>(row) + 4 * q, v);
}
__device__ __forceinline__ void load_row_group(const std::uint16_t* row, std::uint32_t q, float (&v)[16]) {
    const uint4* g = reinterpret_cast<const uint4*>(row) + 2 * q;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint4 u = g[i];
        const std::uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = bf16_lo(w[j]);
            v[8 * i + 2 * j + 1] = bf16_hi(w[j]);
        }
    }
}
}  // namespace strix::cuda::core::nvfp4::tile::sm120
