#pragma once

// Device helpers shared by the deepseek41 CUDA expert kernels: exact FP16 decode, IQ2_XXS sign handling, the
// declared butterfly and the A8 epilogue. Compiled with --fmad=false (no contraction of the declared
// FP32 operations), IEEE division and denormals.

#include "strix/cuda_ds41_moe.hpp"
#include "strix/ds41_moe.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace strix::ds41::cuda::moe_detail {

constexpr unsigned kFull = 0xffffffffU;

// FP16 bits to FP32, exactly as the host's half_to_float (subnormals, infinities and NaN payloads).
__device__ __forceinline__ float half_bits(std::uint32_t bits) {
    const std::uint32_t sign = (bits & 0x8000U) << 16U, exponent = (bits >> 10U) & 0x1fU, mantissa = bits & 0x3ffU;
    if (exponent == 0U) {
        const float magnitude = static_cast<float>(mantissa) * 5.9604644775390625e-8F;  // exact: m * 2^-24
        return __uint_as_float(__float_as_uint(magnitude) | sign);
    }
    if (exponent == 31U) return __uint_as_float(sign | 0x7f800000U | (mantissa << 13U));
    return __uint_as_float(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

// IQ2_XXS: byte masks (0xff = negative) of four sign bits, and the signed codes of four grid magnitudes.
__device__ __forceinline__ std::uint32_t sign_mask4(std::uint32_t bits4) {
    return ((bits4 * 0x00204081U) & 0x01010101U) * 0xffU;
}
__device__ __forceinline__ std::uint32_t apply_signs(std::uint32_t magnitudes, std::uint32_t mask) {
    return (magnitudes ^ mask) + (mask & 0x01010101U);
}
// Sign byte of a 7-bit index: the index with its parity as bit 7 (iq2_xxs_sign_byte).
__device__ __forceinline__ std::uint32_t sign_byte(std::uint32_t index) {
    index &= 127U;
    return index | ((__popc(index) & 1U) << 7U);
}

// Butterfly of the declared product: v[l] = v[l] + v[l ^ d], d = 16 .. 1; lane 0 holds the reference result.
__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (int d = 16; d >= 1; d /= 2) v = v + __shfl_xor_sync(kFull, v, d);
    return v;
}

// quantize_a8 of one 32-value group held one value per lane: writes the lane's code, returns (delta, sum) in every
// lane. max ignores NaN as std::max(amax, |x|) does; NaN values give code -127 as the host's clamps.
struct A8Group {
    float delta;
    std::int32_t sum;
    std::int8_t code;
};
__device__ __forceinline__ A8Group a8_group(float value) {
    float amax = fabsf(value);
#pragma unroll
    for (int d = 16; d >= 1; d /= 2) amax = fmaxf(amax, __shfl_xor_sync(kFull, amax, d));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float rounded = tiny ? 0.0F : rintf(value / delta);
    const float low = -127.0F < rounded ? rounded : -127.0F;  // std::max(-127, rounded)
    const float clamped = low < 127.0F ? low : 127.0F;        // std::min(127, low)
    const auto code = static_cast<std::int8_t>(clamped);
    std::int32_t sum = code;
#pragma unroll
    for (int d = 16; d >= 1; d /= 2) sum += __shfl_xor_sync(kFull, sum, d);
    return {delta, sum, code};
}

}  // namespace strix::ds41::cuda::moe_detail
