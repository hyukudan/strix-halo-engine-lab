#pragma once

#include "strix/msvc_math_builtins.hpp"
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_HALF_HD __host__ __device__
#else
#define STRIX_HALF_HD
#endif

namespace strix::weight::ieee_half {

// Exact IEEE binary16 decoding, including signed zeros and NaN payloads.
// Preserve the established host, HIP and CUDA conversion forms.
STRIX_HALF_HD inline float decode(std::uint16_t h) {
#if defined(__CUDA_ARCH__)
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    const bool nan = (h & 0x7c00U) == 0x7c00U && (h & 0x3ffU) != 0U;
    return nan ? __uint_as_float((static_cast<std::uint32_t>(h & 0x8000U) << 16U) | 0x7f800000U | (static_cast<std::uint32_t>(h & 0x3ffU) << 13U)) : f;
#elif defined(__HIPCC__)
    // Keep the qualified HIP implementation; CUDA and host use their own forms.
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    std::uint32_t exponent = (h >> 10U) & 0x1fU, mantissa = h & 0x3ffU;
    if (exponent == 0x1fU) return __builtin_bit_cast(float, sign | 0x7f800000U | (mantissa << 13U));
    if (exponent == 0) {
        if (mantissa == 0) return __builtin_bit_cast(float, sign);
        // subnormal half: mantissa * 2^-24, exact in float
        const float v = static_cast<float>(mantissa) * 5.9604644775390625e-8F;
        return sign ? -v : v;
    }
    return __builtin_bit_cast(float, sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
#else
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    const std::uint32_t exponent = (h >> 10U) & 0x1fU, mantissa = h & 0x3ffU;
    const std::uint32_t special = sign | 0x7f800000U | (mantissa << 13U);
    // zero or subnormal half: mantissa * 2^-24, exact in float (+0 for a zero mantissa), with the sign bit
    const std::uint32_t small = sign | __builtin_bit_cast(std::uint32_t, static_cast<float>(mantissa) * 5.9604644775390625e-8F);
    const std::uint32_t normal = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    return __builtin_bit_cast(float, exponent == 0x1fU ? special : exponent == 0U ? small : normal);
#endif
}

// FP32 -> FP16, RNE, gradual underflow, signed zero. False means nonfinite or
// rounding overflow; callers must fail the request, never silently clip.
STRIX_HALF_HD inline bool encode_finite(float x, std::uint16_t& out) {
    const auto u = __builtin_bit_cast(std::uint32_t, x), a = u & 0x7fffffffU;
    const auto sign = std::uint16_t((u >> 16U) & 0x8000U);
    if (a >= 0x7f800000U) return false;
    if (a < 0x33000000U) { out = sign; return true; }
    if (a < 0x38800000U) {
        const unsigned shift = 126U - (a >> 23U);
        const auto m = (a & 0x7fffffU) | 0x800000U;
        const auto q = m >> shift, remainder = m & ((1U << shift) - 1U);
        const auto midpoint = 1U << (shift - 1U);
        out = std::uint16_t(sign | (q + unsigned(remainder > midpoint || (remainder == midpoint && (q & 1U)))));
        return true;
    }
    const auto q = ((a - 0x38000000U) + 0xfffU + ((a >> 13U) & 1U)) >> 13U;
    if (q >= 0x7c00U) return false;
    out = std::uint16_t(sign | q);
    return true;
}

} // namespace strix::weight::ieee_half

#undef STRIX_HALF_HD
