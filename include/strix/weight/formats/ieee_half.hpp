#pragma once

#include "strix/msvc_math_builtins.hpp"
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_HALF_HD __host__ __device__
#else
#define STRIX_HALF_HD
#endif

namespace strix::weight::ieee_half {

enum class NanMode : std::uint8_t { preserve, quiet };

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

// Opt in per format. Quiet on the input bits so no host or device FP
// operation consumes a signaling value merely to change its NaN payload.
STRIX_HALF_HD inline float decode(std::uint16_t h, NanMode mode) {
    if (mode == NanMode::quiet && (h & 0x7c00U) == 0x7c00U && (h & 0x03ffU) != 0U)
        h = static_cast<std::uint16_t>(h | 0x0200U);
    return decode(h);
}

} // namespace strix::weight::ieee_half

#undef STRIX_HALF_HD
