#pragma once

// Declared scalar arithmetic shared by every engine: basic IEEE operations only, one definition for host,
// HIP and CUDA. Engine headers re-export these names in their own namespaces; shared code binds this header.

#include "strix/moe/declared_exp.hpp"
#include "strix/msvc_math_builtins.hpp"
#include "strix/weight/formats/ieee_half.hpp"

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_NUMERIC_HD __host__ __device__
#else
#define STRIX_NUMERIC_HD
#endif

namespace strix::numeric {

STRIX_NUMERIC_HD inline float declared_exp(float x) { return ::strix::moe::declared_math::declared_exp(x); }

// Correctly rounded square root. On the device through double precision (the float rounding of the
// correctly rounded double root is the correctly rounded float root); the single-precision device
// root is not correctly rounded over the whole range.
STRIX_NUMERIC_HD inline float declared_sqrt(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return static_cast<float>(__dsqrt_rn(static_cast<double>(x)));
#elif defined(__CUDA_ARCH__)
    return __fsqrt_rn(x);  // IEEE round-to-nearest single-precision root on NVIDIA devices
#else
    return __builtin_sqrtf(x);
#endif
}

STRIX_NUMERIC_HD inline float declared_tanh(float x) { return ::strix::moe::declared_math::declared_tanh(x); }
STRIX_NUMERIC_HD inline float declared_sigmoid(float x) { return 1.0F / (1.0F + declared_exp(-x)); }
STRIX_NUMERIC_HD inline float declared_rsqrt(float x) { return 1.0F / declared_sqrt(x); }
// IEEE half to float (exact; NaN payloads kept). CUDA uses hardware conversion with payload restoration;
// the host uses selects, and HIP keeps the qualified branch-based formula.
STRIX_NUMERIC_HD inline float f16_to_float(std::uint16_t h) {
    return ::strix::weight::ieee_half::decode(h);
}
// f16_to_float by selects instead of branches (so that loads ahead of it are not held behind a branch per
// value): the same bits for every input, NaN payloads included.
STRIX_NUMERIC_HD inline float f16_to_float_select(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    const std::uint32_t exponent = (h >> 10U) & 0x1fU, mantissa = h & 0x3ffU;
    const std::uint32_t special = sign | 0x7f800000U | (mantissa << 13U);
    const std::uint32_t normal = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    const std::uint32_t small = sign | __builtin_bit_cast(std::uint32_t, static_cast<float>(mantissa) * 5.9604644775390625e-8F);
    return __builtin_bit_cast(float, exponent == 0x1fU ? special : (exponent == 0 ? small : normal));
}
// Round a float to the nearest bfloat16 (ties to even), returned as float. NaN stays NaN.
STRIX_NUMERIC_HD inline float round_to_bf16(float x) {
    const auto bits = __builtin_bit_cast(std::uint32_t, x);
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x7fffffU)) return x;
    const std::uint32_t rounded = (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U;
    return __builtin_bit_cast(float, rounded);
}

// sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)): the sign bit of dot (also of -0) carries over and a
// NaN dot stays NaN.
STRIX_NUMERIC_HD inline float engram_gate_of(float dot) {
    const auto bits = __builtin_bit_cast(std::uint32_t, dot);
    const float a = __builtin_bit_cast(float, bits & 0x7fffffffU);
    const float root = declared_sqrt(a < 1e-6F ? 1e-6F : a);
    return declared_sigmoid(__builtin_bit_cast(float, __builtin_bit_cast(std::uint32_t, root) | (bits & 0x80000000U)));
}

}  // namespace strix::numeric
