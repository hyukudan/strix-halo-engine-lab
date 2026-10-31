#pragma once
#include "gemma4/contracts/g4_identity.hpp"

namespace strix::gemma4::gpu {
struct alignas(8) RsqrtInterval {
    std::uint32_t lower, upper;
};
namespace interval_detail {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
static __device__ __constant__ const RsqrtInterval bounds[] = {
#else
inline constexpr RsqrtInterval bounds[] = {
#endif
#include "norm_intervals.inc"
};
} // namespace interval_detail
// Precondition: mx * 2^ex is a positive finite FP32 value. The real reciprocal
// square root is in [2^result, 2^(result+1)); an even power of two is exact.
STRIX_G4_HD inline int rsqrt_exponent(std::uint32_t mx, int ex) {
#if defined(__HIPCC__)
    const int leading = __builtin_clz(mx);
#else
    const int leading = std::countl_zero(mx);
#endif
    const int exponent = ex + 31 - leading;
    const int half = exponent >= 0 ? exponent / 2 : -((-exponent + 1) / 2);
    return -half - 1 + (exponent % 2 == 0 && (mx & (mx - 1U)) == 0U ? 1 : 0);
}
// Integer normalization gives m in [2^23,2^24). Within an output exponent,
// the floor significand solves m*y*y <= 2^70 (odd input exponent) or 2^71
// (even, non-power input). Monotonic endpoint certificates bound each bin.
STRIX_G4_HD inline void rsqrt_interval_bounds(std::uint32_t mx, int ex, int result_exponent,
                                             unsigned& lo, unsigned& hi) {
#if defined(__HIPCC__)
    const int shift = __builtin_clz(mx) - 8;
#else
    const int shift = std::countl_zero(mx) - 8;
#endif
    const auto normalized = mx << unsigned(shift);
    const int exponent = ex - shift + 23;
    const unsigned base = unsigned(result_exponent + 127) << 23U;
    if (normalized == 0x800000U && exponent % 2 == 0) {
        lo = hi = base;
        return;
    }
    const auto index = ((normalized & 0x7fffffU) >> 15U) + (exponent % 2 == 0 ? 256U : 0U);
    const auto interval = interval_detail::bounds[index];
    const auto upper = interval.upper > 0xffffffU ? 0xffffffU : interval.upper;
    lo = base + interval.lower - 0x800000U;
    hi = base + upper - 0x800000U;
}
} // namespace strix::gemma4::gpu
