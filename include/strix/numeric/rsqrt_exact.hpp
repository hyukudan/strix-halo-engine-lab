#pragma once
// Correctly rounded single-precision x^(-1/2) (round to nearest even), host and device, from integer dyadics only: no
// approximate rsqrt, no sqrt/divide equivalence, no native reciprocal and no host libm. The result is the float nearest to
// the exact value, midpoint ties to even, for every input (subnormals included).
// Specials: +-0 -> +-inf; negative or NaN -> quiet NaN; +inf -> +0.
// Home: shared numeric helper. Bound by: the correctly rounded inverse square root form of the shared norm
// (src/ops/norm/norm_numeric_traits.hpp).
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_NUMERIC_RSQRT_HD __host__ __device__
#else
#define STRIX_NUMERIC_RSQRT_HD
#endif

namespace strix::numeric {
namespace rsqrt_detail {

STRIX_NUMERIC_RSQRT_HD inline std::uint32_t bits_of(float x) { return __builtin_bit_cast(std::uint32_t, x); }
STRIX_NUMERIC_RSQRT_HD inline float float_of(std::uint32_t b) { return __builtin_bit_cast(float, b); }

// Compare mx * my^2 * 2^exponent with 1, exactly (at most 74 significant bits): -1, 0 or 1.
STRIX_NUMERIC_RSQRT_HD inline int square_compare(std::uint32_t mx, std::uint32_t my, int exponent) {
    const std::uint64_t sq = std::uint64_t(my) * my;
    const std::uint64_t a = (sq & 0xffffffffULL) * mx, b = (sq >> 32U) * mx;
    const std::uint64_t lo = a + (b << 32U), hi = (b >> 32U) + (lo < a ? 1U : 0U);
    const int k = -exponent;
    if (k < 0) return 1;
    if (k >= 96) return -1;
    const std::uint64_t th = k >= 64 ? std::uint64_t(1) << (k - 64) : 0;
    const std::uint64_t tl = k < 64 ? std::uint64_t(1) << k : 0;
    if (hi != th) return hi < th ? -1 : 1;
    return lo == tl ? 0 : (lo < tl ? -1 : 1);
}

// A finite non-negative float as m * 2^e with an integer significand m.
STRIX_NUMERIC_RSQRT_HD inline void dyadic(std::uint32_t bits, std::uint32_t& m, int& e) {
    const auto field = bits >> 23U;
    m = (bits & 0x7fffffU) | (field != 0 ? 0x800000U : 0U);
    e = field != 0 ? int(field) - 150 : -149;
}

}  // namespace rsqrt_detail

// The largest normal y with x * y^2 <= 1 by bisection over the float bit patterns, then the midpoint rule on y and its
// successor (ties to the even pattern).
STRIX_NUMERIC_RSQRT_HD inline float rsqrt_correctly_rounded(float x) {
    using namespace rsqrt_detail;
    const auto bits = bits_of(x);
    if ((bits & 0x7fffffffU) == 0) return float_of((bits & 0x80000000U) | 0x7f800000U);
    if ((bits >> 31U) != 0 || bits > 0x7f800000U) return float_of(0x7fc00000U);
    if (bits == 0x7f800000U) return 0.0F;
    std::uint32_t mx;
    int ex;
    dyadic(bits, mx, ex);
    std::uint32_t lo = 0x00800000U, hi = 0x7f7fffffU;
    while (lo < hi) {
        const auto mid = lo + (hi - lo + 1U) / 2U;
        std::uint32_t my;
        int ey;
        dyadic(mid, my, ey);
        if (square_compare(mx, my, ex + 2 * ey) <= 0) lo = mid;
        else hi = mid - 1U;
    }
    std::uint32_t my;
    int ey;
    dyadic(lo, my, ey);
    const int c = square_compare(mx, 2U * my + 1U, ex + 2 * (ey - 1));
    return float_of(lo + (c < 0 || (c == 0 && (lo & 1U)) ? 1U : 0U));
}

}  // namespace strix::numeric
