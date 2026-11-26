#pragma once
// Model-neutral home of the DS41-declared exp/SwiGLU arithmetic. Frozen FP32
// expression and operation order; compile with each backend's existing FP flags.
#include "strix/moe/activation_policy.hpp"
#include "strix/msvc_math_builtins.hpp"
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_DECL_MOE_HD __host__ __device__
#else
#define STRIX_DECL_MOE_HD
#endif

namespace strix::moe::declared_math {
enum class ExpForm { frozen_v1, fixed_rne_v2 };

namespace detail {
STRIX_DECL_MOE_HD inline float pow2(int k) {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(k + 127) << 23U);
}

// Round an unsigned fixed-point value after a right shift, with ties to even.
STRIX_DECL_MOE_HD inline std::uint64_t round_shift_even(std::uint64_t value, unsigned shift) {
    const auto whole = value >> shift;
    const auto half = std::uint64_t{1} << (shift - 1U);
    const auto remainder = value & ((half << 1U) - 1U);
    return whole + (remainder > half || (remainder == half && (whole & 1U) != 0U));
}

// Horner products use |r| <= ln(2)/2 and p <= 1.42 in Q32, so their unsigned
// magnitude fits 64 bits. The sign is carried outside the rounding operation.
STRIX_DECL_MOE_HD inline std::int64_t mul_q32(std::int64_t p, std::int64_t r) {
    const auto magnitude = static_cast<std::uint64_t>(r < 0 ? -r : r);
    const auto rounded = static_cast<std::int64_t>(round_shift_even(static_cast<std::uint64_t>(p) * magnitude, 32));
    return r < 0 ? -rounded : rounded;
}

STRIX_DECL_MOE_HD inline float pack_exp_q32(std::uint64_t p, int k) {
    constexpr auto unit = std::uint64_t{1} << 32U;
    const int exponent = p >= unit ? k : k - 1;
    if (exponent > 127) return __builtin_huge_valf();
    if (exponent < -150) return 0.0F;

    if (exponent < -126) {
        // A binary32 subnormal is an integer multiple of 2^-149. Synthesize
        // its bits directly so a backend's flush-to-zero mode cannot drop it.
        const auto bits = static_cast<std::uint32_t>(round_shift_even(p, static_cast<unsigned>(32 - k - 149)));
        return __builtin_bit_cast(float, bits);
    }
    const auto significand = static_cast<std::uint32_t>(round_shift_even(p, p >= unit ? 9U : 8U));
    // Addition carries a rounded significand of 2.0 into the next exponent.
    const auto bits = (static_cast<std::uint32_t>(exponent + 126) << 23U) + significand;
    return __builtin_bit_cast(float, bits);
}
} // namespace detail

// exp(x) from basic IEEE operations only (no fused operations, no library call), so that host and
// device give the same bits: Cody-Waite reduction by ln 2, a degree-7 Taylor polynomial in Horner
// form and a two-step power-of-two scaling. Below -103 the result is 0, above 88 it is +inf.
STRIX_DECL_MOE_HD inline float declared_exp(float x) {
    if (!(x == x)) return x;
    if (x > 88.0F) return __builtin_huge_valf();
    if (x < -103.0F) return 0.0F;
    const float n = __builtin_rintf(x * 1.44269504F);
    const float r = (x - n * 0.693359375F) - n * -2.12194440e-4F;
    float p = 1.98412698e-4F;
    p = p * r + 1.38888889e-3F;
    p = p * r + 8.33333333e-3F;
    p = p * r + 4.16666667e-2F;
    p = p * r + 1.66666667e-1F;
    p = p * r + 0.5F;
    p = p * r + 1.0F;
    p = p * r + 1.0F;
    const int k = static_cast<int>(n), k1 = k / 2, k2 = k - k1;
    return (p * detail::pow2(k1)) * detail::pow2(k2);
}

// This explicit form evaluates a degree-7 exp approximation with Q32 range
// reduction and Horner steps. Its use in engine bindings requires a versioned
// arithmetic identity. The binary64 scaling and subtraction below are exact for
// binary32 input. Every inexact rounding is integer round-to-nearest-even.
template <ExpForm Form>
STRIX_DECL_MOE_HD inline float declared_exp(float x) {
    if constexpr (Form == ExpForm::frozen_v1) {
        return declared_exp(x);
    } else {
        const auto bits = __builtin_bit_cast(std::uint32_t, x);
        if ((bits & 0x7fffffffU) > 0x7f800000U) return x;
        if (x >= 89.0F) return __builtin_huge_valf();
        if (x <= -104.0F) return 0.0F;

        constexpr std::int64_t unit = std::int64_t{1} << 32U;
        constexpr std::int64_t ln2 = 2977044472;  // round(ln(2) * 2^32)
        const double magnitude = static_cast<double>((bits >> 31U) != 0U ? -x : x) * static_cast<double>(unit);
        const auto whole = static_cast<std::int64_t>(magnitude);
        const double fraction = magnitude - static_cast<double>(whole);
        const auto fixed = whole + (fraction > 0.5 || (fraction == 0.5 && (whole & 1) != 0));
        const auto quotient = fixed / ln2;
        const auto remainder = fixed % ln2;
        const auto k = quotient + (remainder > ln2 / 2 || (remainder == ln2 / 2 && (quotient & 1) != 0));
        const int sign = (bits >> 31U) != 0U ? -1 : 1;
        const auto r = sign * (fixed - k * ln2);

        std::int64_t p = 852176;  // round(2^32 / 7!)
        p = 5965232 + detail::mul_q32(p, r);
        p = 35791394 + detail::mul_q32(p, r);
        p = 178956971 + detail::mul_q32(p, r);
        p = 715827883 + detail::mul_q32(p, r);
        p = unit / 2 + detail::mul_q32(p, r);
        p = unit + detail::mul_q32(p, r);
        p = unit + detail::mul_q32(p, r);
        return detail::pack_exp_q32(static_cast<std::uint64_t>(p), sign * static_cast<int>(k));
    }
}
// SwiGLU with the model's clamps (the shared moe clamp trait: gate clamped from above, up on both sides), then
// silu(g) * u with silu(g) = g / (1 + exp(-g)). NaN inputs propagate.
STRIX_DECL_MOE_HD inline float swiglu_clamped(float gate, float up, float limit) {
    const auto c = strix::moe::swiglu_clamp(gate, up, limit);
    return (c.gate / (1.0F + declared_exp(-c.gate))) * c.up;
}


} // namespace strix::moe::declared_math
#undef STRIX_DECL_MOE_HD
