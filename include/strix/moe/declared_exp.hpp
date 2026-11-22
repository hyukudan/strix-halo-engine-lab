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
namespace detail {
STRIX_DECL_MOE_HD inline float pow2(int k) {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(k + 127) << 23U);
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
// SwiGLU with the model's clamps (the shared moe clamp trait: gate clamped from above, up on both sides), then
// silu(g) * u with silu(g) = g / (1 + exp(-g)). NaN inputs propagate.
STRIX_DECL_MOE_HD inline float swiglu_clamped(float gate, float up, float limit) {
    const auto c = strix::moe::swiglu_clamp(gate, up, limit);
    return (c.gate / (1.0F + declared_exp(-c.gate))) * c.up;
}


} // namespace strix::moe::declared_math
#undef STRIX_DECL_MOE_HD
