#pragma once
// Numeric traits of the shared RMSNorm (host and device): how a lane accumulates its squares, how many values one wave
// sums per chunk in the wave_chunk_serial order, and the correctly rounded inverse square root form. Every trait is
// optional in a Traits struct; an absent trait keeps today's arithmetic (fma squares, lane_strided order, the Traits'
// own inv_sqrt), so existing bindings compile to the same code.
// Home: shared (norm numeric contract). Bound by: norm_device.hpp row bodies, reference.hpp, norm_presets.hpp.
#include "norm.hpp"

#include "strix/numeric/rsqrt_exact.hpp"

#include <cstdint>

namespace strix::ops::norm {

// How a lane adds one square to its running sum a:
//   fma:     a = fma(v, v, a)    (one rounding; the existing bindings);
//   mul_add: a = a + (v * v)     (the existing identity, requires strict compiler flags on the host);
//   mul_add_uncontracted: a = a + (v * v) with two roundings, even in a contraction-enabled translation unit.
enum class SquareForm { fma, mul_add, mul_add_uncontracted };

template <class Traits>
STRIX_OPS_NORM_HD constexpr SquareForm square_form_of() {
    if constexpr (requires { Traits::square_form; }) return Traits::square_form;
    else return SquareForm::fma;
}

template <SquareForm F>
STRIX_OPS_NORM_HD inline float square_step(float a, float v) {
    if constexpr (F == SquareForm::fma) {
        return __builtin_fmaf(v, v, a);
    } else {
#if defined(__CUDA_ARCH__)
        return __fadd_rn(a, __fmul_rn(v, v));  // nvcc contracts plain a + v * v under -fmad
#else
        if constexpr (F == SquareForm::mul_add_uncontracted) {
            // A volatile intermediate prevents contraction even if the caller
            // requests it across expressions. Device admission checks its cost.
            const volatile float square = v * v;
            return a + square;
        } else {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
            const float square = v * v;
            return a + square;
        }
#endif
    }
}

// ReductionOrder::wave_chunk_serial: the row is cut into chunks of wave_chunk values (0 = the whole row is one chunk);
// wave w of the block sums chunks w, w + waves, ...: lane l adds the squares of values first + l, first + l + 32, ... in
// ascending order (square_form_of), the 32 lane sums are added by the xor butterfly d = 16, 8, 4, 2, 1; the row sum is
// then the chunk sums added serially in chunk order, starting from chunk 0's sum.
inline constexpr unsigned kWaveLanes = 32;

template <class Traits>
STRIX_OPS_NORM_HD constexpr unsigned wave_chunk_of() {
    if constexpr (requires { Traits::wave_chunk; }) return Traits::wave_chunk;
    else return 0U;
}

// Chunks of a row of n values (n > 0) at chunk length c (0 = one chunk).
STRIX_OPS_NORM_HD constexpr unsigned wave_chunk_count(unsigned n, unsigned c) { return c == 0U ? 1U : (n + c - 1U) / c; }

// Inverse square root forms for Traits::inv_sqrt (u = sumsq / n + eps):
//   declared (the existing presets): 1 / declared_sqrt(u), two correctly rounded operations;
//   correctly_rounded_finite: the correctly rounded u^(-1/2); a non-finite u (the sum of squares overflowed or is NaN)
//   gives a quiet NaN factor, so every output of such a row is NaN.
STRIX_OPS_NORM_HD inline float inv_sqrt_correctly_rounded_finite(float u) {
    const auto bits = __builtin_bit_cast(std::uint32_t, u);
    if ((bits & 0x7f800000U) == 0x7f800000U) return __builtin_bit_cast(float, 0x7fc00000U);
    return numeric::rsqrt_correctly_rounded(u);
}

}  // namespace strix::ops::norm
