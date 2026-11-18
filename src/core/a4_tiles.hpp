#pragma once
// A4 activation groups of the dense W4A4 class (planned dense-w4a4-iu4 policy of f16_dense_tall), shared rule of
// integer_math (fast_i4): per row, groups of G consecutive inputs (G = 32 or 64), step = amax / 7, code = rint(v / step)
// clamped to +-7, value = code * step (FP32). Optional rotated basis: an orthonormal Sylvester Hadamard of 128 inputs
// (butterflies h = 1, 2, 4, ..., 64 ascending, a + b / a - b, then x 1/sqrt(128)) before the groups and again after
// (H is its own inverse), the activation half of a rotated W4A4 class.
//
// Two forms with the same bits: `reference` (host / device, one row) and Emu<G, Rotate>::apply (device, lane form for
// the x16 tile producers of f16_dense_tall: one lane holds 8 consecutive inputs, a group is G / 8 neighbouring lanes,
// a Hadamard block 16 lanes). Emu tiles are fp16_rne(value): the element values a W4A4 kernel multiplies, up to that
// one FP16 rounding of code * step, so the exact FP16 GEMM over them is the quality emulation of the class (gate
// instrument; it moves no work). Model-agnostic; Home: shared (backend core, HIP).
#include "integer_math.hpp"
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__HIPCC__)
#define STRIX_A4_HD __host__ __device__
#else
#define STRIX_A4_HD
#endif

namespace strix::core::a4_tiles {
inline constexpr float kH128 = 0.0883883476483184405501f;  // 1 / sqrt(128), FP32

STRIX_A4_HD inline float fabs_(float v) { return v < 0.0f ? -v : v; }
// One butterfly pair: the lower element gets a + b, the upper a - b.
STRIX_A4_HD inline void fwht128(float* v) {
    for (unsigned h = 1; h < 128U; h *= 2U)
        for (unsigned i = 0; i < 128U; ++i)
            if ((i & h) == 0U) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
    for (unsigned i = 0; i < 128U; ++i) v[i] *= kH128;
}
STRIX_A4_HD inline void dequant_group(float* v, unsigned n) {
    float amax = 0.0f;
    for (unsigned i = 0; i < n; ++i) amax = fabs_(v[i]) > amax ? fabs_(v[i]) : amax;
    const float step = integer_math::a4_step_f(amax);
    for (unsigned i = 0; i < n; ++i) v[i] = static_cast<float>(integer_math::a4_code_f(v[i], step)) * step;
}
// One row of `in` inputs (in % 128 == 0), in place.
STRIX_A4_HD inline void reference(float* row, unsigned in, unsigned group, bool rotate) {
    for (unsigned b = 0; b < in; b += 128U) {
        if (rotate) fwht128(row + b);
        for (unsigned g = 0; g < 128U; g += group) dequant_group(row + b + g, group);
        if (rotate) fwht128(row + b);
    }
}

#if defined(__HIPCC__)
// Lane form: v = this lane's 8 inputs k = 8 c .. 8 c + 7 of one row; the lanes of a row are consecutive in c and a row
// starts on a 16-lane boundary (in % 128 == 0). All lanes of a group / block must be active.
template <unsigned G, bool Rotate> struct Emu {
    static_assert(G == 32U || G == 64U);
    template <class V8> __device__ static void fwht(V8& v) {
#pragma unroll
        for (unsigned h = 1; h < 8U; h *= 2U)
#pragma unroll
            for (unsigned i = 0; i < 8U; ++i)
                if ((i & h) == 0U) {
                    const float a = v[i], b = v[i + h];
                    v[i] = a + b;
                    v[i + h] = a - b;
                }
        const unsigned lane = threadIdx.x % 32U;
#pragma unroll
        for (unsigned d = 1; d < 16U; d *= 2U) {
            const bool upper = (lane & d) != 0U;
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const float p = __shfl_xor(v[j], static_cast<int>(d), 32);
                v[j] = upper ? p - v[j] : v[j] + p;  // upper element: a - b with a = the lower (partner) value
            }
        }
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            float t = v[j] * kH128;
            asm volatile("" : "+v"(t));  // no contraction of the scale into a following add or FP16 cast
            v[j] = t;
        }
    }
    template <class V8> __device__ static void apply(V8& v) {
        if constexpr (Rotate) fwht(v);
        float amax = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) amax = fabs_(v[j]) > amax ? fabs_(v[j]) : amax;
#pragma unroll
        for (unsigned d = 1; d < G / 8U; d *= 2U) {
            const float o = __shfl_xor(amax, static_cast<int>(d), 32);
            amax = o > amax ? o : amax;
        }
        const float step = integer_math::a4_step_f(amax);
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            float t = static_cast<float>(integer_math::a4_code_f(v[j], step)) * step;
            asm volatile("" : "+v"(t));  // FP32 product rounded once, as the reference
            v[j] = t;
        }
        if constexpr (Rotate) fwht(v);
    }
};
#endif
}  // namespace strix::core::a4_tiles
#undef STRIX_A4_HD
