#pragma once
#include <cmath>
#include <cstdint>
#if defined(__HIPCC__)
#define STRIX_INT_HD __host__ __device__
#else
#define STRIX_INT_HD
#endif
namespace strix::core::integer_math {
// Explicit RN32 boundaries; compile with contraction off.
STRIX_INT_HD inline float product(int dot, float sw, float sa) {
    return float(dot) * (sw * sa);
}
STRIX_INT_HD inline float affine(int dot, int minimum, float d, float dm, float sa) {
    const float scaled = d * float(dot);
    const float offset = dm * float(minimum);
    return (scaled - offset) * sa;
}
// FAST_I4 grid approximation: GGUF {8,25,43} -> {1,4,7}.
STRIX_INT_HD inline int iq4(int value) {
    const int mag = value < 0 ? -value : value;
    const int q = mag == 8 ? 1 : (mag == 25 ? 4 : 7);
    return value < 0 ? -q : q;
}
STRIX_INT_HD inline unsigned nibble(int q) { return unsigned(q) & 15U; }
STRIX_INT_HD inline int signed_nibble(unsigned q) { return int(q & 7U) - int(q & 8U); }
STRIX_INT_HD inline float a4_step(int amax) { return amax == 0 ? 1.0f : float(amax) / 7.0f; }
STRIX_INT_HD inline int a4_code(int value, float step) {
    const int q = int(::rintf(float(value) / step));
    return q < -7 ? -7 : (q > 7 ? 7 : q);
}
// The same A4 rule on FP32 values (a group quantised directly, no A8 step): step = amax / 7 (1 for a zero or an
// underflowing group), code = rint(v / step) clamped to +-7 (a NaN quotient gives code 0).
STRIX_INT_HD inline float a4_step_f(float amax) {
    const float s = amax / 7.0f;
    return s > 0.0f ? s : 1.0f;
}
STRIX_INT_HD inline int a4_code_f(float value, float step) {
    const float q = ::rintf(value / step);
    return q < -7.0f ? -7 : (q > 7.0f ? 7 : (q == q ? int(q) : 0));
}
// Clip candidates of a load-time MSE step search (the schedule of the host low-bit packers, weight/host_lowbit.hpp):
// ratio 1.00, 0.98, ..., 0.60 of the unclipped maximum.
inline constexpr unsigned kA4ClipSteps = 21;
STRIX_INT_HD inline float a4_clip_ratio(unsigned i) { return 1.0f - 0.02f * static_cast<float>(i); }
// High-class-bit-first butterfly. Low-bit-first changes the output bits.
STRIX_INT_HD inline float tree32(float (&v)[32]) {
    for (unsigned s = 16; s; s /= 2)
        for (unsigned i = 0; i < s; ++i) v[i] = v[i] + v[i + s];
    return v[0];
}
}
#undef STRIX_INT_HD
