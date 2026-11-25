#pragma once
// NVFP4 activation scale rules (Home: shared, backend-neutral; host and device). An NVFP4 activation group is 16 inputs with one
// E4M3 scale code c (scale = value(c) * ts, ts = the row scale) and 16 E2M1 codes (round to nearest even, clamped to +-6).
// Any positive finite E4M3 code is a legal operand of the block-scaled FP4 MMA, so a rule changes only the image bits: plan,
// tiles, GEMM accumulation order and the projection's arithmetic identity are unchanged.
//   nearest@1 (default of every producer): c = E4M3 RN(a16 / 6 / ts); the group max lands on code 6.
//   mse@1: candidates are the nearest code c0, then c0 + 1 .. c0 + 6, c0 - 1, c0 - 2 (valid codes 1 .. 0x7E); the FIRST
//          candidate with the strictly smallest group error sum (q * scale - v)^2 wins, so a group that no candidate improves
//          keeps the nearest rule's bits, and the chosen group error is never above the nearest rule's.
//          The scores of one group are computed on the group and its scales multiplied by one common power of two
//          (score_normalizer): 1 for a group max in [2^-32, 2^32), so those scores are bit-for-bit the plain FP32 sums;
//          outside that band the max is brought to [0.5, 1), so a finite group never overflows (or underflows) every
//          score into a tie that hides the better candidate. Exact power-of-two scaling keeps the candidate order.
// The device bodies (backend core) take c0 from their conversion instruction and call search_scale_code below; host tests
// use the same functions.
#include <cmath>
#include <cstdint>

#include "strix/weight/formats/scalar_float.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_NVFP4_HD __host__ __device__
#else
#define STRIX_NVFP4_HD
#endif

namespace strix::moe_core::nvfp4_act {

enum class ScaleRule : std::uint8_t { nearest, mse };

constexpr const char* identity(ScaleRule rule) {
    return rule == ScaleRule::mse ? "nvfp4-act-scale-mse@1" : "nvfp4-act-scale-nearest@1";
}

inline constexpr unsigned kMaxCode = 0x7EU;  // 448
inline constexpr unsigned kCandidates = 8U;  // offsets after c0

// Offset of search candidate k (0 .. kCandidates - 1): +1 .. +6, then -1, -2.
STRIX_NVFP4_HD constexpr int search_offset(unsigned k) { return k < 6U ? static_cast<int>(k) + 1 : 5 - static_cast<int>(k); }

// Value of a positive E4M3 code (bias 7, 3 mantissa bits, subnormals m * 2^-9); exact.
STRIX_NVFP4_HD inline float e4m3_value(unsigned code) {
    const unsigned e = (code >> 3U) & 15U, m = code & 7U;
    if (e == 0U) return static_cast<float>(m) * 0.001953125F;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ldexpf(static_cast<float>(8U + m), static_cast<int>(e) - 10);
#else
    return std::ldexp(static_cast<float>(8U + m), static_cast<int>(e) - 10);
#endif
}

// E2M1 magnitude of |u| <= 6, round to nearest, ties to the even code (0, 0.5, 1, 1.5, 2, 3, 4, 6 = codes 0 .. 7).
STRIX_NVFP4_HD constexpr float e2m1_round(float a) {
    return a <= 0.25F ? 0.F : a < 0.75F ? 0.5F : a <= 1.25F ? 1.F : a < 1.75F ? 1.5F
         : a <= 2.5F ? 2.F : a < 3.5F ? 3.F : a <= 5.F ? 4.F : 6.F;
}

STRIX_NVFP4_HD inline float fma_rn(float a, float b, float c) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return fmaf(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

// Two-term rule r2@1 (the classes nvfp4_mse_r2*, moe_core/grouped_w4a4.hpp): term 1 is the group's image under the class scale
// rule; term 2 is the image, at the SAME row scale ts and the same rule, of the residual
//   r_i = v_i - q_i * scale  (one FMA rounding; q_i = value of code i of term 1 by the shared E2M1 decoder
//   weight::scalar_float::e2m1, code 8 = -0; scale = value(term-1 code) * ts in FP32).
// Codes in canonical order (word 0 = values 0..7, low nibble = even value). Per element |r_i - q2_i * scale2| <= |r_i|
// (0 is an E2M1 code and the clamp keeps the sign side), so two terms never increase any element's error.
STRIX_NVFP4_HD inline void residual(const float (&v)[16], float scale, const std::uint32_t (&word)[2], float (&r)[16]) {
    for (int i = 0; i < 16; ++i) {
        const unsigned nibble = (word[i >> 3] >> (4 * (i & 7))) & 15U;
        r[i] = -fma_rn(weight::scalar_float::e2m1(static_cast<std::uint8_t>(nibble)), scale, -v[i]);
    }
}
// Error of one group at scale `scale` (0: every code 0), summed in input order.
STRIX_NVFP4_HD inline float group_error(const float (&v)[16], float scale) {
    float e = 0.F;
    for (int i = 0; i < 16; ++i) {
        float u = scale != 0.F ? v[i] / scale : 0.F;
        u = u > 6.F ? 6.F : u < -6.F ? -6.F : u;
        const float q = u < 0.F ? -e2m1_round(-u) : e2m1_round(u);
        const float d = fma_rn(q, scale, -v[i]);
        e = fma_rn(d, d, e);
    }
    return e;
}

// Common power of two of one group's scores: 1 when the group max is in [2^-32, 2^32) (each squared residual stays
// below ~2^70, far from FP32 overflow), else 2^-k for a max of m * 2^k, m in [0.5, 1), clamped to normal factors
// 2^-126 .. 2^127 so a flush-to-zero device never sees a subnormal factor.
STRIX_NVFP4_HD inline float score_normalizer(const float (&v)[16]) {
    float a = 0.F;
    for (int i = 0; i < 16; ++i) {
        const float m = v[i] < 0.F ? -v[i] : v[i];
        a = m > a ? m : a;
    }
    if (!(a > 0.F)) return 1.F;
    int k = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    (void)frexpf(a, &k);
#else
    (void)std::frexp(a, &k);
#endif
    if (k >= -31 && k <= 32) return 1.F;
    const int shift = -k < -126 ? -126 : -k > 127 ? 127 : -k;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ldexpf(1.F, shift);
#else
    return std::ldexp(1.F, shift);
#endif
}

// mse@1: the scale code for a group whose nearest-rule code is c0 (0 when ts or the group is zero).
STRIX_NVFP4_HD inline std::uint8_t search_scale_code(const float (&v)[16], float ts, std::uint8_t c0) {
    if (!(ts > 0.F)) return c0;
    const float s = score_normalizer(v);
    float w[16];
    for (int i = 0; i < 16; ++i) w[i] = v[i] * s;  // exact for normal results; s == 1 leaves every bit
    std::uint8_t best = c0;
    float best_error = group_error(w, c0 ? e4m3_value(c0) * ts * s : 0.F);
    for (unsigned k = 0; k < kCandidates; ++k) {
        const int c = static_cast<int>(c0) + search_offset(k);
        if (c < 1 || c > static_cast<int>(kMaxCode)) continue;
        const float e = group_error(w, e4m3_value(static_cast<unsigned>(c)) * ts * s);
        if (e < best_error) {
            best_error = e;
            best = static_cast<std::uint8_t>(c);
        }
    }
    return best;
}

}  // namespace strix::moe_core::nvfp4_act
