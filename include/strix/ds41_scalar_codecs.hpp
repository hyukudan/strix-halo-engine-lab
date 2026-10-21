#pragma once
// Scalar FP8 / FP4 codecs of the deepseek41 caches and weights, one definition for host and device: E8M0 scales,
// E4M3 (finite-only variant) and E2M1 values and their round-to-nearest-even encoders. Every decode is exact; special
// codes follow the formats: E8M0 255 and E4M3 S.1111.111 are NaN, E8M0 0 is 2^-127, E2M1 code 8 is -0.
// Branch-free so device kernels can decode in the inner loops.
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS41C_HD __host__ __device__ inline
#else
#define DS41C_HD inline
#endif

namespace strix::ds41::codec {

DS41C_HD std::uint32_t f2u(float x) { return __builtin_bit_cast(std::uint32_t, x); }
DS41C_HD float u2f(std::uint32_t u) { return __builtin_bit_cast(float, u); }
DS41C_HD float pow2f(int e) { return u2f(static_cast<std::uint32_t>(e + 127) << 23U); }  // e in [-126, 127]

// E8M0: 2^(e - 127); 0 -> 2^-127 (subnormal), 255 -> NaN.
DS41C_HD float e8m0(std::uint8_t e) {
    const std::uint32_t b = e == 0U ? 0x00400000U : static_cast<std::uint32_t>(e) << 23U;
    return u2f(e == 255U ? 0x7FC00000U : b);
}
// Half of E8M0 (scale of doubled E2M1 values): 2^(e - 128); 255 -> NaN.
DS41C_HD float e8m0_half(std::uint8_t e) {
    const std::uint32_t b = e < 2U ? 0x00200000U << e : static_cast<std::uint32_t>(e - 1U) << 23U;
    return u2f(e == 255U ? 0x7FC00000U : b);
}
// E4M3: bias 7, subnormals m * 2^-9, S.1111.111 NaN, no infinities.
DS41C_HD float e4m3(std::uint8_t c) {
    const std::uint32_t e = (c >> 3U) & 15U, m = c & 7U;
    const float normal = u2f(((e + 120U) << 23U) | (m << 20U)), sub = static_cast<float>(m) * 0.001953125F;
    const std::uint32_t mag = (c & 0x7FU) == 0x7FU ? 0x7FC00000U : f2u(e == 0U ? sub : normal);
    return u2f(mag | (static_cast<std::uint32_t>(c & 0x80U) << 24U));
}
// Round to nearest even, saturate to +-448; NaN -> 0x7f.
DS41C_HD std::uint8_t e4m3_encode(float x) {
    if (x != x) return 0x7FU;
    const std::uint8_t sign = (f2u(x) >> 31U) != 0U ? 0x80U : 0U;
    float a = x < 0.0F ? -x : x;
    if (a > 448.0F) a = 448.0F;
    std::uint8_t code;
    if (a < 0.015625F) {
        code = static_cast<std::uint8_t>(__builtin_rintf(a * 512.0F));
    } else {
        int e = static_cast<int>((f2u(a) >> 23U) & 255U) - 127;
        const float mant = a * pow2f(-e);
        float q = __builtin_rintf((mant - 1.0F) * 8.0F);
        if (q == 8.0F) { ++e; q = 0.0F; }
        code = static_cast<std::uint8_t>(((e + 7) << 3) | static_cast<int>(q));
    }
    return static_cast<std::uint8_t>(code | sign);
}
// E2M1: (1 + m/2) 2^(e-1), subnormal m/2, sign bit 3 (code 8 = -0).
DS41C_HD float e2m1(std::uint8_t c) {
    const std::uint32_t e = (c >> 1U) & 3U, m = c & 1U;
    const float normal = u2f(((e + 126U) << 23U) | (m << 22U)), sub = static_cast<float>(m) * 0.5F;
    return u2f(f2u(e == 0U ? sub : normal) | (static_cast<std::uint32_t>(c & 8U) << 28U));
}
// 2 x E2M1 as an integer in [-12, 12] (the MXFP4 / index-key integer form).
DS41C_HD int e2m1_doubled(std::uint8_t c) {
    const unsigned m = c & 7U;
    const int v = m < 5U ? static_cast<int>(m) : m == 5U ? 6 : m == 6U ? 8 : 12;
    return (c & 8U) != 0U ? -v : v;
}
// Round to nearest even over {0, .5, 1, 1.5, 2, 3, 4, 6}, saturate to +-6.
DS41C_HD std::uint8_t e2m1_encode(float x) {
    const std::uint8_t sign = (f2u(x) >> 31U) != 0U ? 8U : 0U;
    const float a = x < 0.0F ? -x : x;
    const std::uint8_t m = a <= 0.25F ? 0U : a < 0.75F ? 1U : a <= 1.25F ? 2U : a < 1.75F ? 3U : a <= 2.5F ? 4U
        : a < 3.5F ? 5U : a <= 5.0F ? 6U : 7U;
    return static_cast<std::uint8_t>(m | sign);
}

}  // namespace strix::ds41::codec
