#pragma once

// g4-v1: FP32 RNE, gradual underflow, separate multiply/add, no fast math or reassociation.
// Identity also binds model bytes, per-tensor formats, immutable rotary-table bytes and FP32 KV.
// Arithmetic changes require a named opt-in profile. Nonfinite weights/execution are refused.
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define STRIX_G4_HD __host__ __device__
#else
#define STRIX_G4_HD
#endif

namespace strix::gemma4 {

inline constexpr char arithmetic_id[] = "g4-v1";
inline constexpr std::uint32_t hidden = 2560, layers = 42, inter = 10240, vocab = 262144;
inline constexpr std::uint32_t heads = 8, kv_heads = 2, group = 4, ple_dim = 256;
inline constexpr std::uint32_t window = 512, max_positions = 65536, attention_chunk = 64;
inline constexpr std::uint32_t tap_layers[] = {5, 11, 17, 23, 35};
inline constexpr float norm_eps = 1e-6F, attention_scale = 1.0F, logit_cap = 30.0F;
inline constexpr float embedding_scale = 50.5F, ple_embedding_scale = 16.0F;
inline constexpr float model_ple_scale = 0.019764235376052371F, ple_combine_scale = 0.707106781186547524F;

template<class T, class U> STRIX_G4_HD inline T bit_cast(U x) {
#if defined(__HIPCC__)
    return __builtin_bit_cast(T, x);
#else
    return std::bit_cast<T>(x);
#endif
}
STRIX_G4_HD inline float bf16_value(std::uint16_t x) { return bit_cast<float>(std::uint32_t(x) << 16U); }
STRIX_G4_HD inline std::uint16_t bf16_bits(float x) {
    const auto u = bit_cast<std::uint32_t>(x);
    if ((u & 0x7fffffffU) > 0x7f800000U) return std::uint16_t((u >> 16U) | 0x40U);
    return std::uint16_t((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
STRIX_G4_HD inline std::uint16_t load16(const std::uint8_t* p) { return std::uint16_t(p[0] | (std::uint32_t(p[1]) << 8U)); }
STRIX_G4_HD inline float fp16_value(std::uint16_t h) {
    const auto sign = std::uint32_t(h & 0x8000U) << 16U;
    std::uint32_t e = (h >> 10U) & 31U, m = h & 1023U;
    if (e == 31U) return bit_cast<float>(sign | 0x7f800000U | (m << 13U));
    if (e == 0U) {
        if (m == 0U) return bit_cast<float>(sign);
        int k = -14;
        while ((m & 1024U) == 0U) { m <<= 1U; --k; }
        return bit_cast<float>(sign | (std::uint32_t(k + 127) << 23U) | ((m & 1023U) << 13U));
    }
    return bit_cast<float>(sign | ((e + 112U) << 23U) | (m << 13U));
}

// Original packed blocks, little endian: Q8_0=34/32, Q4_K=144/256, Q6_K=210/256 bytes/values.
STRIX_G4_HD inline float q8_0_value(const std::uint8_t* b, std::uint32_t i) {
    const int q = b[2U + i] < 128U ? b[2U + i] : int(b[2U + i]) - 256;
    return fp16_value(load16(b)) * float(q);
}
STRIX_G4_HD inline float q4_k_value(const std::uint8_t* b, std::uint32_t i) {
    const auto j = i / 32U;
    const auto* s = b + 4;
    const int scale = j < 4U ? s[j] & 63 : (s[j + 4U] & 15) | ((s[j - 4U] >> 6U) << 4U);
    const int minimum = j < 4U ? s[j + 4U] & 63 : (s[j + 4U] >> 4U) | ((s[j] >> 6U) << 4U);
    const auto qbyte = b[16U + (i / 64U) * 32U + i % 32U];
    const int q = (qbyte >> ((j & 1U) * 4U)) & 15;
    return fp16_value(load16(b)) * float(scale * q) - fp16_value(load16(b + 2)) * float(minimum);
}
STRIX_G4_HD inline float q6_k_value(const std::uint8_t* b, std::uint32_t i) {
    const auto half = i / 128U, quarter = (i % 128U) / 32U, l = i % 32U;
    const auto low = b[half * 64U + l + (quarter & 1U) * 32U];
    const auto high = b[128U + half * 32U + l];
    const int q = ((low >> (quarter >= 2U ? 4U : 0U)) & 15) | (((high >> (2U * quarter)) & 3) << 4);
    const auto raw = b[192U + i / 16U];
    const int s = raw < 128U ? raw : int(raw) - 256;
    return fp16_value(load16(b + 208)) * float(s * (q - 32));
}

namespace detail {
STRIX_G4_HD inline float pow2(int k) { return bit_cast<float>(std::uint32_t(k + 127) << 23U); }
// Compare mx * my^2 * 2^exponent with 1, exactly (at most 74 significant bits).
STRIX_G4_HD inline int square_compare(std::uint32_t mx, std::uint32_t my, int exponent) {
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
STRIX_G4_HD inline void dyadic(std::uint32_t bits, std::uint32_t& m, int& e) {
    const auto field = bits >> 23U;
    m = (bits & 0x7fffffU) | (field != 0 ? 0x800000U : 0U);
    e = field != 0 ? int(field) - 150 : -149;
}
}

// Correctly rounded x^(-1/2), checked using integer dyadics, including midpoint ties.
// No approximate rsqrt, sqrt/divide equivalence, native reciprocal or host libm assumption.
STRIX_G4_HD inline float pow32_rsqrt(float x) {
    const auto bits = bit_cast<std::uint32_t>(x);
    if ((bits & 0x7fffffffU) == 0) return bit_cast<float>((bits & 0x80000000U) | 0x7f800000U);
    if ((bits >> 31U) != 0 || bits > 0x7f800000U) return bit_cast<float>(0x7fc00000U);
    if (bits == 0x7f800000U) return 0.0F;
    std::uint32_t mx; int ex;
    detail::dyadic(bits, mx, ex);
    std::uint32_t lo = 0x00800000U, hi = 0x7f7fffffU;
    while (lo < hi) {
        const auto mid = lo + (hi - lo + 1U) / 2U;
        std::uint32_t my; int ey;
        detail::dyadic(mid, my, ey);
        if (detail::square_compare(mx, my, ex + 2 * ey) <= 0) lo = mid;
        else hi = mid - 1U;
    }
    std::uint32_t my; int ey;
    detail::dyadic(lo, my, ey);
    const int c = detail::square_compare(mx, 2U * my + 1U, ex + 2 * (ey - 1));
    return bit_cast<float>(lo + (c < 0 || (c == 0 && (lo & 1U)) ? 1U : 0U));
}

// MiniMax declared_exp copied verbatim in constants/order/saturation; portable RNE intrinsic spelling.
STRIX_G4_HD inline float declared_exp(float x) {
    if (!(x == x)) return x;
    if (x > 88.0F) return bit_cast<float>(0x7f800000U);
    if (x < -103.0F) return 0.0F;
#if defined(__HIPCC__)
    const float n = __builtin_rintf(x * 1.44269504F);
#else
    const float n = std::rint(x * 1.44269504F);
#endif
    const float r = (x - n * 0.693359375F) - n * -2.12194440e-4F;
    float p = 1.98412698e-4F;
    p = p * r + 1.38888889e-3F;
    p = p * r + 8.33333333e-3F;
    p = p * r + 4.16666667e-2F;
    p = p * r + 1.66666667e-1F;
    p = p * r + 0.5F;
    p = p * r + 1.0F;
    p = p * r + 1.0F;
    const int k = int(n), k1 = k / 2, k2 = k - k1;
    return (p * detail::pow2(k1)) * detail::pow2(k2);
}
STRIX_G4_HD inline float declared_tanh(float x) {
    const auto sign = bit_cast<std::uint32_t>(x) & 0x80000000U;
    const float a = bit_cast<float>(bit_cast<std::uint32_t>(x) & 0x7fffffffU);
    const float argument = -2.0F * a;
    if ((bit_cast<std::uint32_t>(argument) & 0x7f800000U) == 0x7f800000U) return bit_cast<float>(0x7fc00000U);
    const float e = declared_exp(argument);
    return bit_cast<float>(bit_cast<std::uint32_t>((1.0F - e) / (1.0F + e)) | sign);
}
STRIX_G4_HD inline float gelu_tanh(float x) {
    const float cube = (x * x) * x;
    if ((bit_cast<std::uint32_t>(cube) & 0x7f800000U) == 0x7f800000U) return bit_cast<float>(0x7fc00000U);
    return ((0.5F * x) * (1.0F + declared_tanh(0.79788456080286535588F * (x + 0.044715F * cube))));
}
STRIX_G4_HD inline float softcap(float x) { return declared_tanh(x / logit_cap) * logit_cap; }
STRIX_G4_HD inline bool better(float x, std::uint32_t id, float best, std::uint32_t best_id) {
    return x > best || (x == best && id < best_id);
}
STRIX_G4_HD inline float embedding_value(float decoded, float scale = embedding_scale) {
    return bf16_value(bf16_bits(bf16_value(bf16_bits(decoded)) * scale));
}
STRIX_G4_HD inline float butterfly(float* v) {
    for (unsigned d = 16; d != 0; d >>= 1U)
        for (unsigned l = 0; l < d; ++l) v[l] = v[l] + v[l + d];
    return v[0];
}
template<class WeightAt> STRIX_G4_HD inline float chunk16_dot(WeightAt weight, const float* x, std::uint32_t width) {
    float v[32] = {};
    for (std::uint32_t l = 0; l < 32; ++l)
        for (std::uint32_t c = l; c < (width + 15U) / 16U; c += 32U)
            for (std::uint32_t e = 0; e < 16; ++e) {
                const auto i = c * 16U + e;
                if (i < width) v[l] = v[l] + weight(i) * x[i];
                else v[l] = v[l] + 0.0F;
            }
    return butterfly(v);
}
STRIX_G4_HD inline float lane_dot(const float* a, const float* b, std::uint32_t width) {
    float v[32] = {};
    for (std::uint32_t l = 0; l < 32; ++l)
        for (std::uint32_t i = l; i < width; i += 32U) v[l] = v[l] + a[i] * b[i];
    return butterfly(v);
}
STRIX_G4_HD inline void rms_norm(const float* x, const float* w, float* out, std::uint32_t width, float eps = norm_eps) {
    const float u = lane_dot(x, x, width) / float(width) + eps;
    const float r = (bit_cast<std::uint32_t>(u) & 0x7f800000U) == 0x7f800000U ? bit_cast<float>(0x7fc00000U) : pow32_rsqrt(u);
    for (std::uint32_t i = 0; i < width; ++i) out[i] = w ? (x[i] * r) * w[i] : x[i] * r;
}

// Tables are generated offline with correctly rounded real pow/sin/cos, then hashed and shared.
// The callback is an offline rounding oracle, not a device-dependent libm implementation.
struct RopeOracle {
    float (*pow_real)(float theta, float exponent);
    float (*sin_real)(float angle);
    float (*cos_real)(float angle);
};
inline void rope_table_row(std::uint32_t position, bool global, float* cs, float* sn, const RopeOracle& oracle) {
    const auto dim = global ? 512U : 256U, active = global ? 64U : 128U;
    for (std::uint32_t i = 0; i < dim / 2U; ++i) {
        cs[i] = 1.0F; sn[i] = 0.0F;
        if (i < active) {
            const float a = float(2U * i) / float(dim);
            const float b = oracle.pow_real(global ? 1000000.0F : 10000.0F, a);
            const float f = 1.0F / b, angle = float(position) * f;
            cs[i] = oracle.cos_real(angle); sn[i] = oracle.sin_real(angle);
        }
    }
}
STRIX_G4_HD inline void apply_rope(float* x, std::uint32_t dim, std::uint32_t active, const float* cs, const float* sn) {
    for (std::uint32_t i = 0; i < active; ++i) {
        const float a = x[i], b = x[i + dim / 2U];
        x[i] = a * cs[i] - b * sn[i];
        x[i + dim / 2U] = b * cs[i] + a * sn[i];
    }
}

} // namespace strix::gemma4
