#pragma once
// Host C++ reference of the FP32 prefill identity (fp32_identity_device.inc), and a portable deterministic test-data
// generator, shared by the gfx1151 and RTX PRO tests so both platforms are checked against the same bits.
// Compile with -ffp-contract=off (MSVC: /fp:strict).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace strix::minimax::rtxpf::ref {

inline float f32_of(std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline std::uint32_t bits_of(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }

inline float half_to_float(std::uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : (e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) : std::ldexp(static_cast<float>(m | 1024), e - 25));
    return s ? -v : v;
}
// IEEE RNE float -> half (finite inputs within half range or subnormal).
inline std::uint16_t float_to_half(float f) {
    const std::uint32_t b = bits_of(f);
    const std::uint16_t sign = static_cast<std::uint16_t>((b >> 16) & 0x8000U);
    const std::uint32_t a = b & 0x7fffffffU;
    if (a >= 0x7f800000U) return static_cast<std::uint16_t>(sign | 0x7c00U | (a > 0x7f800000U ? 0x200U : 0));
    const int e = static_cast<int>(a >> 23) - 127;
    if (e > 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
    std::uint32_t mant = (a & 0x7fffffU) | 0x800000U;
    int shift;  // bits to drop from the 24-bit significand
    std::uint32_t base;
    if (e >= -14) { shift = 13; base = static_cast<std::uint32_t>(e + 15) << 10; }
    else { shift = 13 + (-14 - e); base = 0; if (shift > 24) return sign; }
    if (a == 0) return sign;
    const std::uint32_t keep = mant >> shift, rem = mant & ((1U << shift) - 1U), half = 1U << (shift - 1);
    std::uint32_t r = keep + ((rem > half || (rem == half && (keep & 1U))) ? 1U : 0U);
    std::uint32_t out = e >= -14 ? base + (r - 0x400U) : r;  // normal: drop the implicit bit (carry into the exponent works)
    return static_cast<std::uint16_t>(sign | out);
}
inline float round_half(float v) { return half_to_float(float_to_half(v)); }

inline std::uint32_t tile_state(const std::uint32_t* w, int bits, int position) {
    const int words = bits * 8;
    const int end = (position + 1) * bits + 256 * bits;
    const int start = end - 16;
    const std::uint32_t earlier = w[(start / 32) % words];
    const std::uint32_t later = w[((end - 1) / 32) % words];
    const std::uint64_t pair = (static_cast<std::uint64_t>(earlier) << 32) | later;
    return static_cast<std::uint32_t>(pair >> ((32 - (end & 31)) & 31)) & 65535U;
}
inline float mul1(std::uint32_t state) {
    const std::uint32_t p = state * 0x83dcd12dU;
    const unsigned sum = (p & 255U) + ((p >> 8) & 255U) + ((p >> 16) & 255U) + (p >> 24);
    return round_half(std::fmaf(static_cast<float>(1024 + sum), 0x1.bb8p-8F, -0x1.4c4p+3F));
}
inline void pos_kn(int lane, int j, int& k, int& n) {
    k = 2 * (lane % 4) + (j % 2) + 8 * ((j / 2) % 2);
    n = lane / 4 + 8 * (j / 4);
}
// Dense FP32 copy of an EXL3 matrix: w[k * output + n].
inline std::vector<float> decode(const std::uint32_t* trellis, unsigned input, unsigned output, int bits) {
    std::vector<float> w(static_cast<std::size_t>(input) * output);
    for (unsigned tk = 0; tk < input / 16; ++tk)
        for (unsigned tn = 0; tn < output / 16; ++tn) {
            const std::uint32_t* t = trellis + (static_cast<std::size_t>(tk) * (output / 16) + tn) * static_cast<std::size_t>(bits) * 8;
            for (int lane = 0; lane < 32; ++lane)
                for (int j = 0; j < 8; ++j) {
                    int k, n;
                    pos_kn(lane, j, k, n);
                    w[static_cast<std::size_t>(tk * 16 + static_cast<unsigned>(k)) * output + tn * 16 + static_cast<unsigned>(n)] = mul1(tile_state(t, bits, 8 * lane + j));
                }
        }
    return w;
}
// hadamard128 on 128 values laid out as v[lane + 32 j].
inline void hadamard128(float* x) {
    float v[32][4];
    for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 4; ++j) v[l][j] = x[l + 32 * j];
    for (int s = 1; s < 32; s *= 2) {
        float n[32][4];
        for (int l = 0; l < 32; ++l)
            for (int j = 0; j < 4; ++j) n[l][j] = (l & s) ? v[l ^ s][j] - v[l][j] : v[l][j] + v[l ^ s][j];
        std::memcpy(v, n, sizeof(v));
    }
    for (int s = 1; s < 4; s *= 2)
        for (int l = 0; l < 32; ++l)
            for (int j = 0; j < 4; ++j)
                if (!(j & s)) {
                    const float a = v[l][j], b = v[l][j + s];
                    v[l][j] = a + b;
                    v[l][j + s] = a - b;
                }
    for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 4; ++j) x[l + 32 * j] = v[l][j] * 0x1.6a09e6p-4F;
}
// One row of one linear: y[0..output) from x[0..input).
inline void linear_row(const float* w, const std::uint16_t* suh, const std::uint16_t* svh, unsigned input, unsigned output, const float* x,
                       float* y) {
    std::vector<float> xt(input), raw(output);
    for (unsigned c = 0; c < input; ++c) xt[c] = x[c] * half_to_float(suh[c]);
    for (unsigned b = 0; b < input / 128; ++b) hadamard128(xt.data() + b * 128);
    for (unsigned o = 0; o < output; ++o) {
        float acc = 0.0F;
        for (unsigned k = 0; k < input; ++k) acc = std::fmaf(xt[k], w[static_cast<std::size_t>(k) * output + o], acc);
        raw[o] = acc;
    }
    for (unsigned b = 0; b < output / 128; ++b) hadamard128(raw.data() + b * 128);
    for (unsigned o = 0; o < output; ++o) y[o] = raw[o] * half_to_float(svh[o]);
}

inline float exp_f(float x) {  // strix/minimax_msa.hpp exp_f
    if (x < -87.0F) return 0.0F;
    if (x > 88.0F) return std::numeric_limits<float>::infinity();
    const float n = std::nearbyint(x * 1.44269504088896341F);
    float r = std::fmaf(n, -0.693145751953125F, x);
    r = std::fmaf(n, -1.42860676533018e-06F, r);
    float p = 2.48015873e-05F;
    p = std::fmaf(p, r, 1.98412698e-04F);
    p = std::fmaf(p, r, 1.38888889e-03F);
    p = std::fmaf(p, r, 8.33333333e-03F);
    p = std::fmaf(p, r, 4.16666667e-02F);
    p = std::fmaf(p, r, 1.66666667e-01F);
    p = std::fmaf(p, r, 0.5F);
    p = std::fmaf(p, r, 1.0F);
    p = std::fmaf(p, r, 1.0F);
    return p * f32_of(static_cast<std::uint32_t>(static_cast<int>(n) + 127) << 23U);
}
inline float swiglu(float g, float u) {
    const float gg = g < 7.0F ? g : 7.0F;
    const float uu = u < -7.0F ? -7.0F : (u > 7.0F ? 7.0F : u);
    return gg * (1.0F / (1.0F + exp_f(-1.702F * gg))) * (uu + 1.0F);
}

// Portable generator (splitmix64) and values: identical on every compiler / standard library.
struct Gen {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    // float in (-2^e, 2^e) with a random exponent spread of 2^-spread .. 1 and full mantissa
    float value(int e, int spread) {
        const std::uint64_t r = next();
        const int ex = 127 + e - 1 - static_cast<int>(r % static_cast<std::uint64_t>(spread + 1));
        return f32_of((static_cast<std::uint32_t>((r >> 8) & 1U) << 31) | (static_cast<std::uint32_t>(ex) << 23) |
                      static_cast<std::uint32_t>((r >> 9) & 0x7fffffU));
    }
    std::uint16_t half_scale() { return float_to_half(0.5F + static_cast<float>(next() % 1000U) / 1000.0F); }
};

inline std::uint64_t fnv(const float* p, std::size_t n, std::uint64_t h = 1469598103934665603ULL) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t b = bits_of(p[i]);
        for (int k = 0; k < 4; ++k) { h ^= (b >> (8 * k)) & 255U; h *= 1099511628211ULL; }
    }
    return h;
}

}  // namespace strix::minimax::rtxpf::ref
