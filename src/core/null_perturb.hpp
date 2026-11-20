#pragma once
// Null perturbations for the FAST quality gate (v2 nulls), every engine and backend.
//
// A gate null is the reference arithmetic plus one perturbation of the size of the arithmetic's own rounding. Every HIP
// prefill linear stages its input as FP16 (Hadamard-transformed, then rounded), so one f32 ulp added to an activation is
// absorbed at the next consumer with probability ~1 - 2^-13: the "null" then equals the reference bitwise, its KL is 0
// and the null-relative gate limit falls to its floor. fp16_ulp_up_bits returns the next FP16 value above round-to-FP16
// of x (as f32 bits): the smallest move of x whose FP16 rounding always differs from x's. Non-finite x is unchanged.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_NULL_PERTURB_HD __host__ __device__
#else
#define STRIX_NULL_PERTURB_HD
#endif

namespace strix::core {

// FP16 spacing at magnitude a >= 0: 2^(floor(log2 a) - 10) in the normal range, 2^-24 below it.
STRIX_NULL_PERTURB_HD inline float fp16_spacing(float a) {
    const int e = static_cast<int>((__builtin_bit_cast(std::uint32_t, a) >> 23) & 0xffU) - 127;
    return __builtin_bit_cast(float, static_cast<std::uint32_t>((e < -14 ? -14 : e) - 10 + 127) << 23);
}

// Round to the nearest FP16 value, ties to even (finite x within the FP16 range); exact in f32: x / spacing is a
// power-of-two scaling and |x / spacing| < 2^12, so adding and subtracting 1.5 * 2^23 rounds it to an integer.
STRIX_NULL_PERTURB_HD inline float round_fp16_value(float x) {
    const float s = fp16_spacing(x < 0.0f ? -x : x);
    const float magic = 12582912.0f;
    const float q = x / s;
    return ((q + magic) - magic) * s;
}

STRIX_NULL_PERTURB_HD inline std::uint32_t fp16_ulp_up_bits(std::uint32_t bits) {
    if (((bits >> 23) & 0xffU) == 0xffU) return bits;
    const float r = round_fp16_value(__builtin_bit_cast(float, bits));
    return __builtin_bit_cast(std::uint32_t, r + fp16_spacing(r < 0.0f ? -r : r));
}

// Gate null site of the dense engines (test hook, host-only): "LAYER:POS:ELEM" = one FP16 ulp (fp16_ulp_up_bits) on
// residual element ELEM of the row at absolute position POS at the input of layer LAYER. Fields are decimal;
// LAYER < layers, ELEM < width, POS < 2^32. Anything else throws std::invalid_argument naming the variable.
struct NullSite {
    unsigned layer = 0;
    std::uint64_t pos = 0;
    unsigned elem = 0;
};

inline NullSite parse_null_site(std::string_view text, unsigned layers, unsigned width, const char* name) {
    const auto bad = [&] {
        return std::invalid_argument(std::string(name) + " must be LAYER:POS:ELEM (decimal, LAYER < " + std::to_string(layers) +
                                     ", ELEM < " + std::to_string(width) + "), got '" + std::string(text) + "'");
    };
    std::uint64_t v[3] = {};
    for (unsigned i = 0; i < 3U; ++i) {
        const auto colon = text.find(':');
        if ((colon == std::string_view::npos) != (i == 2U)) throw bad();
        const std::string_view t = text.substr(0, colon);
        if (t.empty() || t.size() > 10U) throw bad();
        for (const char c : t) {
            if (c < '0' || c > '9') throw bad();
            v[i] = v[i] * 10U + static_cast<std::uint64_t>(c - '0');
        }
        if (i < 2U) text.remove_prefix(colon + 1);
    }
    if (v[0] >= layers || v[2] >= width || v[1] > 0xffffffffULL) throw bad();
    return NullSite{static_cast<unsigned>(v[0]), v[1], static_cast<unsigned>(v[2])};
}

}  // namespace strix::core
