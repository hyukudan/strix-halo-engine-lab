#pragma once
// kv-int8-tok-v1 arithmetic steps (host / device neutral; contract strix/kv_storage.hpp Form::int8_token, HIP writer
// kv_int8_store_head in src/hip_indexed_attention_launcher.hip, the host reference below).
//
// One (token, KV head, K/V half) row of n features: x_i = BF16 round-to-nearest-even of the FP32 source (the incumbent
// bit formula, round_bf16); amax = max over i of magnitude(src_i, x_i) folded with max_step in any order; scale
// s = amax / 127 when amax is finite, else NaN; code_i = clamp(rint(x_i / s), -127, 127) when s > 0, else 0; the reader's
// value is float(code_i) * s.
//
// Domain (NUM-10): a finite row is encoded exactly as before (same maximum, same scale, same codes), so the form keeps
// its identity on every finite input. A row holding a NaN or an infinity (in the source, or a finite source whose BF16
// rounding overflows) gets scale NaN and codes 0, so every feature of that row decodes to NaN: non-finite in ->
// non-finite out, the class of the BF16 form, never a finite wrong value. The incumbent reduction (fmaxf) dropped a NaN
// and clamped its code to -127 with a finite scale, and an infinite maximum gave s = +Inf, rebuilding -127 * Inf = -Inf
// (sign flipped) or 0 * Inf = NaN from the row's codes. A writer that owns a status word raises status_nonfinite for
// such a row (encode_row_ref returns it).
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_KV8T_HD __host__ __device__
#else
#define STRIX_KV8T_HD
#endif

namespace strix::attn::kv_int8_token {

inline constexpr unsigned status_nonfinite = 8U;  // the decode graph's arithmetic-failure bit
inline constexpr float code_limit = 127.0F;

STRIX_KV8T_HD inline bool finite(float x) {
    return (__builtin_bit_cast(std::uint32_t, x) & 0x7f800000U) != 0x7f800000U;
}
// BF16 bits of x, round to nearest even by the incumbent formula (a NaN source is classified by magnitude, not here).
STRIX_KV8T_HD inline std::uint16_t bf16_bits(float x) {
    const std::uint32_t bits = __builtin_bit_cast(std::uint32_t, x);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
STRIX_KV8T_HD inline float round_bf16(float x) {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(bf16_bits(x)) << 16U);
}
// Magnitude of one feature for the maximum: |x| of the rounded value, NaN when the source or its rounding is not finite.
STRIX_KV8T_HD inline float magnitude(float source, float rounded) {
    const float a = __builtin_bit_cast(float, __builtin_bit_cast(std::uint32_t, rounded) & 0x7fffffffU);
    return finite(source) && finite(rounded) ? a : __builtin_nanf("");
}
// One step of the maximum (also the combine of two partial maxima): a NaN wins over any number, so the result does not
// depend on the order; on numbers it equals fmaxf.
STRIX_KV8T_HD inline float max_step(float amax, float a) { return (a > amax || a != a) ? a : amax; }
STRIX_KV8T_HD inline float scale(float amax) { return finite(amax) ? amax / code_limit : __builtin_nanf(""); }
STRIX_KV8T_HD inline float code(float x, float s) {
    if (!(s > 0.0F))
        return 0.0F;
    const float c = __builtin_rintf(x / s);
    return c > code_limit ? code_limit : (c < -code_limit ? -code_limit : c);
}
STRIX_KV8T_HD inline float decode(int code_value, float s) { return static_cast<float>(code_value) * s; }

// Host reference of one row: codes and scale for n FP32 source features; returns status_nonfinite or 0.
inline unsigned encode_row_ref(const float* source, unsigned n, std::int8_t* codes, float& s) {
    float amax = 0.0F;
    for (unsigned i = 0; i < n; ++i)
        amax = max_step(amax, magnitude(source[i], round_bf16(source[i])));
    s = scale(amax);
    for (unsigned i = 0; i < n; ++i)
        codes[i] = static_cast<std::int8_t>(code(round_bf16(source[i]), s));
    return finite(s) ? 0U : status_nonfinite;
}

}  // namespace strix::attn::kv_int8_token

#undef STRIX_KV8T_HD
