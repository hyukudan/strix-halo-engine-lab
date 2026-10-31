#pragma once

// g4-a8-v1: a NAMED, opt-in numerical identity. It is g4-v1 (same graph, model bytes, tensor formats,
// rotary tables, FP32 KV, norms, attention, PLE, softcap and ordered selection) except for the dot
// product of every Q8_0, Q4_K and Q6_K linear, which uses int8 activations and integer products:
//
// 1. Activation quantization (A8G32), per input row and per 32-element group g (every quantized
//    linear width is a multiple of 256):
//      amax  = max |x_i| over the group (exact; inputs are finite, nonfinite execution fails);
//      delta = RN(amax / 127)                         (IEEE FP32 division, round to nearest even);
//      if delta == +0 (all-zero or underflowing group): delta = 1, every code 0;
//      else  code_i = clamp(rint(RN(x_i / delta)), -127, 127) with rint ties-to-even;
//      S_g   = sum of the group's 32 codes (exact int32).
// 2. Integer products, exact in int32 (any evaluation order or lane split gives the same value):
//      Q8_0 (block = group, signed code c, FP16 scale d):        I = sum c_i * a_i
//      Q4_K (sub-block j of a super-block; unsigned 4-bit q, unsigned 6-bit scale sc and min m):
//                                                                I = sum q_i * a_i
//      Q6_K (q in 0..63, signed 8-bit scales s0/s1 of the group's two 16-element halves):
//                                                                J = s0 * sum_{i<16}(q_i-32) a_i
//                                                                  + s1 * sum_{i>=16}(q_i-32) a_i
//    Bounds: |sc*I| <= 3,840,480; |m*S| <= 256,032; |J| <= 16,646,144; |I(Q8_0)| <= 520,192, so
//    every integer converts to FP32 exactly.
// 3. FP32 group term, each displayed operation rounded separately (no contraction):
//      Q8_0: c_g = delta * (d * float(I))
//      Q4_K: c_g = delta * ((d * float(sc*I)) - (dmin * float(m*S)))
//      Q6_K: c_g = delta * (d * float(J))
//    d and dmin are the exact FP32 expansions of the FP16 header fields.
// 4. Fixed reduction: 16 slots; slot p accumulates a = RN(a + c_g) from +0 over g = p, p+16, ...
//    ascending; then the slots are combined by the tree v[l] = v[l] + v[l+d] for d = 8,4,2,1 and the
//    result is v[0]. Row count, slot/output tiling, peers and padding never change this order, so a
//    causal R-row pass equals R serial passes (row invariance is structural, as in g4-v1).
// 5. Epilogues (GELU-tanh product, PLE activation, softcap, norms, residuals) are the g4-v1 FP32
//    operations applied to that dot value. BF16/F32 linears (model PLE projection) keep the g4-v1
//    chunk16 recipe.
// The host reference below is the oracle for device bytes; HF fidelity is a separate qualification.
#include "gemma4/contracts/g4_identity.hpp"
#include "gemma4/contracts/g4_types.hpp"

namespace strix::gemma4::a8 {

inline constexpr char arithmetic_id[] = "g4-a8-v1";
inline constexpr char recipe[] =
    "g4-a8-v1:a8g32:delta=rn(amax/127):zero-or-underflow->delta1-codes0:code=clamp127(rint-even(rn(x/delta))):"
    "dot-i8xi8-i32-exact:q8_0=delta*(d*I):q4_k=delta*((d*(sc*I))-(dmin*(m*S))):q6_k=delta*(d*(s0*I0+s1*I1)):"
    "slots16-groups-ascending-from+0:tree-8-4-2-1:epilogues=g4-v1:bf16-f32-linears=g4-v1";
inline constexpr std::uint32_t group_width = 32, slots = 16;

struct Meta {
    float delta;
    std::int32_t sum;
};

STRIX_G4_HD inline bool quantized(WeightFormat f) {
    return f == WeightFormat::q8_0 || f == WeightFormat::q4_k || f == WeightFormat::q6_k;
}
STRIX_G4_HD inline float round_even(float x) {
#if defined(__HIPCC__)
    return __builtin_rintf(x);
#else
    return std::rint(x);
#endif
}
// One A8G32 group: 32 FP32 inputs -> 32 codes, delta and code sum.
STRIX_G4_HD inline Meta quantize_group(const float* x, std::int8_t* codes) {
    float amax = 0.0F;
    for (std::uint32_t i = 0; i < group_width; ++i) {
        const float a = bit_cast<float>(bit_cast<std::uint32_t>(x[i]) & 0x7fffffffU);
        amax = a > amax ? a : amax;
    }
    const float delta = amax / 127.0F;
    if (delta == 0.0F) {
        for (std::uint32_t i = 0; i < group_width; ++i)
            codes[i] = 0;
        return {1.0F, 0};
    }
    std::int32_t sum = 0;
    for (std::uint32_t i = 0; i < group_width; ++i) {
        float q = round_even(x[i] / delta);
        q = q > 127.0F ? 127.0F : (q < -127.0F ? -127.0F : q);
        codes[i] = std::int8_t(int(q));
        sum += codes[i];
    }
    return {delta, sum};
}
// Q4_K sub-block j (0..7) of one 144-byte super-block.
STRIX_G4_HD inline void q4_k_factors(const std::uint8_t* b, std::uint32_t j, int& sc, int& m) {
    const auto* s = b + 4;
    sc = j < 4U ? s[j] & 63 : (s[j + 4U] & 15) | ((s[j - 4U] >> 6U) << 4U);
    m = j < 4U ? s[j + 4U] & 63 : (s[j + 4U] >> 4U) | ((s[j] >> 6U) << 4U);
}
STRIX_G4_HD inline float q4_k_term(float d, float dmin, int sc, int m, std::int32_t integer, Meta a) {
    const float u = d * float(sc * integer);
    const float v = dmin * float(m * a.sum);
    return a.delta * (u - v);
}
STRIX_G4_HD inline float scaled_term(float d, std::int32_t integer, Meta a) { return a.delta * (d * float(integer)); }

// Group terms from the original packed bytes (scalar decode; the device decodes words, same integers).
STRIX_G4_HD inline float q8_0_group(const std::uint8_t* row, std::uint32_t g, const std::int8_t* a, Meta meta) {
    const auto* b = row + std::size_t(g) * 34U;
    std::int32_t integer = 0;
    for (std::uint32_t i = 0; i < 32U; ++i)
        integer += std::int32_t(std::int8_t(b[2U + i])) * a[i];
    return scaled_term(fp16_value(load16(b)), integer, meta);
}
STRIX_G4_HD inline float q4_k_group(const std::uint8_t* row, std::uint32_t g, const std::int8_t* a, Meta meta) {
    const auto* b = row + std::size_t(g / 8U) * 144U;
    const auto j = g % 8U;
    int sc = 0, m = 0;
    q4_k_factors(b, j, sc, m);
    std::int32_t integer = 0;
    for (std::uint32_t i = 0; i < 32U; ++i)
        integer += std::int32_t((b[16U + (j / 2U) * 32U + i] >> ((j & 1U) * 4U)) & 15U) * a[i];
    return q4_k_term(fp16_value(load16(b)), fp16_value(load16(b + 2)), sc, m, integer, meta);
}
STRIX_G4_HD inline float q6_k_group(const std::uint8_t* row, std::uint32_t g, const std::int8_t* a, Meta meta) {
    const auto* b = row + std::size_t(g / 8U) * 210U;
    const auto j = g % 8U, half = j / 4U, quarter = j % 4U;
    std::int32_t part[2] = {0, 0};
    for (std::uint32_t i = 0; i < 32U; ++i) {
        const auto low = b[half * 64U + i + (quarter & 1U) * 32U];
        const auto high = b[128U + half * 32U + i];
        const int q = ((low >> (quarter >= 2U ? 4U : 0U)) & 15) | (((high >> (2U * quarter)) & 3) << 4);
        part[i / 16U] += (q - 32) * a[i];
    }
    const auto s0 = std::int32_t(std::int8_t(b[192U + 2U * j])), s1 = std::int32_t(std::int8_t(b[193U + 2U * j]));
    return scaled_term(fp16_value(load16(b + 208)), s0 * part[0] + s1 * part[1], meta);
}
STRIX_G4_HD inline float group_term(LinearView w, std::uint32_t o, std::uint32_t g, const std::int8_t* a, Meta meta) {
    const auto* row = w.data + std::size_t(o) * w.row_bytes;
    switch (w.format) {
    case WeightFormat::q8_0:
        return q8_0_group(row, g, a, meta);
    case WeightFormat::q4_k:
        return q4_k_group(row, g, a, meta);
    case WeightFormat::q6_k:
        return q6_k_group(row, g, a, meta);
    default:
        break;
    }
    return bit_cast<float>(0x7fc00000U);
}
// The declared 16-slot ascending accumulation and 8/4/2/1 tree over group terms.
template <class Term> STRIX_G4_HD inline float slot_dot(Term term, std::uint32_t groups) {
    float v[slots] = {};
    for (std::uint32_t p = 0; p < slots; ++p)
        for (std::uint32_t g = p; g < groups; g += slots)
            v[p] = v[p] + term(g);
    for (std::uint32_t d = slots / 2U; d != 0; d >>= 1U)
        for (std::uint32_t l = 0; l < d; ++l)
            v[l] = v[l] + v[l + d];
    return v[0];
}
// Host oracle for one quantized row: codes [width], meta [width / 32].
STRIX_G4_HD inline float dot(LinearView w, std::uint32_t o, const std::int8_t* codes, const Meta* meta) {
    return slot_dot([&](std::uint32_t g) { return group_term(w, o, g, codes + std::size_t(g) * group_width, meta[g]); },
                    w.inputs / group_width);
}

} // namespace strix::gemma4::a8
