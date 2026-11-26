#pragma once
// Shared attention KV shadow contract (host / device neutral; HIP body src/attn_core/cache_shadow.hpp).
//
// A shadow is a second, reduced representation of rows that already live in an FP32 master cache (one row per token
// position, K and V planes of `width` elements each). The master stays the arithmetic reference: exact routes read only
// the master; a FAST reader may read the shadow of rows [0, mirrored) and must convert any newer row from the master
// itself with the same encode, so its values never depend on when the shadow was synchronized. Rows are converted whole,
// in place of the master's index (element i of position p at p * width + i; Int8G16 scales at p * (width / 16) + g).
//
// Forms (each a policy with encode / decode, used by the writer, by the reader's on-the-fly conversion and by the host
// reference):
//   F16      "kv-shadow-f16-v1":  half(x) round-to-nearest-even; decode float(half). An infinite or NaN image raises
//            flag_overflow.
//   Int8G16  "kv-shadow-i8g16-v1": groups of 16 consecutive features of one K or V row. amax = max |x| scanned in feature
//            order with `a > amax` (a NaN never wins); scale s = half(amax / 127) (FP32 division, RNE half), decode
//            scale float(s); code = 0 when float(s) == 0, else clamp(rint(x / float(s)), -127, 127); value = float(s) * code
//            (exact in FP32). A non-finite input raises flag_nonfinite, an infinite or NaN scale flag_overflow; a NaN
//            quotient (only in a flagged group) codes 0. Finite groups lose only bounded magnitude: a group with
//            amax < 2^-25 * 127 (about 3.8e-6) has scale 0 and codes 0, and a subnormal-half scale may saturate codes, so
//            the error of such a group is at most amax per feature (no flag: the values are below the form's resolution).
// Validity of a master row (policy of the writer): Untagged (rows are valid by construction) or Tagged (position tags:
// row p is valid iff tags[p] == p; checked once per row, a mismatch raises flag_tag).
// Failure word: OR of flag_nonfinite = 1, flag_tag = 2, flag_overflow = 4. The shadow's bits are written either way; a
// pass whose failure word is nonzero must not publish the converted rows.
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_CS_HD __host__ __device__
#else
#define STRIX_CS_HD
#endif

namespace strix::attn::cache_shadow {

inline constexpr unsigned flag_nonfinite = 1U, flag_tag = 2U, flag_overflow = 4U;
inline constexpr unsigned group = 16U;  // Int8G16 features per scale

// A half whose exponent field is all ones (infinity or NaN).
STRIX_CS_HD inline bool half_special(std::uint16_t bits) { return (bits & 0x7c00U) == 0x7c00U; }
STRIX_CS_HD inline bool finite(float x) { return (__builtin_bit_cast(std::uint32_t, x) & 0x7f800000U) != 0x7f800000U; }

struct F16 {
    using Code = std::uint16_t;
    STRIX_CS_HD static std::uint16_t encode(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }
    STRIX_CS_HD static float decode(std::uint16_t bits) { return float(__builtin_bit_cast(_Float16, bits)); }
};

struct Int8G16 {
    using Code = std::int8_t;
    STRIX_CS_HD static float magnitude(float x) {
        return __builtin_bit_cast(float, __builtin_bit_cast(std::uint32_t, x) & 0x7fffffffU);
    }
    // One step of the group maximum (feature order; also the combine of two partial maxima).
    STRIX_CS_HD static float max_step(float amax, float a) { return a > amax ? a : amax; }
    STRIX_CS_HD static std::uint16_t scale_bits(float amax) {
        return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(amax / 127.0F));
    }
    STRIX_CS_HD static float scale_value(std::uint16_t bits) { return float(__builtin_bit_cast(_Float16, bits)); }
    STRIX_CS_HD static int code(float x, float d) {
        if (d == 0.0F)
            return 0;
        float c = __builtin_rintf(x / d);
        if (c != c)
            return 0; // NaN quotient: x or the scale not finite (encode_group has raised its flag); no UB conversion
        c = c > 127.0F ? 127.0F : (c < -127.0F ? -127.0F : c);
        return int(c);
    }
    // code: the int8 code widened to int (callers write int(std::int8_t(byte))).
    STRIX_CS_HD static float decode(int code, float d) { return d * float(code); }
};

struct Untagged {
    static constexpr bool checked = false;
};
struct Tagged {
    static constexpr bool checked = true;
};

// Host reference of the writer for rows [from, from + rows) of K and V (master [slots][width] FP32 each). Returns the
// failure word the device writer ORs for the same rows; the shadow bits are the same either way.
template <class Tags = Untagged>
inline unsigned mirror_f16_ref(const float* k, const float* v, const std::uint64_t* tags, std::uint64_t from,
                               unsigned rows, unsigned width, std::uint16_t* hk, std::uint16_t* hv) {
    unsigned failure = 0U;
    for (std::uint64_t p = from; p < from + rows; ++p) {
        if constexpr (Tags::checked)
            if (tags[p] != p)
                failure |= flag_tag;
        for (unsigned i = 0; i < width; ++i) {
            const auto at = static_cast<std::size_t>(p) * width + i;
            hk[at] = F16::encode(k[at]);
            hv[at] = F16::encode(v[at]);
            if (half_special(hk[at]) || half_special(hv[at]))
                failure |= flag_overflow;
        }
    }
    return failure;
}

// One Int8G16 group of `x` (16 values): codes and the scale bits; returns its failure bits.
STRIX_CS_HD inline unsigned encode_group(const float* x, std::int8_t* codes, std::uint16_t& scale) {
    unsigned failure = 0U;
    float amax = 0.0F;
    for (unsigned j = 0; j < group; ++j) {
        if (!finite(x[j]))
            failure |= flag_nonfinite;
        amax = Int8G16::max_step(amax, Int8G16::magnitude(x[j]));
    }
    scale = Int8G16::scale_bits(amax);
    if (half_special(scale))
        failure |= flag_overflow;
    const float d = Int8G16::scale_value(scale);
    for (unsigned j = 0; j < group; ++j)
        codes[j] = std::int8_t(Int8G16::code(x[j], d));
    return failure;
}

template <class Tags = Untagged>
inline unsigned mirror_i8g16_ref(const float* k, const float* v, const std::uint64_t* tags, std::uint64_t from,
                                 unsigned rows, unsigned width, std::int8_t* ck, std::int8_t* cv, std::uint16_t* sk,
                                 std::uint16_t* sv) {
    unsigned failure = 0U;
    const unsigned groups = width / group;
    for (std::uint64_t p = from; p < from + rows; ++p) {
        if constexpr (Tags::checked)
            if (tags[p] != p)
                failure |= flag_tag;
        for (unsigned g = 0; g < groups; ++g) {
            const auto at = static_cast<std::size_t>(p) * width + std::size_t(g) * group;
            const auto s = static_cast<std::size_t>(p) * groups + g;
            failure |= encode_group(k + at, ck + at, sk[s]);
            failure |= encode_group(v + at, cv + at, sv[s]);
        }
    }
    return failure;
}

}  // namespace strix::attn::cache_shadow
