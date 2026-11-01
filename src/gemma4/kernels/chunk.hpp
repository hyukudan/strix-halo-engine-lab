#pragma once
#include "internal.hpp"

namespace strix::gemma4::gpu {
// A full lane chunk never crosses a packed sub-block. Decode its metadata once;
// keep d * float(scale * code), rather than changing the rounded FP32 product.
template <bool Streaming = false> __device__ inline uint4 chunk_bytes(const std::uint8_t* p) {
    if constexpr (Streaming) {
        // Packed formats need not align vector payloads. Keep byte alignment;
        // the cache policy does not change the bytes or their interpretation.
        typedef unsigned Words __attribute__((ext_vector_type(4), aligned(1), may_alias));
        const auto words = __builtin_nontemporal_load(reinterpret_cast<const Words*>(p));
        return make_uint4(words[0], words[1], words[2], words[3]);
    }
    uint4 v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
__device__ inline unsigned chunk_word(uint4 v, unsigned i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
__device__ inline unsigned chunk_byte(uint4 v, unsigned i) {
    return (chunk_word(v, i / 4U) >> ((i % 4U) * 8U)) & 255U;
}
// Half scales: the hardware conversion is exact for every non-NaN half while
// half denormals are preserved (all kernels of this target run in that mode),
// so it equals the shared branchy decoder. Model tensors are finite.
__device__ inline float half_scale(std::uint16_t h, bool hardware) {
    return hardware ? float(__builtin_bit_cast(_Float16, h)) : fp16_value(h);
}
template <WeightFormat Format, bool Streaming = false> struct WeightChunk;
template <bool Streaming> struct WeightChunk<WeightFormat::bf16, Streaming> {
    uint4 a, b;
    __device__ WeightChunk() = default;
    __device__ WeightChunk(const std::uint8_t* p, unsigned i, bool = false, bool = false)
        : a(chunk_bytes<Streaming>(p + 2ULL * i)), b(chunk_bytes<Streaming>(p + 2ULL * i + 16)) {}
    __device__ float at(unsigned e) const {
        const auto w = chunk_word(e < 8U ? a : b, (e % 8U) / 2U);
        return bf16_value(std::uint16_t(w >> ((e % 2U) * 16U)));
    }
};
template <bool Streaming> struct WeightChunk<WeightFormat::f32, Streaming> {
    uint4 a, b, c, d;
    __device__ WeightChunk() = default;
    __device__ WeightChunk(const std::uint8_t* p, unsigned i, bool = false, bool = false)
        : a(chunk_bytes<Streaming>(p + 4ULL * i)), b(chunk_bytes<Streaming>(p + 4ULL * i + 16)),
          c(chunk_bytes<Streaming>(p + 4ULL * i + 32)), d(chunk_bytes<Streaming>(p + 4ULL * i + 48)) {}
    __device__ float at(unsigned e) const {
        return bit_cast<float>(chunk_word(e < 4U ? a : e < 8U ? b : e < 12U ? c : d, e % 4U));
    }
};
template <bool Streaming> struct WeightChunk<WeightFormat::q8_0, Streaming> {
    uint4 codes;
    float d;
    __device__ WeightChunk() = default;
    __device__ WeightChunk(const std::uint8_t* p, unsigned i, bool = false, bool hardware = false) {
        p += std::size_t(i / 32U) * 34;
        d = half_scale(load16(p), hardware);
        codes = chunk_bytes<Streaming>(p + 2 + i % 32U);
    }
    __device__ float at(unsigned e) const {
        const auto q = chunk_byte(codes, e);
        return d * float(q < 128U ? int(q) : int(q) - 256);
    }
};
template <bool Streaming> struct WeightChunk<WeightFormat::q4_k, Streaming> {
    uint4 codes;
    float d, minimum;
    int scale;
    unsigned shift;
    __device__ WeightChunk() = default;
    __device__ WeightChunk(const std::uint8_t* p, unsigned i, bool wide_header = false, bool hardware = false) {
        p += std::size_t(i / 256U) * 144;
        i %= 256U;
        const auto j = i / 32U;
        if (wide_header) {
            const auto h = chunk_bytes<Streaming>(p);
            codes = chunk_bytes<Streaming>(p + 16U + (i / 64U) * 32U + i % 32U);
            scale = int(j < 4U ? chunk_byte(h, 4U + j) & 63U
                              : (chunk_byte(h, 8U + j) & 15U) | ((chunk_byte(h, j) >> 6U) << 4U));
            const int m = int(j < 4U ? chunk_byte(h, 8U + j) & 63U
                                     : (chunk_byte(h, 8U + j) >> 4U) | ((chunk_byte(h, 4U + j) >> 6U) << 4U));
            d = half_scale(std::uint16_t(h.x), hardware);
            minimum = half_scale(std::uint16_t(h.x >> 16U), hardware) * float(m);
            shift = (j & 1U) * 4U;
            return;
        }
        const auto* s = p + 4;
        scale = j < 4U ? s[j] & 63 : (s[j + 4U] & 15) | ((s[j - 4U] >> 6U) << 4U);
        const int m = j < 4U ? s[j + 4U] & 63 : (s[j + 4U] >> 4U) | ((s[j] >> 6U) << 4U);
        d = half_scale(load16(p), hardware);
        minimum = half_scale(load16(p + 2), hardware) * float(m);
        shift = (j & 1U) * 4U;
        codes = chunk_bytes<Streaming>(p + 16U + (i / 64U) * 32U + i % 32U);
    }
    __device__ float at(unsigned e) const {
        const int q = int((chunk_byte(codes, e) >> shift) & 15U);
        return d * float(scale * q) - minimum;
    }
    // The same value: scale < 64 (6-bit fields, constructor) and q < 16
    // (4-bit code), so the 24-bit multiply returns the same integer at full
    // rate (v_mul_u32_u24, not v_mul_lo_u32). A format reusing this decoder
    // must keep both operands inside these bounds.
    static constexpr unsigned scale_limit = 64U, code_limit = 16U;
    static_assert(scale_limit * code_limit <= (1U << 24U), "Q4_K scale * code must fit the unsigned 24-bit multiply");
    __device__ float at24(unsigned e) const {
        const unsigned q = (chunk_byte(codes, e) >> shift) & 15U;
        return d * float(int(__umul24(unsigned(scale), q))) - minimum;
    }
};
template <bool Streaming> struct WeightChunk<WeightFormat::q6_k, Streaming> {
    uint4 low, high;
    float d;
    int scale;
    unsigned low_shift, high_shift;
    __device__ WeightChunk() = default;
    __device__ WeightChunk(const std::uint8_t* p, unsigned i, bool wide_header = false, bool hardware = false) {
        p += std::size_t(i / 256U) * 210;
        i %= 256U;
        const auto half = i / 128U, quarter = (i % 128U) / 32U, l = i % 32U;
        low = chunk_bytes<Streaming>(p + half * 64U + l + (quarter & 1U) * 32U);
        high = chunk_bytes<Streaming>(p + 128U + half * 32U + l);
        const auto raw = wide_header ? chunk_byte(chunk_bytes<Streaming>(p + 192U), i / 16U) : unsigned(p[192U + i / 16U]);
        scale = raw < 128U ? int(raw) : int(raw) - 256;
        d = half_scale(load16(p + 208), hardware);
        low_shift = quarter >= 2U ? 4U : 0U;
        high_shift = 2U * quarter;
    }
    __device__ float at(unsigned e) const {
        const int q = int(((chunk_byte(low, e) >> low_shift) & 15U) |
                          (((chunk_byte(high, e) >> high_shift) & 3U) << 4U));
        return d * float(scale * (q - 32));
    }
    // The same value: scale is a signed byte (-128..127) and q - 32 lies in
    // -32..31 (6-bit code), so the signed 24-bit multiply returns the same
    // integer at full rate (v_mul_i32_i24). A format reusing this decoder must
    // keep both operands inside these bounds.
    static constexpr int scale_limit = 128, code_limit = 32;
    static_assert(scale_limit * code_limit < (1 << 23), "Q6_K scale * (code - 32) must fit the signed 24-bit multiply");
    __device__ float at24(unsigned e) const {
        const int q = int(((chunk_byte(low, e) >> low_shift) & 15U) |
                          (((chunk_byte(high, e) >> high_shift) & 3U) << 4U));
        return d * float(__mul24(scale, q - 32));
    }
};
} // namespace strix::gemma4::gpu
