#pragma once
// Shared 16-weight chunk decoders for the small-row GEMV sweeps (gemv_rows_direct.inc binds
// GEMV_ROWS_CHUNK / GEMV_ROWS_LOAD_CHUNK to these). One copy for BF16 and Q8_0 rows:
//   bf16  weight = float(bits << 16)                        (exact)
//   q8_0  34-byte blocks {f16 d, int8 q[32]}; weight = d * float(q)   (one FP32 product, as the g4 chunk decoder)
//   q4_g64 / q6_g64  128-element superblocks (strix/weight/host_lowbit.hpp): weight = d * float(q), d the FP16 scale of
//         the element's 64-group; a 16-weight chunk never crosses a group, its code words load 8 + 4 bytes aligned.
//         This is the FP32 sweep decoder. The opt-in FP16 tile decodes integer operands via lowbit_f16_staging.hpp.
//   e4m3_b128  row-local 132-byte blocks {f32 s, e4m3 q[128]} (strix/weight/formats/fp8_block.hpp pack_row_blocks: a
//         128 x 128 block-scaled FP8 checkpoint matrix with its tile scale copied into each row); weight =
//         e4m3(q) * s, the one FP32 product of the host decode (BlockScaledE4M3::decode_row), so the same weight bits.
#include "strix/weight/formats/scalar_float.hpp"
#include "strix/weight/host_lowbit.hpp"

#include <cstddef>
#include <cstdint>
#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#define STRIX_WEIGHT_CHUNK_DEVICE __device__
#define STRIX_WEIGHT_CHUNK_INLINE __device__ __forceinline__
#else
#include <bit>
#include <cstring>
#define STRIX_WEIGHT_CHUNK_DEVICE
#define STRIX_WEIGHT_CHUNK_INLINE inline
#endif

namespace strix::core::weight_decode {

enum class Format : std::uint8_t { bf16, q8_0, q4_g64, q6_g64, e4m3_b128 };
enum class HalfScale : std::uint8_t { hardware, software };

#if !defined(__HIPCC__) && !defined(__HIP__)
struct uint4 { unsigned x, y, z, w; };
#endif

STRIX_WEIGHT_CHUNK_INLINE uint4 load_chunk_bytes(const std::uint8_t* p) {
    uint4 v;
#if defined(__HIPCC__) || defined(__HIP__)
    __builtin_memcpy(&v, p, sizeof(v));
#else
    std::memcpy(&v, p, sizeof(v));
#endif
    return v;
}
STRIX_WEIGHT_CHUNK_INLINE unsigned word_of(uint4 v, unsigned i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
STRIX_WEIGHT_CHUNK_INLINE float bf16_to_float(std::uint16_t h) {
#if defined(__HIPCC__) || defined(__HIP__)
    return __builtin_bit_cast(float, std::uint32_t(h) << 16U);
#else
    return std::bit_cast<float>(std::uint32_t(h) << 16U);
#endif
}

template <bool Streaming>
STRIX_WEIGHT_CHUNK_INLINE uint4 load_chunk(const std::uint8_t* p) {
#if defined(__HIPCC__) || defined(__HIP__)
    if constexpr (Streaming) {
        // Q8_0 payloads need not be vector-aligned.
        typedef unsigned Words __attribute__((ext_vector_type(4), aligned(1), may_alias));
        const auto words = __builtin_nontemporal_load(reinterpret_cast<const Words*>(p));
        return make_uint4(words[0], words[1], words[2], words[3]);
    }
#endif
    return load_chunk_bytes(p);
}

STRIX_WEIGHT_CHUNK_INLINE std::uint32_t software_half_bits(std::uint16_t h) {
    const auto sign = std::uint32_t(h & 0x8000U) << 16U;
    const std::uint32_t e = (h >> 10U) & 31U;
    std::uint32_t m = h & 1023U;
    if (e == 31U) return sign | 0x7f800000U | (m << 13U);
    if (e == 0U) {
        if (m == 0U) return sign;
        int k = -14;
        while ((m & 1024U) == 0U) { m <<= 1U; --k; }
        return sign | (std::uint32_t(k + 127) << 23U) | ((m & 1023U) << 13U);
    }
    return sign | ((e + 112U) << 23U) | (m << 13U);
}

template <Format F, bool Streaming = false, HalfScale Scale = HalfScale::hardware> struct Chunk;
template <bool Streaming, HalfScale Scale> struct Chunk<Format::bf16, Streaming, Scale> {
    uint4 a, b;
    STRIX_WEIGHT_CHUNK_DEVICE Chunk() = default;
    STRIX_WEIGHT_CHUNK_DEVICE Chunk(const std::uint8_t* row, unsigned i)
        : a(load_chunk<Streaming>(row + 2ULL * i)), b(load_chunk<Streaming>(row + 2ULL * i + 16)) {}
    STRIX_WEIGHT_CHUNK_DEVICE float at(unsigned e) const {
        const auto w = word_of(e < 8U ? a : b, (e % 8U) / 2U);
        return bf16_to_float(std::uint16_t(w >> ((e % 2U) * 16U)));
    }
};
template <bool Streaming, HalfScale Scale> struct Chunk<Format::q8_0, Streaming, Scale> {
    uint4 codes;
    float d;
    STRIX_WEIGHT_CHUNK_DEVICE Chunk() = default;
    // A 16-chunk never crosses a 32-block; the 34-byte block stride keeps the payload 2-byte aligned only.
    STRIX_WEIGHT_CHUNK_DEVICE Chunk(const std::uint8_t* row, unsigned i) {
        const std::uint8_t* p = row + std::size_t(i / 32U) * 34U;
        std::uint16_t h;
#if defined(__HIPCC__) || defined(__HIP__)
        __builtin_memcpy(&h, p, 2);
        if constexpr (Scale == HalfScale::hardware)
            d = float(__builtin_bit_cast(_Float16, h));
        else
            d = __builtin_bit_cast(float, software_half_bits(h));
        if constexpr (!Streaming)
            __builtin_memcpy(&codes, p + 2 + i % 32U, sizeof(codes));
        else
            codes = load_chunk<Streaming>(p + 2 + i % 32U);
#else
        // Host arithmetic fixture. Both modes widen every finite scale exactly.
        std::memcpy(&h, p, sizeof(h));
        d = std::bit_cast<float>(software_half_bits(h));
        codes = load_chunk<Streaming>(p + 2 + i % 32U);
#endif
    }
    STRIX_WEIGHT_CHUNK_DEVICE float at(unsigned e) const {
        const unsigned q = (word_of(codes, e / 4U) >> ((e % 4U) * 8U)) & 255U;
        return d * float(q < 128U ? int(q) : int(q) - 256);
    }
};
template <bool Streaming, HalfScale Scale> struct Chunk<Format::e4m3_b128, Streaming, Scale> {
    uint4 codes;
    float s;
    STRIX_WEIGHT_CHUNK_DEVICE Chunk() = default;
    // A 16-chunk never crosses a 128-block; the 132-byte block stride keeps the payload 4-byte aligned only.
    STRIX_WEIGHT_CHUNK_DEVICE Chunk(const std::uint8_t* row, unsigned i) {
        const std::uint8_t* p = row + std::size_t(i / 128U) * 132U;
#if defined(__HIPCC__) || defined(__HIP__)
        __builtin_memcpy(&s, p, sizeof(s));
#else
        std::memcpy(&s, p, sizeof(s));
#endif
        codes = load_chunk<Streaming>(p + 4 + i % 128U);
    }
    STRIX_WEIGHT_CHUNK_DEVICE float at(unsigned e) const {
        const auto q = static_cast<std::uint8_t>(word_of(codes, e / 4U) >> ((e % 4U) * 8U));
        return weight::scalar_float::e4m3(q) * s;
    }
};
STRIX_WEIGHT_CHUNK_INLINE float half_scale(std::uint16_t h, HalfScale scale) {
#if defined(__HIPCC__) || defined(__HIP__)
    if (scale == HalfScale::hardware) return float(__builtin_bit_cast(_Float16, h));
    return __builtin_bit_cast(float, software_half_bits(h));
#else
    (void)scale;
    return std::bit_cast<float>(software_half_bits(h));
#endif
}
// The 4-byte word at a 4-byte aligned superblock offset (low-bit planes).
STRIX_WEIGHT_CHUNK_INLINE unsigned load_word(const std::uint8_t* p) {
#if defined(__HIPCC__) || defined(__HIP__)
    return *static_cast<const unsigned*>(__builtin_assume_aligned(p, 4));
#else
    unsigned v;
    std::memcpy(&v, p, sizeof(v));
    return v;
#endif
}
template <class Storage, HalfScale Scale>
struct LowBitChunk {
    unsigned lo0, lo1, hi;
    float d;
    STRIX_WEIGHT_CHUNK_DEVICE LowBitChunk() = default;
    STRIX_WEIGHT_CHUNK_DEVICE LowBitChunk(const std::uint8_t* row, unsigned i) {
        const std::uint8_t* block = row + std::size_t(i / Storage::elements) * Storage::bytes;
        const unsigned j = i % Storage::elements;
        lo0 = load_word(block + j / 2U);
        lo1 = load_word(block + j / 2U + 4U);
        if constexpr (Storage::high_bytes != 0U) hi = load_word(block + Storage::low_bytes + j / 4U);
        else hi = 0U;
        d = half_scale(Storage::scale_bits(block, j / Storage::group), Scale);
    }
    STRIX_WEIGHT_CHUNK_DEVICE float at(unsigned e) const {
        const unsigned low = ((e < 8U ? lo0 : lo1) >> ((e % 8U) * 4U)) & 15U;
        int q;
        if constexpr (Storage::high_bytes != 0U)
            q = static_cast<int>((low | (((hi >> (e * 2U)) & 3U) << 4U)) << 26U) >> 26;
        else
            q = static_cast<int>(low << 28U) >> 28;
        return d * float(q);
    }
};
template <bool Streaming, HalfScale Scale>
struct Chunk<Format::q4_g64, Streaming, Scale> : LowBitChunk<weight::Q4G64Storage, Scale> {
    using LowBitChunk<weight::Q4G64Storage, Scale>::LowBitChunk;
};
template <bool Streaming, HalfScale Scale>
struct Chunk<Format::q6_g64, Streaming, Scale> : LowBitChunk<weight::Q6G64Storage, Scale> {
    using LowBitChunk<weight::Q6G64Storage, Scale>::LowBitChunk;
};
inline constexpr std::size_t row_bytes(Format f, std::size_t inputs) {
    return f == Format::bf16        ? inputs * 2U
           : f == Format::q8_0      ? (inputs / 32U) * 34U
           : f == Format::e4m3_b128 ? (inputs / 128U) * 132U
           : f == Format::q4_g64    ? std::size_t(weight::Q4G64Storage::row_bytes(inputs))
                                    : std::size_t(weight::Q6G64Storage::row_bytes(inputs));
}
// Elements per storage block: the K granularity a matrix of this format needs.
inline constexpr unsigned block_elements(Format f) { return f == Format::bf16 ? 16U : f == Format::q8_0 ? 32U : 128U; }

} // namespace strix::core::weight_decode

#undef STRIX_WEIGHT_CHUNK_INLINE
#undef STRIX_WEIGHT_CHUNK_DEVICE
