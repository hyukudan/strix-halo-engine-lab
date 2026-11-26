#pragma once
// Candidate row layouts of the block-layout bench (tests/hip_strixpack_bench.hip). Bench instruments: not registered
// weight formats. Host and device read the same bytes; the device decoders are in core/strixpack_operand.hpp.
//
// A row is addressed by its ANCHOR. Code superblocks (128 elements) ascend from the anchor and the scale words descend
// below it, so a decoder needs only (anchor, k): no K and no second pointer.
//   codes of superblock j:       anchor + j * code_bytes      (16-byte aligned; 64-byte aligned when Pad is 64)
//   scale word of superblock j:  anchor - 4 (j + 1)           (FP16 x 2 for 64-groups, or 4 bytes for 32-groups)
// Kinds (value of element = scale of its group * integer level):
//   q8   128 B  signed bytes, element e in byte e                                        level = byte
//   p4   64 B   4-bit u = level + 8; element 8 d + i in dword d: byte i % 4, low nibble for i < 4, high for i >= 4
//   p4b  as p4, one scale byte per 32 elements (value (1 + m / 8) 2^(e - 31), e = byte >> 3, m = byte & 7)
//   p2   32 B   2-bit u, level = 2 u - 3; element 16 d + 4 a + b in dword d at field 4 b + {0, 2, 1, 3}[a] (2 bits each)
//   p2t  as p2, level = {-4, -1, +1, +4}[u]
//   p3   48 B   3-bit u = lo + 4 hi, level = 2 u - 7; lo plane = bytes [0, 32) laid out as p2, hi plane = bytes [32, 48),
//               element e at bit e
// The element orders are the ones the device splits produce without a permute (mask / shift only).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strix::core::strixpack {

enum class Kind : std::uint8_t { q8, p4, p4b, p3, p2, p2t };

inline constexpr unsigned kSuper = 128U;
constexpr unsigned code_bytes(Kind f) {
    return f == Kind::q8 ? 128U : f == Kind::p4 || f == Kind::p4b ? 64U : f == Kind::p3 ? 48U : 32U;
}
constexpr unsigned group_of(Kind f) { return f == Kind::p4b ? 32U : 64U; }
constexpr const char* name_of(Kind f) {
    return f == Kind::q8 ? "sp_q8" : f == Kind::p4 ? "sp_p4" : f == Kind::p4b ? "sp_p4b" : f == Kind::p3 ? "sp_p3"
           : f == Kind::p2 ? "sp_p2" : "sp_p2t";
}
// Bytes a decoder reads per row (codes + scale words), and the padded pitch pieces.
constexpr std::size_t read_bytes(Kind f, std::size_t K) { return K / kSuper * (code_bytes(f) + 4U); }
constexpr std::size_t pad_to(std::size_t n, std::size_t a) { return (n + a - 1U) / a * a; }
constexpr std::size_t front_bytes(std::size_t K, std::size_t pad) { return pad_to(K / kSuper * 4U, pad); }
constexpr std::size_t pitch(Kind f, std::size_t K, std::size_t pad) {
    return front_bytes(K, pad) + pad_to(K / kSuper * code_bytes(f), pad);
}

inline constexpr unsigned kField2[4] = {0U, 2U, 1U, 3U};
// Bit offset of the 2-bit field of element i (0..15) inside its plane dword.
constexpr unsigned field2_shift(unsigned i) { return 2U * (4U * (i % 4U) + kField2[i / 4U]); }

// Host: raw code u of element k of the row at `anchor`.
inline unsigned raw_code(Kind f, const std::uint8_t* anchor, std::size_t k) {
    const std::uint8_t* block = anchor + k / kSuper * code_bytes(f);
    const auto e = static_cast<unsigned>(k % kSuper);
    const auto dword = [&](unsigned d) {
        std::uint32_t w;
        std::memcpy(&w, block + 4U * d, 4U);
        return w;
    };
    if (f == Kind::q8) return block[e];
    if (f == Kind::p4 || f == Kind::p4b) {
        const unsigned i = e % 8U;
        return (dword(e / 8U) >> (8U * (i % 4U) + 4U * (i / 4U))) & 15U;
    }
    const unsigned lo = (dword(e / 16U) >> field2_shift(e % 16U)) & 3U;
    if (f != Kind::p3) return lo;
    return lo | (((block[32U + e / 8U] >> (e % 8U)) & 1U) << 2U);
}
inline void set_raw_code(Kind f, std::uint8_t* anchor, std::size_t k, unsigned u) {
    std::uint8_t* block = anchor + k / kSuper * code_bytes(f);
    const auto e = static_cast<unsigned>(k % kSuper);
    const auto put = [&](unsigned d, unsigned shift, unsigned mask, unsigned v) {
        std::uint32_t w;
        std::memcpy(&w, block + 4U * d, 4U);
        w = (w & ~(mask << shift)) | ((v & mask) << shift);
        std::memcpy(block + 4U * d, &w, 4U);
    };
    if (f == Kind::q8) {
        block[e] = static_cast<std::uint8_t>(u);
    } else if (f == Kind::p4 || f == Kind::p4b) {
        const unsigned i = e % 8U;
        put(e / 8U, 8U * (i % 4U) + 4U * (i / 4U), 15U, u);
    } else {
        put(e / 16U, field2_shift(e % 16U), 3U, u);
        if (f == Kind::p3) {
            auto& byte = block[32U + e / 8U];
            byte = static_cast<std::uint8_t>((byte & ~(1U << (e % 8U))) | (((u >> 2U) & 1U) << (e % 8U)));
        }
    }
}
// Integer level of a raw code.
constexpr int level_of(Kind f, unsigned u) {
    return f == Kind::q8    ? (u < 128U ? static_cast<int>(u) : static_cast<int>(u) - 256)
           : f == Kind::p4 || f == Kind::p4b ? static_cast<int>(u) - 8
           : f == Kind::p3  ? 2 * static_cast<int>(u) - 7
           : f == Kind::p2  ? 2 * static_cast<int>(u) - 3
                            : (u == 0U ? -4 : u == 1U ? -1 : u == 2U ? 1 : 4);
}
constexpr unsigned raw_codes(Kind f) {
    return f == Kind::q8 ? 256U : f == Kind::p4 || f == Kind::p4b ? 16U : f == Kind::p3 ? 8U : 4U;
}

// Scale bits of the group of element k: FP16 bits (64-groups) or the scale byte (32-groups).
inline unsigned scale_bits(Kind f, const std::uint8_t* anchor, std::size_t k) {
    std::uint32_t w;
    std::memcpy(&w, anchor - 4U * (k / kSuper + 1U), 4U);
    const auto e = static_cast<unsigned>(k % kSuper);
    return group_of(f) == 64U ? (w >> (16U * (e / 64U))) & 0xFFFFU : (w >> (8U * (e / 32U))) & 0xFFU;
}
inline void set_scale_bits(Kind f, std::uint8_t* anchor, std::size_t k, unsigned bits) {
    std::uint32_t w;
    std::uint8_t* p = anchor - 4U * (k / kSuper + 1U);
    std::memcpy(&w, p, 4U);
    const auto e = static_cast<unsigned>(k % kSuper);
    const unsigned shift = group_of(f) == 64U ? 16U * (e / 64U) : 8U * (e / 32U);
    const unsigned mask = group_of(f) == 64U ? 0xFFFFU : 0xFFU;
    w = (w & ~(mask << shift)) | ((bits & mask) << shift);
    std::memcpy(p, &w, 4U);
}
// FP32 bits of a scale byte: (1 + m / 8) 2^(e - 31).
constexpr std::uint32_t byte_scale_bits(unsigned byte) { return (((byte >> 3U) + 96U) << 23U) | ((byte & 7U) << 20U); }

}  // namespace strix::core::strixpack
