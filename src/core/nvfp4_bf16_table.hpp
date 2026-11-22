#pragma once

#include "strix/weight/formats/microscaled.hpp"

#if defined(__HIPCC__)
#define STRIX_NVFP4_TABLE_HD __host__ __device__ __forceinline__
#else
#define STRIX_NVFP4_TABLE_HD inline
#endif

namespace strix::core {

namespace nvfp4_table_detail {

// Selectors 0..3 address the second operand, 4..7 the first, and 12 gives zero.
STRIX_NVFP4_TABLE_HD std::uint32_t perm(std::uint32_t high, std::uint32_t low, std::uint32_t selectors) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_perm(high, low, selectors);
#else
    const auto source = (std::uint64_t{high} << 32U) | low;
    std::uint32_t result = 0;
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        const auto selector = (selectors >> shift) & 0xffU;
        if (selector < 8U) {
            result |= static_cast<std::uint32_t>((source >> (8U * selector)) & 0xffU) << shift;
        }
    }
    return result;
#endif
}

struct Bytes {
    std::uint32_t low, high;
};

// Hide the table's code bytes from the optimiser: with a constant negative
// E2M1 value the compiler may rewrite (-c) * s as -(c * s), which flips the
// sign of a NaN product and breaks bitwise agreement with decode_pair.
STRIX_NVFP4_TABLE_HD std::uint8_t opaque_code(std::uint32_t code) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(code));
#elif defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+r"(code));
#else
    volatile std::uint32_t barrier = code;
    code = barrier;
#endif
    return static_cast<std::uint8_t>(code);
}

template <unsigned First>
STRIX_NVFP4_TABLE_HD Bytes build_four(const weight::microscaled::PackedOperand<weight::microscaled::MatrixFormat::nvfp4>& operand,
                                      float scale) {
    const auto first = opaque_code(First | ((First + 1U) << 4U));
    const auto second = opaque_code((First + 2U) | ((First + 3U) << 4U));
    const auto pair01 = operand.decode_pair(&first, scale);
    const auto pair23 = operand.decode_pair(&second, scale);
    return {perm(pair23, pair01, 0x06040200U), perm(pair23, pair01, 0x07050301U)};
}

struct Magnitudes {
    Bytes first, second;

    STRIX_NVFP4_TABLE_HD Bytes lookup(std::uint32_t selectors) const {
        return {perm(second.low, first.low, selectors), perm(second.high, first.high, selectors)};
    }
};

STRIX_NVFP4_TABLE_HD std::uint32_t expand_codes(std::uint32_t packed_codes) {
    return perm(packed_codes >> 4U, packed_codes, 0x05010400U);
}

// The input is known finite. Keep the RNE BF16 bits in bytes 2 and 3.
STRIX_NVFP4_TABLE_HD std::uint32_t round_finite(std::uint32_t bits) {
    return bits + 0x7fffU + ((bits >> 16U) & 1U);
}

}  // namespace nvfp4_table_detail

struct Nvfp4Bf16Quad {
    std::uint32_t low, high;  // Four low bytes and four high bytes in input order.

    template <unsigned I>
    STRIX_NVFP4_TABLE_HD float widen() const {
        static_assert(I < 4U);
        constexpr auto select = ((I + 4U) << 24U) | (I << 16U) | 0x0c0cU;
        return weight::scalar_float::u2f(nvfp4_table_detail::perm(high, low, select));
    }
};

// Finite E4M3 scales make the first product exact and finite. The second
// product and BF16 RNE are sign symmetric, even at underflow and overflow.
struct Nvfp4Bf16Table {
    using Operand = weight::microscaled::PackedOperand<weight::microscaled::MatrixFormat::nvfp4>;
    nvfp4_table_detail::Magnitudes magnitudes;

    STRIX_NVFP4_TABLE_HD static bool sign_symmetric(float scale, float outer) {
        using weight::scalar_float::f2u;
        constexpr std::uint32_t exponent = 0x7f800000U;
        return (f2u(scale) & exponent) != exponent && (f2u(outer) & exponent) != exponent;
    }

    // Precondition: sign_symmetric for the decoded block scale and outer scale.
    STRIX_NVFP4_TABLE_HD static Nvfp4Bf16Table build(const Operand& operand, float scale) {
        using namespace nvfp4_table_detail;
        using weight::scalar_float::f2u;
        const float outer = operand.matrix.outer_scale;
        // Multiplying the decoded E4M3 scale by 1 is exact. Form the other
        // exact first product before applying the outer scale as in decode_pair.
        const auto a = f2u(scale * outer);
        const auto b = f2u((1.5F * scale) * outer);
        // |a| <= |b|. A's FP32 exponent must be at least 2 for halving.
        // B must round below BF16 exponent 253 for quadrupling. The upper
        // bound is the RNE midpoint that rounds up to exponent 253.
        if ((a & 0x7fffffffU) >= 0x01000000U && (b & 0x7fffffffU) < 0x7e7f8000U) {
            // Positive power-of-two shifts commute with both roundings in
            // this range. Packed additions stay inside each BF16 halfword.
            const auto bases = perm(round_finite(b), round_finite(a), 0x07060302U);
            const auto twice = bases + 0x00800080U;
            const auto quadruple = bases + 0x01000100U;
            const auto half_and_zero = ((bases - 0x80U) << 16U) | (bases & 0x8000U);
            // [signed zero, A/2, A, B], then [2A, 2B, 4A, 4B].
            return {{{perm(half_and_zero, bases, 0x0200060cU),
                      perm(half_and_zero, bases, 0x03010705U)},
                     {perm(quadruple, twice, 0x06040200U),
                      perm(quadruple, twice, 0x07050301U)}}};
        }
        return {{build_four<0U>(operand, scale), build_four<4U>(operand, scale)}};
    }

    STRIX_NVFP4_TABLE_HD Nvfp4Bf16Quad decode_quad(std::uint32_t packed_codes) const {
        using namespace nvfp4_table_detail;
        const auto codes = expand_codes(packed_codes);
        const auto bytes = magnitudes.lookup(codes & 0x07070707U);
        const auto high = bytes.high ^ ((codes & 0x08080808U) << 4U);
        return {bytes.low, high};
    }
};

// Nonfinite products can break sign symmetry. Obtain each sign and payload
// from the reference arithmetic and use the sign only to select bytes.
struct Nvfp4Bf16SignedTable {
    nvfp4_table_detail::Magnitudes positive, negative;

    STRIX_NVFP4_TABLE_HD static Nvfp4Bf16SignedTable build(const Nvfp4Bf16Table::Operand& operand, float scale) {
        using nvfp4_table_detail::build_four;
        return {{build_four<0U>(operand, scale), build_four<4U>(operand, scale)},
                {build_four<8U>(operand, scale), build_four<12U>(operand, scale)}};
    }

    STRIX_NVFP4_TABLE_HD Nvfp4Bf16Quad decode_quad(std::uint32_t packed_codes) const {
        using namespace nvfp4_table_detail;
        const auto codes = expand_codes(packed_codes);
        const auto selectors = codes & 0x07070707U;
        const auto pos = positive.lookup(selectors);
        const auto neg = negative.lookup(selectors);
        const auto sign = ((codes & 0x08080808U) >> 3U) * 0xffU;
        const auto low = (pos.low & ~sign) | (neg.low & sign);
        const auto high = (pos.high & ~sign) | (neg.high & sign);
        return {low, high};
    }
};

// PackedOperand<Format>::decode_bf16(tile, out): the eight packed BF16 words of one 16-value tile (word p holds values
// 2p and 2p + 1, low half first). NVFP4 tiles with a finite block and outer scale go through the block table (one build,
// four quad lookups, two byte permutes per quad) instead of sixteen code x scale products and RNE encodes; the table
// reproduces decode_pair bit for bit. Other tiles and formats use the operand's own decoder.
template <class Operand>
STRIX_NVFP4_TABLE_HD void decode_tile_bf16(const Operand& operand, const typename Operand::Tile16& tile,
                                           std::uint32_t (&out)[8]) {
    if constexpr (Operand::kNvfp4) {
        const float scale = Operand::decode_scale(tile.scale_code);
        if (Nvfp4Bf16Table::sign_symmetric(scale, operand.matrix.outer_scale)) {
            const auto table = Nvfp4Bf16Table::build(operand, scale);
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
            for (unsigned quad = 0; quad < 4U; ++quad) {
                const auto values = table.decode_quad(std::uint32_t{tile.codes[2U * quad]} |
                                                      (std::uint32_t{tile.codes[2U * quad + 1U]} << 8U));
                // [low0, high0, low1, high1] and [low2, high2, low3, high3].
                out[2U * quad] = nvfp4_table_detail::perm(values.high, values.low, 0x05010400U);
                out[2U * quad + 1U] = nvfp4_table_detail::perm(values.high, values.low, 0x07030602U);
            }
            return;
        }
    }
    operand.decode_bf16(tile, out);
}

}  // namespace strix::core

#undef STRIX_NVFP4_TABLE_HD
