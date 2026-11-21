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

template <unsigned First>
STRIX_NVFP4_TABLE_HD Bytes build_four(const weight::microscaled::PackedOperand<weight::microscaled::MatrixFormat::nvfp4>& operand,
                                      float scale) {
    constexpr auto first = static_cast<std::uint8_t>(First | ((First + 1U) << 4U));
    constexpr auto second = static_cast<std::uint8_t>((First + 2U) | ((First + 3U) << 4U));
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
        using nvfp4_table_detail::build_four;
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

}  // namespace strix::core

#undef STRIX_NVFP4_TABLE_HD
