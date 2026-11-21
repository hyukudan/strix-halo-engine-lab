#pragma once

#include "core/nvfp4_bf16_table.hpp"

#include <array>
#include <cstdio>

namespace mini_nvfp4_table_cases {

inline constexpr auto outer_bits = [] {
    constexpr std::array special{
        0x3f800000U, 0xbf800000U, 0x3f000000U, 0x3b727bb3U, 0x7f61b1e6U, 0x477fe000U,
        0x00000000U, 0x80000000U, 0x00000001U, 0x00400000U, 0x007fffffU,
        0x7fc00000U, 0xffc58000U, 0x7f800000U,
    };
    // With scale 1, both products hit each listed exponent. The mantissas
    // exercise BF16 ties and the transition to the next BF16 exponent,
    // including the neighbours of B's transition with scale 1.
    constexpr std::array exponents{1U, 2U, 3U, 251U, 252U, 253U, 254U};
    constexpr std::array mantissas{0U, 0x008000U, 0x2a5554U, 0x2a5555U, 0x2a5556U,
                                   0x7f7fffU, 0x7f8000U};
    std::array<std::uint32_t, special.size() + 2U * exponents.size() * mantissas.size()> cases{};
    std::size_t index = 0;
    for (const auto bits : special) cases[index++] = bits;
    for (const auto exponent : exponents) {
        for (const auto mantissa : mantissas) {
            const auto bits = (exponent << 23U) | mantissa;
            cases[index++] = bits;
            cases[index++] = bits | 0x80000000U;
        }
    }
    return cases;
}();
inline constexpr unsigned scale_count = 256U;
inline constexpr unsigned code_count = 16U;

struct Mismatch {
    unsigned quad;
    std::uint32_t actual, expected;
};

#if defined(__HIPCC__)
#define STRIX_NVFP4_TEST_HD __host__ __device__
#else
#define STRIX_NVFP4_TEST_HD
#endif

template <class Table>
STRIX_NVFP4_TEST_HD Mismatch check_quads(const Table& table, const strix::core::Nvfp4Bf16Table::Operand& operand,
                                       float scale) {
    using strix::weight::scalar_float::f2u;
    for (unsigned code = 0; code < code_count; ++code) {
        // Exercise every code in each selector position, with mixed signs and
        // magnitudes so a reversed permutation cannot pass on repeated bytes.
        const auto low = static_cast<std::uint8_t>(code | (((code + 5U) & 15U) << 4U));
        const auto high = static_cast<std::uint8_t>(((code + 9U) & 15U) | (((code + 12U) & 15U) << 4U));
        const auto quad = table.decode_quad(std::uint32_t{low} | (std::uint32_t{high} << 8U));
        const auto actual_first = (f2u(quad.template widen<0U>()) >> 16U) | f2u(quad.template widen<1U>());
        const auto actual_second = (f2u(quad.template widen<2U>()) >> 16U) | f2u(quad.template widen<3U>());
        // Production decodes codes loaded from memory; keep the unrolled
        // constant codes opaque so the reference sees the same arithmetic.
        const auto low_code = strix::core::nvfp4_table_detail::opaque_code(low);
        const auto high_code = strix::core::nvfp4_table_detail::opaque_code(high);
        const auto expected_first = operand.decode_pair(&low_code, scale);
        const auto expected_second = operand.decode_pair(&high_code, scale);
        if (actual_first != expected_first) return {2U * code, actual_first, expected_first};
        if (actual_second != expected_second) return {2U * code + 1U, actual_second, expected_second};
    }
    return {2U * code_count, 0U, 0U};
}

STRIX_NVFP4_TEST_HD inline Mismatch check_block(float outer, std::uint8_t scale_code) {
    using strix::core::Nvfp4Bf16SignedTable;
    using strix::core::Nvfp4Bf16Table;
    const Nvfp4Bf16Table::Operand operand{{nullptr, nullptr, 0, 0, outer}};
    const auto scale = operand.decode_scale(scale_code);
    if (Nvfp4Bf16Table::sign_symmetric(scale, outer)) {
        return check_quads(Nvfp4Bf16Table::build(operand, scale), operand, scale);
    }
    return check_quads(Nvfp4Bf16SignedTable::build(operand, scale), operand, scale);
}

#undef STRIX_NVFP4_TEST_HD

inline bool report(Mismatch mismatch, std::uint32_t outer, unsigned scale) {
    if (mismatch.quad == 2U * code_count) return true;
    std::fprintf(stderr, "NVFP4 outer=%08x scale=%02x quad=%u pair=%u: got %08x, expected %08x\n",
                 outer, scale, mismatch.quad / 2U, mismatch.quad % 2U, mismatch.actual, mismatch.expected);
    return false;
}

}  // namespace mini_nvfp4_table_cases
