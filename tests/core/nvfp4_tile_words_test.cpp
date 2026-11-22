// decode_tile_bf16 (block table) == PackedOperand<nvfp4>::decode_bf16 (reference) word for word, for every block scale
// code, outer scales across the exponent range (signed, zero, subnormal, nonfinite) and code patterns that put every
// code in every nibble position with mixed signs.
#include "core/nvfp4_bf16_table.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    using strix::core::Nvfp4Bf16Table;
    using strix::weight::scalar_float::u2f;
    using Operand = Nvfp4Bf16Table::Operand;
    constexpr std::array<std::uint32_t, 22> outer_bits{
        0x3f800000U, 0xbf800000U, 0x3f000000U, 0x3b727bb3U, 0x7f61b1e6U, 0x477fe000U, 0x00000000U, 0x80000000U,
        0x00000001U, 0x00400000U, 0x007fffffU, 0x7fc00000U, 0xffc58000U, 0x7f800000U, 0x00800000U, 0x01000000U,
        0x01800000U, 0x7e7f7fffU, 0x7e7f8000U, 0x7eaaaaabU, 0xbe2aaaabU, 0x42f60000U};
    unsigned tiles = 0;
    for (const auto bits : outer_bits) {
        const Operand operand{{nullptr, nullptr, 0, 0, u2f(bits)}};
        for (unsigned scale = 0; scale < 256U; ++scale) {
            for (unsigned pattern = 0; pattern < 16U; ++pattern) {
                Operand::Tile16 tile{};
                for (unsigned i = 0; i < Operand::kCodes; ++i) {
                    const unsigned low = (pattern + 3U * i) & 15U, high = (pattern * 7U + 5U * i + 9U) & 15U;
                    tile.codes[i] = static_cast<std::uint8_t>(low | (high << 4U));
                }
                tile.scale_code = static_cast<std::uint8_t>(scale);
                std::uint32_t actual[8]{}, expected[8]{};
                strix::core::decode_tile_bf16(operand, tile, actual);
                operand.decode_bf16(tile, expected);
                if (std::memcmp(actual, expected, sizeof(actual)) != 0) {
                    std::printf("mismatch outer 0x%08x scale %u pattern %u\n", static_cast<unsigned>(bits), scale, pattern);
                    return 1;
                }
                ++tiles;
            }
        }
    }
    std::printf("nvfp4 tile words: %u tiles equal\n", tiles);
    return 0;
}
