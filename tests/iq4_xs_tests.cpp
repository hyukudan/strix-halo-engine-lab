#include "strix/iq4_xs.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void set_u16_le(std::array<std::byte, strix::iq4_xs::block_bytes>& block,
                std::size_t offset,
                std::uint16_t value) {
    block[offset] =
        static_cast<std::byte>(value & 0xffU);
    block[offset + 1U] =
        static_cast<std::byte>((value >> 8U) & 0xffU);
}

void set_local_scales(
    std::array<std::byte, strix::iq4_xs::block_bytes>& block,
    const std::array<std::uint8_t,
                     strix::iq4_xs::subblocks_per_block>& scales) {
    std::uint16_t high = 0;
    for (std::size_t subblock = 0;
         subblock < scales.size();
         ++subblock) {
        const auto value = scales[subblock];
        const auto low =
            static_cast<std::uint8_t>(value & 0x0fU);
        const auto high_part =
            static_cast<std::uint16_t>((value >> 4U) & 0x03U);

        const auto low_offset = 4U + subblock / 2U;
        auto packed =
            std::to_integer<std::uint8_t>(block[low_offset]);
        if ((subblock % 2U) == 0) {
            packed =
                static_cast<std::uint8_t>(
                    (packed & 0xf0U) | low);
        } else {
            packed =
                static_cast<std::uint8_t>(
                    (packed & 0x0fU) | (low << 4U));
        }
        block[low_offset] = static_cast<std::byte>(packed);
        high |= static_cast<std::uint16_t>(
            high_part << (2U * subblock));
    }
    set_u16_le(block, 2U, high);
}

}  // namespace

int main() {
    using namespace strix::iq4_xs;

    expect(block_elements == 256,
           "IQ4_XS block has 256 logical elements");
    expect(block_bytes == 136,
           "IQ4_XS block occupies 136 bytes");
    expect(codebook.front() == -127 &&
               codebook.back() == 113,
           "IQ4_XS nonlinear codebook endpoints match contract");

    expect(fp16_bits_to_float(0x3c00U) == 1.0F,
           "FP16 one converts exactly");
    expect(fp16_bits_to_float(0xc000U) == -2.0F,
           "FP16 negative two converts exactly");
    expect(std::isinf(fp16_bits_to_float(0x7c00U)),
           "FP16 infinity converts");
    expect(std::isnan(fp16_bits_to_float(0x7e00U)),
           "FP16 NaN converts");

    std::array<std::byte, block_bytes> block{};
    set_u16_le(block, 0U, 0x3c00U);

    const std::array<std::uint8_t, subblocks_per_block> scales{
        33, 31, 32, 34, 30, 63, 1, 48,
    };
    set_local_scales(block, scales);

    for (std::size_t subblock = 0;
         subblock < subblocks_per_block;
         ++subblock) {
        expect(local_scale_code(block, subblock) == scales[subblock],
               "local 6-bit scale round-trips");
        const auto base = 8U + subblock * 16U;
        for (std::size_t byte = 0; byte < 16U; ++byte) {
            block[base + byte] = std::byte{0xf0};
        }
    }

    std::array<float, block_elements> values{};
    dequantize_block(block, values);

    for (std::size_t subblock = 0;
         subblock < subblocks_per_block;
         ++subblock) {
        const float multiplier =
            static_cast<float>(
                static_cast<int>(scales[subblock]) - 32);
        for (std::size_t index = 0; index < 16U; ++index) {
            const auto base =
                subblock * subblock_elements;
            expect(values[base + index] ==
                       multiplier * -127.0F,
                   "low nibble dequantizes through codebook");
            expect(values[base + 16U + index] ==
                       multiplier * 113.0F,
                   "high nibble dequantizes through codebook");
        }
    }

    bool subblock_rejected = false;
    try {
        static_cast<void>(
            local_scale_code(block, subblocks_per_block));
    } catch (const std::out_of_range&) {
        subblock_rejected = true;
    }
    expect(subblock_rejected,
           "out-of-range local scale is rejected");

    bool element_rejected = false;
    try {
        static_cast<void>(
            dequantized_value(block, block_elements));
    } catch (const std::out_of_range&) {
        element_rejected = true;
    }
    expect(element_rejected,
           "out-of-range dequant element is rejected");

    if (failures == 0) {
        std::cout << "IQ4_XS tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
