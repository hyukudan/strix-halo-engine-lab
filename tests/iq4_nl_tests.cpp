#include "strix/iq4_nl.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(
    bool condition,
    const char* message) {
    if (!condition) {
        std::cerr
            << "FAIL: "
            << message
            << '\n';
        ++failures;
    }
}

void set_u16_le(
    std::array<
        std::byte,
        strix::iq4_nl::block_bytes>& block,
    std::uint16_t value) {
    block[0] =
        static_cast<std::byte>(
            value & 0xffU);
    block[1] =
        static_cast<std::byte>(
            (value >> 8U) &
            0xffU);
}

}  // namespace

int main() {
    using namespace strix::iq4_nl;

    expect(
        block_elements == 32,
        "IQ4_NL block has 32 logical values");
    expect(
        block_bytes == 18,
        "IQ4_NL block occupies 18 bytes");
    expect(
        codebook.front() == -127 &&
            codebook.back() == 113,
        "IQ4_NL nonlinear grid endpoints are pinned");

    expect(
        fp16_bits_to_float(
            0x3c00U) == 1.0F,
        "FP16 one converts exactly");
    expect(
        fp16_bits_to_float(
            0xc000U) == -2.0F,
        "FP16 negative two converts exactly");

    std::array<
        std::byte,
        block_bytes> block{};
    set_u16_le(
        block,
        0x3800U);  // 0.5

    for (std::size_t index = 0;
         index < 16U;
         ++index) {
        const auto low =
            static_cast<std::uint8_t>(
                index);
        const auto high =
            static_cast<std::uint8_t>(
                15U - index);
        block[2U + index] =
            static_cast<std::byte>(
                low |
                static_cast<std::uint8_t>(
                    high << 4U));
    }

    std::array<
        float,
        block_elements> values{};
    dequantize_block(
        block,
        values);

    for (std::size_t index = 0;
         index < 16U;
         ++index) {
        expect(
            quant_code(
                block,
                index) ==
                index,
            "low nibble code order is pinned");
        expect(
            quant_code(
                block,
                16U + index) ==
                15U - index,
            "high nibble code order is pinned");
        expect(
            values[index] ==
                0.5F *
                static_cast<float>(
                    codebook[index]),
            "low nibble dequant uses nonlinear grid");
        expect(
            values[16U + index] ==
                0.5F *
                static_cast<float>(
                    codebook[15U - index]),
            "high nibble dequant uses nonlinear grid");
    }

    bool rejected = false;
    try {
        static_cast<void>(
            dequantized_value(
                block,
                block_elements));
    } catch (
        const std::out_of_range&) {
        rejected = true;
    }
    expect(
        rejected,
        "IQ4_NL rejects out-of-range element");

    if (failures == 0) {
        std::cout
            << "IQ4_NL tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
