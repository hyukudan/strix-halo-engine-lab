#include "strix/iq4_nl.hpp"

#include <bit>
#include <stdexcept>

namespace strix::iq4_nl {
namespace {

std::uint8_t byte_at(
    std::span<const std::byte, block_bytes> block,
    std::size_t offset) noexcept {
    return std::to_integer<std::uint8_t>(
        block[offset]);
}

}  // namespace

float fp16_bits_to_float(
    std::uint16_t bits) noexcept {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(
            bits & 0x8000U) <<
        16U;
    const std::uint32_t exponent =
        (bits >> 10U) & 0x1fU;
    const std::uint32_t mantissa =
        bits & 0x03ffU;

    std::uint32_t fp32 = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            fp32 = sign;
        } else {
            std::uint32_t normalized =
                mantissa;
            std::uint32_t shift = 0;
            while ((normalized & 0x0400U) == 0) {
                normalized <<= 1U;
                ++shift;
            }
            normalized &= 0x03ffU;
            const std::uint32_t fp32_exponent =
                127U - 15U - shift + 1U;
            fp32 =
                sign |
                (fp32_exponent << 23U) |
                (normalized << 13U);
        }
    } else if (exponent == 0x1fU) {
        fp32 =
            sign |
            0x7f800000U |
            (mantissa << 13U);
        if (mantissa != 0) {
            fp32 |= 0x00400000U;
        }
    } else {
        const std::uint32_t fp32_exponent =
            exponent +
            (127U - 15U);
        fp32 =
            sign |
            (fp32_exponent << 23U) |
            (mantissa << 13U);
    }
    return std::bit_cast<float>(
        fp32);
}

std::uint16_t scale_bits(
    std::span<const std::byte, block_bytes> block) noexcept {
    return static_cast<std::uint16_t>(
        byte_at(block, 0U) |
        (static_cast<std::uint16_t>(
             byte_at(block, 1U))
         << 8U));
}

std::uint8_t quant_code(
    std::span<const std::byte, block_bytes> block,
    std::size_t element) {
    if (element >= block_elements) {
        throw std::out_of_range(
            "IQ4_NL element index out of range");
    }

    // GGML packs the first 16 values in low nibbles and the next 16 in
    // high nibbles of the same 16 bytes.
    const auto within =
        element % 16U;
    const auto packed =
        byte_at(
            block,
            2U + within);
    return element < 16U
        ? static_cast<std::uint8_t>(
              packed & 0x0fU)
        : static_cast<std::uint8_t>(
              packed >> 4U);
}

float dequantized_value(
    std::span<const std::byte, block_bytes> block,
    std::size_t element) {
    const auto code =
        quant_code(
            block,
            element);
    return
        fp16_bits_to_float(
            scale_bits(block)) *
        static_cast<float>(
            codebook[code]);
}

void dequantize_block(
    std::span<const std::byte, block_bytes> block,
    std::span<float, block_elements> output) {
    for (std::size_t element = 0;
         element < block_elements;
         ++element) {
        output[element] =
            dequantized_value(
                block,
                element);
    }
}

}  // namespace strix::iq4_nl
