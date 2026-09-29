#include "strix/iq4_xs.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::iq4_xs {
namespace {

constexpr std::size_t global_scale_offset = 0;
constexpr std::size_t scales_high_offset = 2;
constexpr std::size_t scales_low_offset = 4;
constexpr std::size_t quant_offset = 8;

std::uint8_t byte_at(std::span<const std::byte, block_bytes> block,
                     std::size_t offset) noexcept {
    return std::to_integer<std::uint8_t>(block[offset]);
}

}  // namespace

float fp16_bits_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    const std::uint32_t mantissa = bits & 0x03ffU;

    std::uint32_t fp32 = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            fp32 = sign;
        } else {
            std::uint32_t normalized = mantissa;
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
            exponent + (127U - 15U);
        fp32 =
            sign |
            (fp32_exponent << 23U) |
            (mantissa << 13U);
    }
    return std::bit_cast<float>(fp32);
}

std::uint16_t global_scale_bits(
    std::span<const std::byte, block_bytes> block) noexcept {
    return static_cast<std::uint16_t>(
        byte_at(block, global_scale_offset) |
        (static_cast<std::uint16_t>(
             byte_at(block, global_scale_offset + 1U))
         << 8U));
}

std::uint8_t local_scale_code(
    std::span<const std::byte, block_bytes> block,
    std::size_t subblock) {
    if (subblock >= subblocks_per_block) {
        throw std::out_of_range("IQ4_XS subblock index out of range");
    }

    const std::uint8_t low_packed =
        byte_at(block, scales_low_offset + subblock / 2U);
    const unsigned low_shift =
        static_cast<unsigned>(4U * (subblock % 2U));
    const std::uint8_t low =
        static_cast<std::uint8_t>((low_packed >> low_shift) & 0x0fU);

    const std::uint16_t high_packed =
        static_cast<std::uint16_t>(
            byte_at(block, scales_high_offset) |
            (static_cast<std::uint16_t>(
                 byte_at(block, scales_high_offset + 1U))
             << 8U));
    const unsigned high_shift =
        static_cast<unsigned>(2U * subblock);
    const std::uint8_t high =
        static_cast<std::uint8_t>(
            (high_packed >> high_shift) & 0x03U);

    return static_cast<std::uint8_t>(low | (high << 4U));
}

float dequantized_value(
    std::span<const std::byte, block_bytes> block,
    std::size_t element) {
    if (element >= block_elements) {
        throw std::out_of_range("IQ4_XS element index out of range");
    }

    const std::size_t subblock =
        element / subblock_elements;
    const std::size_t within =
        element % subblock_elements;
    const std::size_t quant_byte =
        quant_offset +
        subblock * (subblock_elements / 2U) +
        (within % 16U);

    const std::uint8_t packed = byte_at(block, quant_byte);
    const std::uint8_t code =
        within < 16U
            ? static_cast<std::uint8_t>(packed & 0x0fU)
            : static_cast<std::uint8_t>(packed >> 4U);

    const float global =
        fp16_bits_to_float(global_scale_bits(block));
    const int local =
        static_cast<int>(local_scale_code(block, subblock)) - 32;
    return global *
           static_cast<float>(local) *
           static_cast<float>(codebook[code]);
}

void dequantize_block(
    std::span<const std::byte, block_bytes> block,
    std::span<float, block_elements> output) {
    for (std::size_t index = 0;
         index < block_elements;
         ++index) {
        output[index] = dequantized_value(block, index);
    }
}

}  // namespace strix::iq4_xs
