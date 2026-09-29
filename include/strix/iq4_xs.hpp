#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::iq4_xs {

inline constexpr std::size_t block_elements = 256;
inline constexpr std::size_t block_bytes = 136;
inline constexpr std::size_t subblock_elements = 32;
inline constexpr std::size_t subblocks_per_block =
    block_elements / subblock_elements;

inline constexpr std::array<std::int8_t, 16> codebook{
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

[[nodiscard]] float fp16_bits_to_float(std::uint16_t bits) noexcept;

[[nodiscard]] std::uint16_t global_scale_bits(
    std::span<const std::byte, block_bytes> block) noexcept;

[[nodiscard]] std::uint8_t local_scale_code(
    std::span<const std::byte, block_bytes> block,
    std::size_t subblock);

[[nodiscard]] float dequantized_value(
    std::span<const std::byte, block_bytes> block,
    std::size_t element);

void dequantize_block(
    std::span<const std::byte, block_bytes> block,
    std::span<float, block_elements> output);

}  // namespace strix::iq4_xs
