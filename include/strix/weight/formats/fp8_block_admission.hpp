#pragma once

#include "strix/weight/formats/scalar_float.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace strix::weight::fp8_block_admission {

inline constexpr std::size_t block = 32;
// The integer E4M3 dot uses 2^(e - 136) as a normal FP32 scale.
inline constexpr std::uint8_t min_exponent = 10, max_exponent = 254;

[[nodiscard]] constexpr bool supports_exponent(std::uint8_t e) noexcept {
    return e >= min_exponent && e <= max_exponent;
}

// Empty on success. Codes and scales are separate, contiguous planes.
[[nodiscard]] inline std::string check(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales,
                  std::size_t rows, std::size_t width) {
    if (rows == 0U || width == 0U || rows % block != 0U || width % block != 0U)
        return "rows and width must be multiples of 32";
    if (codes.size() != rows * width || scales.size() != rows / block * (width / block))
        return "codes or exponents of the wrong size";
    for (const auto c : codes)
        if (scalar_float::e4m3_is_nan(c)) return "NaN code";
    for (const auto e : scales)
        if (!supports_exponent(e)) return "block exponent " + std::to_string(e) + " outside [10, 254]";
    return {};
}

}  // namespace strix::weight::fp8_block_admission
