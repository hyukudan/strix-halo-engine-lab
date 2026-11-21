#pragma once

#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::weight::microscaled {

// Decode one contiguous source row into FP32. The caller selects the expert's
// amax, and performs any operand rounding or matrix layout conversion separately.
// On malformed data the caller must discard the output row.
inline void decode_nvfp4_row(std::span<const std::uint8_t> codes,
                             std::span<const std::uint8_t> block_scales,
                             float amax, float denominator, std::span<float> output) {
    constexpr std::size_t block_columns = 16;
    if (output.empty() || output.size() % block_columns != 0 ||
        codes.size() != output.size() / 2 || block_scales.size() != output.size() / block_columns) {
        throw std::invalid_argument("NVFP4 row has inconsistent dimensions");
    }
    if (!std::isfinite(amax) || amax < 0.0F || !std::isfinite(denominator) || denominator <= 0.0F) {
        throw std::invalid_argument("NVFP4 outer scale requires nonnegative amax and positive denominator");
    }
    const float outer_scale = amax / denominator;
    if (!std::isfinite(outer_scale)) {
        throw std::invalid_argument("NVFP4 outer scale exceeds FP32 range");
    }
    for (std::size_t block = 0; block < block_scales.size(); ++block) {
        const auto encoded_scale = block_scales[block];
        if (encoded_scale > 0x7eU) {
            throw std::invalid_argument("NVFP4 block scale must be nonnegative finite E4M3");
        }
        const float scale = scalar_float::e4m3(encoded_scale);
        const std::size_t first = block * block_columns;
        for (std::size_t i = first; i < first + block_columns; i += 2) {
            const auto byte = codes[i / 2];
            const auto low_code = static_cast<std::uint8_t>(byte & 0x0fU);
            const auto high_code = static_cast<std::uint8_t>(byte >> 4U);
            const float even = (scalar_float::e2m1(low_code) * scale) * outer_scale;
            const float odd = (scalar_float::e2m1(high_code) * scale) * outer_scale;
            if (!std::isfinite(even) || !std::isfinite(odd)) {
                throw std::invalid_argument("NVFP4 weight exceeds FP32 range");
            }
            output[i] = even;
            output[i + 1] = odd;
        }
    }
}

// MXFP8 has one E8M0 scale per 32 columns of each row. Each E4M3 value is a byte.
inline void decode_mxfp8_row(std::span<const std::uint8_t> codes,
                             std::span<const std::uint8_t> block_scales,
                             std::span<float> output) {
    constexpr std::size_t block_columns = 32;
    if (output.empty() || output.size() % block_columns != 0 ||
        codes.size() != output.size() || block_scales.size() != output.size() / block_columns) {
        throw std::invalid_argument("MXFP8 row has inconsistent dimensions");
    }
    for (std::size_t block = 0; block < block_scales.size(); ++block) {
        const float scale = scalar_float::e8m0(block_scales[block]);
        if (!std::isfinite(scale)) {
            throw std::invalid_argument("MXFP8 block scale must be finite E8M0");
        }
        const std::size_t first = block * block_columns;
        for (std::size_t i = first; i < first + block_columns; ++i) {
            const float weight = scalar_float::e4m3(codes[i]) * scale;
            if (!std::isfinite(weight)) {
                throw std::invalid_argument("MXFP8 weight must be finite in FP32");
            }
            output[i] = weight;
        }
    }
}

} // namespace strix::weight::microscaled
