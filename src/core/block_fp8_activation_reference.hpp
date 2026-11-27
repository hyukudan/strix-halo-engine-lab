#pragma once

#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <span>
#include <stdexcept>

namespace strix::core {

// Quantize and restore a row of finite FP32 activation values without changing
// its storage type. Each E4M3 block uses a ceiling UE8M0 scale.
inline void round_block_fp8_activation_reference(std::span<float> row, unsigned columns) {
    if (columns == 0 || row.size() % columns != 0)
        throw std::invalid_argument("block FP8 activation: invalid row width");
    namespace codec = weight::scalar_float;
    constexpr float floor = 1e-10F, maximum = 448.0F;
    for (std::size_t first = 0; first < row.size(); first += columns) {
        auto block = row.subspan(first, columns);
        float amax = floor;
        for (const float value : block) amax = std::fmax(amax, std::fabs(value));
        const auto exponent = codec::log2_ceil_positive_normal(amax * (1.0F / maximum));
        const float scale = codec::pow2f(exponent);
        const float inverse = codec::pow2f(-exponent);
        for (float& value : block) value = codec::e4m3(codec::e4m3_encode(value * inverse)) * scale;
    }
}

} // namespace strix::core
