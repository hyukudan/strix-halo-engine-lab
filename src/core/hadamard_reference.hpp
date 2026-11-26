#pragma once

#include <bit>
#include <cmath>
#include <span>
#include <stdexcept>

namespace strix::core {

// Ascending-stride Walsh-Hadamard butterfly with orthonormal scaling.
inline void hadamard_reference(std::span<float> row) {
    if (!std::has_single_bit(row.size()))
        throw std::invalid_argument("Hadamard: expected a power-of-two width");
    for (std::size_t stride = 1; stride < row.size(); stride *= 2)
        for (std::size_t base = 0; base < row.size(); base += 2 * stride)
            for (std::size_t j = 0; j < stride; ++j) {
                const float left = row[base + j], right = row[base + j + stride];
                row[base + j] = left + right;
                row[base + j + stride] = left - right;
            }
    const float scale = 1.0F / std::sqrt(static_cast<float>(row.size()));
    for (auto& value : row) value *= scale;
}

} // namespace strix::core
