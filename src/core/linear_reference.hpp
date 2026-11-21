#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace strix::core {

inline float dot_f32(std::span<const float> weights, std::span<const float> input) {
    if (weights.size() != input.size()) {
        throw std::invalid_argument("linear reference: dot operands have different widths");
    }
    float sum = 0.0F;
    for (std::size_t i = 0; i < input.size(); ++i) {
        sum = std::fma(weights[i], input[i], sum);
    }
    return sum;
}

// Row-major weights. Input and output must not overlap.
inline void linear_reference(std::span<const float> matrix, std::span<const float> input,
                             std::span<float> output) {
    if (input.empty() || matrix.size() % input.size() != 0 || matrix.size() / input.size() != output.size()) {
        throw std::invalid_argument("linear reference: matrix dimensions do not match input and output");
    }
    for (std::size_t row = 0; row < output.size(); ++row) {
        output[row] = dot_f32(matrix.subspan(row * input.size(), input.size()), input);
    }
}

}  // namespace strix::core
