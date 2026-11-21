#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
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

// Read or decode one row on demand. Only one row of scratch is needed,
// whether the source is a packed matrix, strided view or external reader.
// Input, output and row scratch must not overlap.
template <class ReadRow, class Convert = std::identity>
void linear_reference_rows(std::span<const float> input, std::span<float> output, std::span<float> row_scratch,
                           ReadRow&& read_row, Convert convert = {}) {
    if (input.empty() || row_scratch.size() != input.size()) {
        throw std::invalid_argument("linear reference: row scratch width differs from the input width");
    }
    for (std::size_t row = 0; row < output.size(); ++row) {
        read_row(row, row_scratch);
        for (float& value : row_scratch) value = convert(value);
        output[row] = dot_f32(row_scratch, input);
    }
}

}  // namespace strix::core
