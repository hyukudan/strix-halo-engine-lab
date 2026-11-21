#pragma once

#include "strix/weight/formats/microscaled.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace strix::core {

// Host reference with the same ascending-column FP32 FMA chain as dot_f32.
// Widen each BF16 tile as it is consumed instead of materializing a decoded row.
// Weights must be backed by an admitted packed matrix. Input and output must not overlap.
template <weight::microscaled::MatrixFormat Format>
void microscaled_matvec_reference(weight::microscaled::PackedMatrix weights, std::span<const float> input,
                                  std::span<float> output) {
    using Operand = weight::microscaled::PackedOperand<Format>;
    using weight::microscaled::RowOrder;
    if (!weights.codes || !weights.block_scales || weights.rows == 0 || weights.columns == 0 ||
        weights.columns % Operand::kBlockColumns != 0 || weights.rows != output.size() ||
        weights.columns != input.size() || weights.rows > std::numeric_limits<std::size_t>::max() / weights.columns ||
        (weights.row_order != RowOrder::contiguous &&
         (weights.row_order != RowOrder::interleaved_halves || weights.rows % 2U != 0))) {
        throw std::invalid_argument("microscaled linear reference: invalid matrix geometry");
    }
    if constexpr (Operand::kNvfp4) {
        if (!std::isfinite(weights.outer_scale) || weights.outer_scale < 0.0F)
            throw std::invalid_argument("microscaled linear reference: invalid outer scale");
    }

    const Operand operand{weights};
    for (std::size_t row = 0; row < output.size(); ++row) {
        float sum = 0.0F;
        for (std::size_t k = 0; k < input.size(); k += Operand::kTileColumns) {
            std::uint32_t packed[Operand::kTileColumns / 2U];
            operand.decode_bf16(operand.load_tile16(row, k), packed);
            for (std::size_t pair = 0; pair < Operand::kTileColumns / 2U; ++pair) {
                const auto lo = std::bit_cast<float>(packed[pair] << 16U);
                const auto hi = std::bit_cast<float>(packed[pair] & 0xffff0000U);
                sum = std::fma(lo, input[k + 2U * pair], sum);
                sum = std::fma(hi, input[k + 2U * pair + 1U], sum);
            }
        }
        if (!std::isfinite(sum)) throw std::invalid_argument("microscaled linear reference: nonfinite projection");
        output[row] = sum;
    }
}

}  // namespace strix::core
