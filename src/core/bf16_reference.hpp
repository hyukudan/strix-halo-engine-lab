#pragma once

#include "core/linear_reference.hpp"
#include "strix/attn/group_indexer.hpp"
#include "strix/attn/group_pool.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace strix::core {

inline void round_bf16_reference(std::span<float> row) {
    for (auto& x : row) x = attn::group_indexer::bf16_round(x);
}

// Pointwise sigmoid gate after an upstream projection. The BF16 policy
// models the operand and result stores of the gated output epilogue.
enum class SigmoidGatePrecision { f32, bf16 };

inline void sigmoid_product_reference(std::span<const float> values, std::span<const float> logits,
                                      std::span<float> out, SigmoidGatePrecision precision) {
    if (values.size() != logits.size() || values.size() != out.size())
        throw std::invalid_argument("sigmoid product: row shapes differ");
    const auto round = [precision](float value) {
        return precision == SigmoidGatePrecision::bf16 ? attn::group_indexer::bf16_round(value) : value;
    };
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = round(round(values[i]) * (1.0F / (1.0F + std::exp(-round(logits[i])))));
}

inline void softmax_bf16_reference(std::span<float> row) {
    if (row.empty()) throw std::invalid_argument("BF16 softmax: empty row");
    const auto maximum = *std::ranges::max_element(row);
    for (auto& x : row) x = std::exp(x - maximum);
    const auto total = std::accumulate(row.begin(), row.end(), 0.0F);
    for (auto& x : row) x /= total;
    round_bf16_reference(row);
}

// Shared float storage with explicit BF16 rounding boundaries.
inline void rms_norm_bf16_reference(std::span<float> row, std::span<const float> gain, float epsilon) {
    if (row.empty() || gain.size() != row.size()) throw std::invalid_argument("BF16 RMSNorm: invalid row shape");
    const float inv = attn::group_pool::rms_inv(row.data(), static_cast<unsigned>(row.size()), epsilon);
    for (std::size_t i = 0; i < row.size(); ++i) row[i] = (row[i] * inv) * gain[i];
    round_bf16_reference(row);
}

inline std::vector<float> mix_rms_norm_bf16_reference(std::span<const float> taps, std::span<const float> coefficients,
                       std::span<const float> gain, float epsilon) {
    if (gain.empty() || coefficients.empty() || taps.size() / gain.size() != coefficients.size() ||
        taps.size() % gain.size() != 0U)
        throw std::invalid_argument("BF16 mix RMSNorm: invalid tap shape");
    std::vector<float> row(gain.size());
    for (std::size_t tap = 0; tap < coefficients.size(); ++tap)
        for (std::size_t i = 0; i < row.size(); ++i)
            row[i] += coefficients[tap] * taps[tap * row.size() + i];
    rms_norm_bf16_reference(row, gain, epsilon);
    return row;
}

inline std::vector<float> linear_rows_bf16_reference(std::span<const float> weights, std::span<const float> input,
                           unsigned in_width, unsigned out_width) {
    if (in_width == 0U || input.size() % in_width != 0U)
        throw std::invalid_argument("BF16 linear: invalid input rows");
    const auto rows = input.size() / in_width;
    std::vector<float> output(rows * out_width);
    for (std::size_t row = 0; row < rows; ++row)
        linear_reference(weights, input.subspan(row * in_width, in_width),
                               std::span(output).subspan(row * out_width, out_width));
    round_bf16_reference(output);
    return output;
}

inline void residual_rms_norm_bf16_reference(std::span<float> residual, std::span<const float> branch, std::span<float> normalized,
                   std::span<const float> gain, float epsilon) {
    if (gain.empty() || residual.size() != branch.size() || residual.size() != normalized.size() ||
        residual.size() % gain.size() != 0U)
        throw std::invalid_argument("BF16 residual RMSNorm: invalid row shape");
    for (std::size_t i = 0; i < residual.size(); ++i) normalized[i] = residual[i] + branch[i];
    std::ranges::copy(normalized, residual.begin());
    round_bf16_reference(residual);
    for (std::size_t row = 0; row < residual.size(); row += gain.size())
        rms_norm_bf16_reference(normalized.subspan(row, gain.size()), gain, epsilon);
}

}  // namespace strix::core
