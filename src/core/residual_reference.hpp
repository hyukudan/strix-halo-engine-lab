#pragma once

#include "linear_reference.hpp"
#include "strix/attn/group_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::core::residual_reference {

// Host reference for block-residual attention and previous-pass messages. Rows are independent
// residual states of one token. Normalize once and reuse the bank for multiple learned queries.
[[nodiscard]] inline std::vector<float> normalize_bank(std::span<const float> states, unsigned width,
                                                       float epsilon) {
    if (width == 0 || states.empty() || states.size() % width != 0 ||
        !(epsilon > 0.0F) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("invalid residual bank dimensions or epsilon");
    }

    std::vector<float> normalized(states.size());
    for (std::size_t start = 0; start < states.size(); start += width) {
        const float scale = attn::group_pool::rms_inv(states.data() + start, width, epsilon);
        for (unsigned i = 0; i < width; ++i) {
            normalized[start + i] = states[start + i] * scale;
        }
    }
    return normalized;
}

// The normalized bank supplies both keys and values. Queries and results are [queries][width].
// The dot products, probabilities and output remain FP32. This defines a host reference
// reduction order for comparison by tolerance with native matmul implementations.
[[nodiscard]] inline std::vector<float> mix(std::span<const float> normalized_bank,
                                            std::span<const float> queries, unsigned width) {
    if (width == 0 || normalized_bank.empty() || queries.empty() || normalized_bank.size() % width != 0 ||
        queries.size() % width != 0) {
        throw std::invalid_argument("invalid residual attention dimensions");
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(width));
    std::vector<float> probabilities(normalized_bank.size() / width);
    std::vector<float> output(queries.size());
    for (std::size_t start = 0; start < queries.size(); start += width) {
        const auto query = queries.subspan(start, width);
        for (std::size_t row = 0; row < probabilities.size(); ++row) {
            probabilities[row] = dot_f32(query, normalized_bank.subspan(row * width, width)) * scale;
        }

        const float maximum = *std::max_element(probabilities.begin(), probabilities.end());
        float denominator = 0.0F;
        for (float& probability : probabilities) {
            probability = std::exp(probability - maximum);
            denominator += probability;
        }

        for (std::size_t row = 0; row < probabilities.size(); ++row) {
            const float probability = probabilities[row] / denominator;
            const auto value = normalized_bank.subspan(row * width, width);
            for (unsigned i = 0; i < width; ++i) {
                output[start + i] = std::fma(probability, value[i], output[start + i]);
            }
        }
    }
    return output;
}

enum class DownProjectionPrecision { fp32, bf16 };

struct LowRankGateWeights {
    std::span<const float> down;   // [rank][2 * width], first current then message
    std::span<const float> forget; // [width][rank]
    std::span<const float> gain;   // [width][rank]
};

// A low-rank gate consumes the independently RMS-normalized current and message rows. Its
// forget and gain projections act on the same latent, and the gate mixes the original rows.
// BF16 mode rounds the down input and result to BF16 nearest-even. The callback writes
// one FP32 output per rank row, and the forget/gain path stays FP32.
template <class ProjectDown>
[[nodiscard]] inline std::vector<float> apply_low_rank_gate_with(std::span<const float> current,
                                                                 std::span<const float> message,
                                                                 std::span<const float> forget_weights,
                                                                 std::span<const float> gain_weights, float epsilon,
                                                                 DownProjectionPrecision precision,
                                                                 ProjectDown&& project) {
    if (current.empty() || current.size() != message.size() ||
        current.size() > std::numeric_limits<unsigned>::max() ||
        current.size() > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::invalid_argument("invalid low-rank gate input dimensions");
    }
    const auto width = static_cast<unsigned>(current.size());
    const std::size_t rank = forget_weights.size() / width;
    if (rank == 0 || forget_weights.size() % width != 0 || gain_weights.size() != forget_weights.size()) {
        throw std::invalid_argument("invalid low-rank gate weight dimensions");
    }

    std::vector<float> concatenated(current.begin(), current.end());
    concatenated.insert(concatenated.end(), message.begin(), message.end());
    const auto cast_down = [precision](float value) {
        return precision == DownProjectionPrecision::bf16 ? attn::group_indexer::bf16_round(value) : value;
    };
    std::vector<float> normalized = normalize_bank(concatenated, width, epsilon);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), cast_down);

    std::vector<float> latent(rank);
    project(std::span<const float>(normalized), std::span<float>(latent));
    std::transform(latent.begin(), latent.end(), latent.begin(), cast_down);

    std::vector<float> output(width);
    for (std::size_t i = 0; i < width; ++i) {
        const float forget = std::tanh(dot_f32(forget_weights.subspan(i * rank, rank), latent));
        const float gain = std::tanh(dot_f32(gain_weights.subspan(i * rank, rank), latent));
        output[i] = (1.0F - forget) * current[i] + gain * message[i];
    }
    return output;
}

[[nodiscard]] inline std::vector<float> apply_low_rank_gate(std::span<const float> current,
                                                            std::span<const float> message,
                                                            const LowRankGateWeights& weights, float epsilon,
                                                            DownProjectionPrecision precision) {
    return apply_low_rank_gate_with(current, message, weights.forget, weights.gain, epsilon, precision,
        [&](std::span<const float> input, std::span<float> output) { linear_reference(weights.down, input, output); });
}

} // namespace strix::core::residual_reference
