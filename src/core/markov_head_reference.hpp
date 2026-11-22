#pragma once

#include "core/markov_head_round.hpp"
#include "strix/attn/cache_rows.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace strix::core {

// Fold one LM head row against a low-rank affine output in double
// precision. The caller allocates reusable workspace for the rank columns.
inline float fold_markov_head_row_reference(std::span<const float> lm_row, std::span<const float> weight,
                                            std::span<const float> bias, std::span<double> workspace,
                                            std::span<float> folded) {
    if (folded.empty() || workspace.size() != folded.size() || lm_row.size() != bias.size() ||
        weight.size() % folded.size() != 0U || weight.size() / folded.size() != lm_row.size())
        throw std::invalid_argument("Markov fold: invalid affine shape");
    std::ranges::fill(workspace, 0.0);
    double offset = 0.0;
    for (std::size_t i = 0; i < lm_row.size(); ++i) {
        const double scale = lm_row[i];
        offset += scale * static_cast<double>(bias[i]);
        const auto row = weight.subspan(i * folded.size(), folded.size());
        for (std::size_t j = 0; j < row.size(); ++j) workspace[j] += scale * static_cast<double>(row[j]);
    }
    // Round the FP64 result directly to BF16, correcting an intermediate
    // FP32 midpoint if the exact double is on one side of it.
    const auto bf16 = [](double x) { return attn::cache_rows::decode(markov_fold_bf16(x)); };
    std::ranges::transform(workspace, folded.begin(), bf16);
    return bf16(offset);
}

// BF16 rnn_hidden gate. The affine result is FP32 and laid out as three
// equal slices: retention gate, candidate state and separate output latent.
inline void bf16_markov_gate_reference(std::span<const float> affine, std::span<float> state,
                                      std::span<float> latent) {
    if (state.size() != latent.size() || affine.size() != 3U * state.size())
        throw std::invalid_argument("Markov gate: invalid state shape");
    const auto round = [](float x) { return attn::cache_rows::decode(attn::cache_rows::encode(x)); };
    const auto width = state.size();
    for (std::size_t i = 0; i < width; ++i) {
        const auto gate = round(1.0F / (1.0F + std::exp(-round(affine[i]))));
        const auto candidate = round(std::tanh(round(affine[width + i])));
        state[i] = round(round(gate * state[i]) + round(round(1.0F - gate) * candidate));
        latent[i] = round(std::tanh(round(affine[2U * width + i])));
    }
}

}  // namespace strix::core
