#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace strix::core {

struct ShortConvolution {
    std::size_t channels;
    std::size_t taps;
    // Additional per-token weights are shared by consecutive channels in each group.
    std::size_t dynamic_groups = 0;
    float residual_weight = 0.0F;
    float output_divisor = 1.0F;
};

// Scalar FP32 reference for contiguous, unpadded rows. Input and output are
// [row][channel]. All writable buffers are disjoint from the other arguments.
// Static weights are [lag][channel], dynamic weights are [row][group][lag]. Lag 0
// is the current row. Dynamic weights are optional and, when present, add to the
// static weights on each tap. The projection that supplies them is a separate op.
// History is caller-owned [taps - 1][channel], initialized to zero at sequence start
// and stored oldest first. Keep a separate buffer for each stream and sequence.
inline void causal_short_convolution(
    const ShortConvolution& plan, std::span<const float> input,
    std::span<const float> static_weights, std::span<const float> dynamic_weights,
    std::span<float> history, std::span<float> output) {
    const auto channels = plan.channels;
    const auto taps = plan.taps;
    if (channels == 0 || taps == 0 || input.size() % channels != 0 ||
        output.size() != input.size() || static_weights.size() % channels != 0 ||
        static_weights.size() / channels != taps || history.size() % channels != 0 ||
        history.size() / channels != taps - 1 || !std::isfinite(plan.residual_weight) ||
        !std::isfinite(plan.output_divisor) || plan.output_divisor <= 0.0F) {
        throw std::invalid_argument("short convolution: invalid shape or epilogue");
    }

    const auto rows = input.size() / channels;
    const auto groups = plan.dynamic_groups;
    if (groups == 0) {
        if (!dynamic_weights.empty()) {
            throw std::invalid_argument("short convolution: unexpected dynamic weights");
        }
    } else if (channels % groups != 0 || dynamic_weights.size() % taps != 0 ||
               dynamic_weights.size() / taps % groups != 0 ||
               dynamic_weights.size() / taps / groups != rows) {
        throw std::invalid_argument("short convolution: invalid dynamic weight shape");
    }
    const auto group_width = groups == 0 ? channels : channels / groups;

    for (std::size_t row = 0; row < rows; ++row) {
        const auto input_row = input.subspan(row * channels, channels);
        auto output_row = output.subspan(row * channels, channels);
        std::ranges::fill(output_row, 0.0F);
        for (std::size_t lag = 0; lag < taps; ++lag) {
            const auto sample = lag <= row
                ? input.subspan((row - lag) * channels, channels)
                : std::span<const float>(history).subspan((taps - 1 - (lag - row)) * channels, channels);
            const auto weights = static_weights.subspan(lag * channels, channels);
            for (std::size_t channel = 0; channel < channels; ++channel) {
                float weight = weights[channel];
                if (groups != 0) {
                    const auto group = channel / group_width;
                    weight += dynamic_weights[(row * groups + group) * taps + lag];
                }
                output_row[channel] += sample[channel] * weight;
            }
        }
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const float residual = plan.residual_weight * input_row[channel];
            output_row[channel] = (output_row[channel] + residual) / plan.output_divisor;
        }
    }

    if (!history.empty() && !input.empty()) {
        const auto carried = std::min(history.size(), input.size());
        std::ranges::move(history.subspan(carried), history.begin());
        std::ranges::copy(input.last(carried), history.last(carried).begin());
    }
}

}  // namespace strix::core
