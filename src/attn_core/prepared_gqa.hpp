#pragma once

#include "core/short_convolution.hpp"
#include "strix/attn/group_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::attn_core {

enum class PreparedGqaProjection {
    query, key, value, key_taps, value_taps, relative, relative_bias, sink, output_gate, output,
};

struct PreparedGqaHostWeights {
    std::span<const float> query_gain;      // [query_head][head_width]
    std::span<const float> relative_gain;   // [query_head][relative_width]
    std::span<const float> key_lag_weights;   // [lag][KV channel], current first
    std::span<const float> value_lag_weights; // [lag][KV channel], current first
};

template <class Shape>
struct PreparedGqaRow {
    std::array<float, Shape::kQueryWidth> query;
    std::array<float, Shape::kKvWidth> key;
    std::array<float, Shape::kKvWidth> value;
    std::array<float, Shape::kQueryHeads * Shape::kRelativeExtent> relative_logits;
    std::array<float, Shape::kQueryHeads> sinks;
    std::array<float, Shape::kQueryWidth> output_gate;
};

// Convert channel-major taps, earliest first, to lag-major taps, current first.
template <class T>
[[nodiscard]] std::vector<T> short_conv_lag_weights(
    std::span<const T> channel_taps, std::size_t channels, std::size_t taps) {
    if (channels == 0 || taps == 0 || channel_taps.size() % taps != 0 ||
        channel_taps.size() / taps != channels)
        throw std::invalid_argument("invalid static convolution weights");
    std::vector<T> weights(channel_taps.size());
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t lag = 0; lag < taps; ++lag) {
            weights[lag * channels + channel] = channel_taps[channel * taps + taps - 1U - lag];
        }
    }
    return weights;
}

namespace detail {

inline void round_prepared_gqa_row(std::span<float> row) {
    std::ranges::transform(row, row.begin(), attn::group_indexer::bf16_round);
}

inline void clip_prepared_gqa_head(std::span<float> head, std::span<const float> gain, float threshold,
                                   float epsilon) {
    float squared_norm = 0.0F;
    for (std::size_t i = 0; i < head.size(); ++i) {
        const float scaled = gain.empty() ? head[i] : head[i] * gain[i];
        head[i] = scaled;
        squared_norm = std::fma(scaled, scaled, squared_norm);
    }
    const float scale = std::min(1.0F, threshold / (std::sqrt(squared_norm) + epsilon));
    for (float& value : head) value = attn::group_indexer::bf16_round(value * scale);
}

inline void scale_prepared_gqa_heads(std::span<float> rows, std::span<const float> gains, unsigned width,
                                       float threshold, float position_scale, float epsilon) {
    for (std::size_t base = 0; base < rows.size(); base += width) {
        const auto head = rows.subspan(base, width);
        clip_prepared_gqa_head(head, gains.subspan(base, width), threshold, epsilon);
        if (position_scale != 1.0F) {
            for (float& value : head) value = attn::group_indexer::bf16_round(value * position_scale);
        }
    }
}

template <class Shape, class Project>
void prepare_gqa_kv(PreparedGqaProjection projection, PreparedGqaProjection taps, bool full,
                     std::span<const float> input, std::span<const float> lag_weights,
                     std::span<float> history, std::span<float> out, Project& project) {
    std::array<float, Shape::kKvWidth> raw;
    project(projection, input, raw);
    round_prepared_gqa_row(raw);

    std::array<float, Shape::kKvHeads * Shape::kConvTaps> dynamic_weights;
    std::span<const float> dynamic;
    if (full) {
        project(taps, input, dynamic_weights);
        round_prepared_gqa_row(dynamic_weights);
        dynamic = dynamic_weights;
    }
    const core::ShortConvolution plan{
        Shape::kKvWidth, Shape::kConvTaps, full ? Shape::kKvHeads : 0U, 1.0F,
        full ? Shape::kFullConvDivisor : Shape::kLocalConvDivisor,
    };
    core::causal_short_convolution(plan, raw, lag_weights, dynamic, history, out);
    round_prepared_gqa_row(out);
}

} // namespace detail

// Inputs are the normalized, BF16-rounded Q/K/V streams from the layer adapter.
// Each visit and sequence owns its own K and V histories of (taps - 1) rows.
// A projection supplies FP32 accumulations; this bridge rounds each layer-local
// activation at the published BF16 boundary and keeps gate sigmoid FP32.
template <class Shape, class Project>
[[nodiscard]] PreparedGqaRow<Shape> prepare_gqa_host(
    bool full, unsigned position, std::span<const float> q_input, std::span<const float> k_input,
    std::span<const float> v_input, const PreparedGqaHostWeights& weights, std::span<float> key_history,
    std::span<float> value_history, Project& project) {
    if (position >= Shape::kMaxContext || q_input.size() != Shape::kHiddenWidth ||
        k_input.size() != Shape::kHiddenWidth || v_input.size() != Shape::kHiddenWidth || weights.query_gain.size() != Shape::kQueryWidth ||
        weights.relative_gain.size() != Shape::kQueryHeads * Shape::kRelativeWidth ||
        weights.key_lag_weights.size() != Shape::kConvTaps * Shape::kKvWidth ||
        weights.value_lag_weights.size() != Shape::kConvTaps * Shape::kKvWidth ||
        key_history.size() != (Shape::kConvTaps - 1U) * Shape::kKvWidth || value_history.size() != key_history.size()) {
        throw std::invalid_argument("invalid prepared GQA attention projection shape");
    }
    using Projection = PreparedGqaProjection;
    PreparedGqaRow<Shape> row;
    const float position_scale = full && position >= Shape::kPositionFloor
        ? 1.0F + Shape::kPositionAlpha * std::log(static_cast<float>(position + 1U) /
                                                  static_cast<float>(Shape::kPositionFloor))
        : 1.0F;

    project(Projection::query, q_input, row.query);
    detail::round_prepared_gqa_row(row.query);
    detail::scale_prepared_gqa_heads(row.query, weights.query_gain, Shape::kHeadWidth, Shape::kQkClip,
                                     position_scale, Shape::kAttentionEpsilon);

    detail::prepare_gqa_kv<Shape>(Projection::key, Projection::key_taps, full, k_input, weights.key_lag_weights,
                            key_history, row.key, project);
    detail::prepare_gqa_kv<Shape>(Projection::value, Projection::value_taps, full, v_input, weights.value_lag_weights,
                            value_history, row.value, project);
    for (std::size_t head = 0; head < Shape::kKvHeads; ++head) {
        const auto key_head = std::span<float>(row.key).subspan(head * Shape::kHeadWidth, Shape::kHeadWidth);
        detail::clip_prepared_gqa_head(key_head, {}, Shape::kQkClip, Shape::kAttentionEpsilon);
        const auto value_head = std::span<float>(row.value).subspan(head * Shape::kHeadWidth, Shape::kHeadWidth);
        const float scale = attn::group_pool::rms_inv(value_head.data(), Shape::kHeadWidth, Shape::kAttentionEpsilon);
        for (float& value : value_head) value = attn::group_indexer::bf16_round(value * scale);
    }

    std::array<float, Shape::kQueryHeads * Shape::kRelativeWidth> relative;
    project(Projection::relative, q_input, relative);
    detail::round_prepared_gqa_row(relative);
    detail::scale_prepared_gqa_heads(relative, weights.relative_gain, Shape::kRelativeWidth,
                                       Shape::kRelativeClip, position_scale, Shape::kAttentionEpsilon);
    for (std::size_t head = 0; head < Shape::kQueryHeads; ++head) {
        const auto input = std::span<const float>(relative).subspan(head * Shape::kRelativeWidth, Shape::kRelativeWidth);
        const auto logits = std::span<float>(row.relative_logits).subspan(head * Shape::kRelativeExtent, Shape::kRelativeExtent);
        project(Projection::relative_bias, input, logits);
        detail::round_prepared_gqa_row(logits);
    }

    project(Projection::sink, q_input, row.sinks);
    detail::round_prepared_gqa_row(row.sinks);
    project(Projection::output_gate, q_input, row.output_gate);
    detail::round_prepared_gqa_row(row.output_gate);
    for (float& value : row.output_gate) value = 1.0F / (1.0F + std::exp(-value));
    return row;
}

// Apply the per-channel gate before the output projection. The attended row is
// stored at BF16, as is the gated input to O and the projected hidden row.
template <class Shape, class Project>
void project_gqa_host_output(std::span<const float> attended, std::span<const float> gate,
                                   Project& project, std::span<float> output) {
    if (attended.size() != Shape::kQueryWidth || gate.size() != Shape::kQueryWidth || output.size() != Shape::kHiddenWidth) {
        throw std::invalid_argument("invalid prepared GQA attention output shape");
    }
    std::array<float, Shape::kQueryWidth> gated;
    for (std::size_t i = 0; i < gated.size(); ++i) {
        gated[i] = attn::group_indexer::bf16_round(attn::group_indexer::bf16_round(attended[i]) * gate[i]);
    }
    project(PreparedGqaProjection::output, gated, output);
    detail::round_prepared_gqa_row(output);
}

}  // namespace strix::attn_core
