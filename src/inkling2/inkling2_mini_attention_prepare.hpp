#pragma once

#include "core/short_convolution.hpp"
#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"
#include "strix/attn/group_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::inkling2 {

// A backend binds each role to a bias-free shared linear operator on the
// physical layer's weights. relative_bias takes one head of kRelativeWidth
// input elements and writes kRelativeExtent logits in backward-distance order.
enum class MiniAttentionProjection {
    query, key, value, key_taps, value_taps, relative, relative_bias, sink, output_gate, output,
};

struct MiniAttentionProjectionSpec {
    std::string_view tensor_suffix; // Follows model.llm.layers.<physical layer>.
    unsigned input_width;
    unsigned output_width;
    bool transpose_for_rows = false;
};

[[nodiscard]] constexpr MiniAttentionProjectionSpec mini_attention_projection_spec(MiniAttentionProjection role) {
    using Projection = MiniAttentionProjection;
    switch (role) {
        case Projection::query: return {"attn.wq_du.weight", kHiddenWidth, kQueryWidth};
        case Projection::key: return {"attn.wk_dv.weight", kHiddenWidth, kKvWidth};
        case Projection::value: return {"attn.wv_dv.weight", kHiddenWidth, kKvWidth};
        case Projection::key_taps: return {"attn.k_sconv.weight_proj", kHiddenWidth, kKvHeads * kConvTaps};
        case Projection::value_taps: return {"attn.v_sconv.weight_proj", kHiddenWidth, kKvHeads * kConvTaps};
        case Projection::relative: return {"attn.wr_du.weight", kHiddenWidth, kQueryHeads * kRelativeWidth};
        case Projection::relative_bias: return {"attn.rel_logits_proj.proj", kRelativeWidth, kRelativeExtent, true};
        case Projection::sink: return {"attn.attn_sink_proj.weight", kHiddenWidth, kQueryHeads};
        case Projection::output_gate: return {"attn.gate_attn.weight", kHiddenWidth, kQueryWidth};
        case Projection::output: return {"attn.wo_ud.weight", kQueryWidth, kHiddenWidth};
    }
    throw std::invalid_argument("invalid Inkling 2 Mini attention projection role");
}

struct MiniAttentionPrepWeights {
    std::span<const float> query_gain;      // [query_head][head_width]
    std::span<const float> relative_gain;   // [query_head][relative_width]
    std::span<const float> key_lag_weights;   // [lag][KV channel], current first
    std::span<const float> value_lag_weights; // [lag][KV channel], current first
};

struct MiniAttentionRow {
    std::array<float, kQueryWidth> query;
    std::array<float, kKvWidth> key;
    std::array<float, kKvWidth> value;
    std::array<float, kQueryHeads * kRelativeExtent> relative_logits;
    std::array<float, kQueryHeads> sinks;
    std::array<float, kQueryWidth> output_gate;
};

// Static checkpoint convolutions are [channel][tap], oldest first. The shared
// convolution consumes [lag][channel], current first. Full attention dynamic
// weights are already stored in the latter order and go straight to the core.
[[nodiscard]] inline std::vector<float> mini_short_conv_weights(std::span<const float> channel_taps,
                                                                unsigned channels) {
    if (channels == 0 || channel_taps.size() % kConvTaps != 0 || channel_taps.size() / kConvTaps != channels) {
        throw std::invalid_argument("invalid Inkling 2 Mini static convolution weights");
    }
    std::vector<float> weights(channel_taps.size());
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t lag = 0; lag < kConvTaps; ++lag) {
            weights[lag * channels + channel] = channel_taps[channel * kConvTaps + kConvTaps - 1U - lag];
        }
    }
    return weights;
}

namespace detail {

inline constexpr float kQkClip = 22.627416997969522F;
inline constexpr float kRelativeClip = 8.0F;

inline void round_mini_attention_row(std::span<float> row) {
    std::ranges::transform(row, row.begin(), attn::group_indexer::bf16_round);
}

inline void clip_mini_head(std::span<float> head, std::span<const float> gain, float threshold) {
    float squared_norm = 0.0F;
    for (std::size_t i = 0; i < head.size(); ++i) {
        const float scaled = gain.empty() ? head[i] : head[i] * gain[i];
        head[i] = scaled;
        squared_norm = std::fma(scaled, scaled, squared_norm);
    }
    const float scale = std::min(1.0F, threshold / (std::sqrt(squared_norm) + 1.0e-6F));
    for (float& value : head) value = attn::group_indexer::bf16_round(value * scale);
}

inline void scale_mini_attention_heads(std::span<float> rows, std::span<const float> gains, unsigned width,
                                       float threshold, float position_scale) {
    for (std::size_t base = 0; base < rows.size(); base += width) {
        const auto head = rows.subspan(base, width);
        clip_mini_head(head, gains.subspan(base, width), threshold);
        if (position_scale != 1.0F) {
            for (float& value : head) value = attn::group_indexer::bf16_round(value * position_scale);
        }
    }
}

template <class Project>
void prepare_mini_kv(MiniAttentionProjection projection, MiniAttentionProjection taps, bool full,
                     std::span<const float> input, std::span<const float> lag_weights,
                     std::span<float> history, std::span<float> out, Project& project) {
    std::array<float, kKvWidth> raw;
    project(projection, input, raw);
    round_mini_attention_row(raw);

    std::array<float, kKvHeads * kConvTaps> dynamic_weights;
    std::span<const float> dynamic;
    if (full) {
        project(taps, input, dynamic_weights);
        round_mini_attention_row(dynamic_weights);
        dynamic = dynamic_weights;
    }
    const core::ShortConvolution plan{
        kKvWidth, kConvTaps, full ? kKvHeads : 0U, 1.0F,
        std::sqrt(static_cast<float>(full ? 2U * kConvTaps + 1U : kConvTaps + 1U)),
    };
    core::causal_short_convolution(plan, raw, lag_weights, dynamic, history, out);
    round_mini_attention_row(out);
}

} // namespace detail

// Inputs are the normalized, BF16-rounded Q/K/V streams from the layer adapter.
// Each visit and sequence owns its own K and V histories of (taps - 1) rows.
// A projection supplies FP32 accumulations; this bridge rounds each layer-local
// activation at the published BF16 boundary and keeps gate sigmoid FP32.
template <class Project>
[[nodiscard]] MiniAttentionRow prepare_mini_attention(
    LayerVisit visit, unsigned position, std::span<const float> q_input, std::span<const float> k_input,
    std::span<const float> v_input, const MiniAttentionPrepWeights& weights, std::span<float> key_history,
    std::span<float> value_history, Project& project) {
    if (visit.physical_layer >= kPhysicalLayers || position >= kMaxContext || q_input.size() != kHiddenWidth ||
        k_input.size() != kHiddenWidth || v_input.size() != kHiddenWidth || weights.query_gain.size() != kQueryWidth ||
        weights.relative_gain.size() != kQueryHeads * kRelativeWidth ||
        weights.key_lag_weights.size() != kConvTaps * kKvWidth ||
        weights.value_lag_weights.size() != kConvTaps * kKvWidth ||
        key_history.size() != (kConvTaps - 1U) * kKvWidth || value_history.size() != key_history.size()) {
        throw std::invalid_argument("invalid Inkling 2 Mini attention projection shape");
    }
    using Projection = MiniAttentionProjection;
    MiniAttentionRow row;
    const bool full = is_full_attention(visit.physical_layer);
    const float position_scale = full && position >= 8192U
        ? 1.0F + 0.1F * std::log(static_cast<float>(position + 1U) / 8192.0F)
        : 1.0F;

    project(Projection::query, q_input, row.query);
    detail::round_mini_attention_row(row.query);
    detail::scale_mini_attention_heads(row.query, weights.query_gain, kHeadWidth, detail::kQkClip, position_scale);

    detail::prepare_mini_kv(Projection::key, Projection::key_taps, full, k_input, weights.key_lag_weights,
                            key_history, row.key, project);
    detail::prepare_mini_kv(Projection::value, Projection::value_taps, full, v_input, weights.value_lag_weights,
                            value_history, row.value, project);
    for (std::size_t head = 0; head < kKvHeads; ++head) {
        detail::clip_mini_head(std::span<float>(row.key).subspan(head * kHeadWidth, kHeadWidth), {}, detail::kQkClip);
        const auto value_head = std::span<float>(row.value).subspan(head * kHeadWidth, kHeadWidth);
        const float scale = attn::group_pool::rms_inv(value_head.data(), kHeadWidth, 1.0e-6F);
        for (float& value : value_head) value = attn::group_indexer::bf16_round(value * scale);
    }

    std::array<float, kQueryHeads * kRelativeWidth> relative;
    project(Projection::relative, q_input, relative);
    detail::round_mini_attention_row(relative);
    detail::scale_mini_attention_heads(relative, weights.relative_gain, kRelativeWidth,
                                       detail::kRelativeClip, position_scale);
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
        const auto input = std::span<const float>(relative).subspan(head * kRelativeWidth, kRelativeWidth);
        const auto logits = std::span<float>(row.relative_logits).subspan(head * kRelativeExtent, kRelativeExtent);
        project(Projection::relative_bias, input, logits);
        detail::round_mini_attention_row(logits);
    }

    project(Projection::sink, q_input, row.sinks);
    detail::round_mini_attention_row(row.sinks);
    project(Projection::output_gate, q_input, row.output_gate);
    detail::round_mini_attention_row(row.output_gate);
    for (float& value : row.output_gate) value = 1.0F / (1.0F + std::exp(-value));
    return row;
}

// Apply the per-channel gate before the output projection. The attended row is
// stored at BF16, as is the gated input to O and the projected hidden row.
template <class Project>
void project_mini_attention_output(std::span<const float> attended, std::span<const float> gate,
                                   Project& project, std::span<float> output) {
    if (attended.size() != kQueryWidth || gate.size() != kQueryWidth || output.size() != kHiddenWidth) {
        throw std::invalid_argument("invalid Inkling 2 Mini attention output shape");
    }
    std::array<float, kQueryWidth> gated;
    for (std::size_t i = 0; i < gated.size(); ++i) {
        gated[i] = attn::group_indexer::bf16_round(attn::group_indexer::bf16_round(attended[i]) * gate[i]);
    }
    project(MiniAttentionProjection::output, gated, output);
    detail::round_mini_attention_row(output);
}

} // namespace strix::inkling2
