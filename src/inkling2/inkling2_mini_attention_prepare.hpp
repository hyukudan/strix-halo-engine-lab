#pragma once

#include "attn_core/prepared_gqa.hpp"
#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"

#include <string_view>

namespace strix::inkling2 {

using MiniAttentionProjection = attn_core::PreparedGqaProjection;

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

struct MiniPreparedGqaShape {
    static constexpr auto kHiddenWidth = inkling2::kHiddenWidth;
    static constexpr auto kQueryWidth = inkling2::kQueryWidth;
    static constexpr auto kKvWidth = inkling2::kKvWidth;
    static constexpr auto kQueryHeads = inkling2::kQueryHeads;
    static constexpr auto kKvHeads = inkling2::kKvHeads;
    static constexpr auto kHeadWidth = inkling2::kHeadWidth;
    static constexpr auto kRelativeWidth = inkling2::kRelativeWidth;
    static constexpr auto kRelativeExtent = inkling2::kRelativeExtent;
    static constexpr auto kConvTaps = inkling2::kConvTaps;
    static constexpr auto kMaxContext = inkling2::kMaxContext;
    static constexpr auto kSlidingWindow = inkling2::kSlidingWindow;
    static constexpr float kQkClip = 22.627416997969522F;
    static constexpr float kRelativeClip = 8.0F;
    static constexpr float kAttentionEpsilon = 1.0e-6F;
    static constexpr unsigned kPositionFloor = 8192U;
    static constexpr float kPositionAlpha = 0.1F;
    static constexpr float kLocalConvDivisor = 2.2360679774997898F;
    static constexpr float kFullConvDivisor = 3.0F;
    static constexpr float kScoreScale = 0.08838834764831844F;
    static constexpr unsigned projection_width(MiniAttentionProjection role) {
        return mini_attention_projection_spec(role).output_width;
    }
};

using MiniAttentionPrepWeights = attn_core::PreparedGqaHostWeights;
using MiniAttentionRow = attn_core::PreparedGqaRow<MiniPreparedGqaShape>;

[[nodiscard]] inline std::vector<float> mini_short_conv_weights(std::span<const float> channel_taps,
                                                                unsigned channels) {
    return attn_core::short_conv_lag_weights(channel_taps, channels, kConvTaps);
}

template <class Project>
[[nodiscard]] MiniAttentionRow prepare_mini_attention(
    LayerVisit visit, unsigned position, std::span<const float> q_input, std::span<const float> k_input,
    std::span<const float> v_input, const MiniAttentionPrepWeights& weights, std::span<float> key_history,
    std::span<float> value_history, Project& project) {
    if (visit.physical_layer >= kPhysicalLayers)
        throw std::invalid_argument("invalid Inkling 2 Mini attention projection visit");
    return attn_core::prepare_gqa_host<MiniPreparedGqaShape>(
        is_full_attention(visit.physical_layer), position, q_input, k_input, v_input, weights,
        key_history, value_history, project);
}

template <class Project>
void project_mini_attention_output(std::span<const float> attended, std::span<const float> gate,
                                   Project& project, std::span<float> output) {
    attn_core::project_gqa_host_output<MiniPreparedGqaShape>(attended, gate, project, output);
}

}  // namespace strix::inkling2
