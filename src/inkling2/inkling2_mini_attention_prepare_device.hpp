#pragma once

#include "attn_core/prepared_gqa.hpp"
#include "inkling2_mini_attention_prepare.hpp"

#include <string>

namespace strix::inkling2 {

using MiniProjectionInputs = attn_core::PreparedGqaInputs;
using MiniProjectionWeights = attn_core::PreparedGqaWeights;
using MiniProjectionWorkspace = attn_core::PreparedGqaWorkspace;
using MiniPreparedAttention = attn_core::PreparedGqaRows;
using MiniAttentionOutputRows = attn_core::PreparedGqaOutputRows;
using MiniAttentionLinear = attn_core::PreparedGqaLinear;

// Four immutable lag-major tables per physical layer. All logical visits of that layer share them.
inline constexpr unsigned kMiniConvLayerElements = 2U * (kKvWidth + kHiddenWidth) * kConvTaps;

inline const float* mini_convolution_layer(const float* storage, LayerVisit visit) {
    if (!valid_mini_visit(visit)) throw std::invalid_argument("invalid Mini convolution visit");
    return storage + visit.physical_layer * kMiniConvLayerElements;
}

inline const float* mini_post_convolution_weights(const float* storage, LayerVisit visit, bool after_moe) {
    return mini_convolution_layer(storage, visit) + (2U * kKvWidth + (after_moe ? kHiddenWidth : 0U)) * kConvTaps;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
template <class Resident>
auto bind_mini_convolution_weights(const Resident& resident, float* storage,
                                   core::row_transforms::Stream stream) {
    namespace prep = core::row_transforms;
    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
        const auto prefix = "layers." + std::to_string(layer) + ".";
        const bool full = is_full_attention(layer);
        const auto one = [&](const char* suffix, unsigned width, bool channel_taps) {
            const auto source = resident.bf16(prefix + suffix);
            if (source.size() != std::size_t{width} * kConvTaps)
                throw std::invalid_argument("invalid Mini convolution weight shape");
            const auto status = attn_core::enqueue_bind_convolution_weights(source, storage, width, kConvTaps, channel_taps, stream);
            storage += source.size();
            return status;
        };
        const char* key = full ? "attn.k_sconv.static_w" : "attn.k_sconv.weight";
        const char* value = full ? "attn.v_sconv.static_w" : "attn.v_sconv.weight";
        if (auto status = one(key, kKvWidth, !full); status != prep::Status{}) return status;
        if (auto status = one(value, kKvWidth, !full); status != prep::Status{}) return status;
        if (auto status = one("attn_sconv.weight", kHiddenWidth, true); status != prep::Status{}) return status;
        if (auto status = one("mlp_sconv.weight", kHiddenWidth, true); status != prep::Status{}) return status;
    }
    return prep::Status{};
}

#endif

template <class Resident>
[[nodiscard]] MiniProjectionWeights mini_device_projection_weights(const Resident& resident, LayerVisit visit,
                                                                   const float* lag_weights) {
    if (!valid_mini_visit(visit)) throw std::invalid_argument("invalid Mini device projection visit");
    MiniProjectionWeights result;
    const auto prefix = "layers." + std::to_string(visit.physical_layer) + ".";
    const auto read = [&](std::string_view suffix, std::size_t expected) {
        const auto view = resident.bf16(prefix + std::string(suffix));
        if (view.size() != expected) throw std::invalid_argument("invalid Mini projection weight: " + std::string(suffix));
        return view.data();
    };
    const bool full = is_full_attention(visit.physical_layer);
    using Role = MiniAttentionProjection;
    for (unsigned i = 0; i < result.matrix.size(); ++i) {
        const auto role = static_cast<Role>(i);
        if (!full && (role == Role::key_taps || role == Role::value_taps)) continue;
        const auto spec = mini_attention_projection_spec(role);
#if defined(__HIPCC__)
        if (role == Role::output) {
            const auto view = resident.bf16_rows(prefix + std::string(spec.tensor_suffix));
            if (view.rows != spec.output_width || view.columns != spec.input_width)
                throw std::invalid_argument("invalid Mini attention output weight shape");
            result.matrix[i] = view.data;
            result.output_row_bytes = static_cast<std::size_t>(view.row_bytes);
            continue;
        }
#endif
        // CUDA uses the compact accessor, which rejects pitched resident weights.
        result.matrix[i] = read(spec.tensor_suffix, std::size_t{spec.input_width} * spec.output_width);
    }
    result.query_gain = read("attn.q_l2_clip.scale", kQueryWidth);
    result.relative_gain = read("attn.rel_l2_clip.scale", kQueryHeads * kRelativeWidth);
    result.key_convolution = lag_weights;
    result.value_convolution = lag_weights + kKvWidth * kConvTaps;
    return result;
}

}  // namespace strix::inkling2
