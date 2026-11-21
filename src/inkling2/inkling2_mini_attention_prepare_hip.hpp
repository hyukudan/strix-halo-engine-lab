#pragma once

#include "core/hip_short_convolution.hpp"
#include "inkling2_mini_attention_hip.hpp"
#include "inkling2_mini_attention_prepare_device.hpp"
#include "inkling2_mini_dense_hip.hpp"

namespace strix::inkling2 {

using MiniHipProjectionInputs = MiniProjectionInputs;
using MiniHipProjectionWeights = MiniProjectionWeights;
using MiniHipProjectionWorkspace = MiniProjectionWorkspace;
using MiniHipPreparedAttention = MiniPreparedAttention;

namespace detail {

struct MiniPreparedHipBackend : MiniPreparedRowsDevice<hipStream_t> {
    using Status = hipError_t;
    static constexpr auto success = hipSuccess;
    static constexpr auto invalid = hipErrorInvalidValue;

    Status linear(const float* input, unsigned width, unsigned rows, std::span<const MiniAttentionLinear> views) const {
        constexpr auto max_projections = core::fma_rows::kGroupMax;
        while (!views.empty()) {
            std::array<MiniHipDenseProjection, max_projections> projections{};
            const auto count = std::min<std::size_t>(max_projections, views.size());
            for (std::size_t i = 0; i < count; ++i) {
                const auto& p = views[i];
                projections[i] = {p.weights, p.outputs, p.output};
            }
            const auto status = enqueue_mini_hip_dense_rows(input, width, rows, std::span(projections).first(count), stream);
            if (status != success) return status;
            views = views.subspan(count);
        }
        return success;
    }

    Status convolve(const core::ShortConvolution& plan, unsigned rows, const float* input, const float* lag_weights,
                    const float* dynamic, float* history, float* output) const {
        return core::hip_short_conv::launch(plan, {rows, input, lag_weights, dynamic, history, output}, stream);
    }

    Status rms_values(float* value, unsigned rows) const {
        return enqueue_mini_hip_rms({value, nullptr, value, rows * kKvHeads, kHeadWidth, kHeadWidth, kHeadWidth}, stream);
    }
};

}  // namespace detail

inline hipError_t enqueue_mini_hip_attention_prepare(
    LayerVisit visit, unsigned first_position, unsigned rows, const MiniProjectionInputs& inputs,
    const MiniProjectionWeights& weights, const MiniProjectionWorkspace& work,
    const MiniPreparedAttention& output, hipStream_t stream) {
    return enqueue_mini_prepared_attention(detail::MiniPreparedHipBackend{{stream}},
                                            visit, first_position, rows, inputs, weights, work, output);
}

inline hipError_t enqueue_mini_hip_attention_output(
    unsigned rows, const std::uint16_t* output_weight, const float* attended, const float* gate_logits,
    float* gated, float* output, hipStream_t stream) {
    return enqueue_mini_prepared_attention_output(detail::MiniPreparedHipBackend{{stream}},
                                                   rows, output_weight, attended, gate_logits, gated, output);
}

inline hipError_t enqueue_mini_hip_post_convolution(
    unsigned rows, const std::uint16_t* weights, const float* input, float* lag_weights,
    float* history, float* output, hipStream_t stream) {
    return enqueue_mini_prepared_post_convolution(detail::MiniPreparedHipBackend{{stream}},
                                                  rows, weights, input, lag_weights, history, output);
}

// KV belongs to the same visit as the two histories in work. On an enqueue
// failure the caller rolls back the visit or discards the transaction.
inline hipError_t enqueue_mini_hip_attention_forward(
    LayerVisit visit, unsigned first_position, unsigned rows, const MiniProjectionInputs& inputs,
    const MiniProjectionWeights& weights, const MiniProjectionWorkspace& work, const MiniPreparedAttention& prepared,
    const MiniAttentionOutputRows& output, const MiniHipAttentionCache& cache, unsigned* failure, hipStream_t stream,
    std::span<float> workspace = {}, unsigned splits = 1) {
    const auto* output_weight = weights.projection(MiniAttentionProjection::output);
    if (!mini_attention_device_aligned(output_weight) || !mini_attention_device_aligned(output.gated) ||
        !mini_attention_device_aligned(output.output) || output.gated == output.output) return hipErrorInvalidValue;
    const MiniHipAttentionRows attention{prepared.query, prepared.key, prepared.value, prepared.relative_logits,
                                         prepared.sinks, output.attended, first_position, rows};
    if (!can_enqueue_mini_hip_attention(visit, attention, cache, failure, workspace.size(), splits))
        return hipErrorInvalidValue;
    if (auto status = enqueue_mini_hip_attention_prepare(visit, first_position, rows, inputs, weights,
            work, prepared, stream); status != hipSuccess) return status;
    if (auto status = enqueue_mini_hip_attention(visit, attention, cache, failure, stream, workspace, splits);
        status != hipSuccess) return status;
    return enqueue_mini_hip_attention_output(rows, output_weight, output.attended, prepared.gate_logits,
                                             output.gated, output.output, stream);
}

}  // namespace strix::inkling2
