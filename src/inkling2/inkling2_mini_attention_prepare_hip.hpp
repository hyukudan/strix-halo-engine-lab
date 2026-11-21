#pragma once

#include "inkling2_mini_attention_hip.hpp"
#include "inkling2_mini_attention_prepare_device.hpp"
#include "inkling2_mini_dense_hip.hpp"

namespace strix::inkling2 {

using MiniHipProjectionInputs = MiniProjectionInputs;
using MiniHipProjectionWeights = MiniProjectionWeights;
using MiniHipProjectionWorkspace = MiniProjectionWorkspace;
using MiniHipPreparedAttention = MiniPreparedAttention;

namespace detail {

struct MiniPreparedHipBackend : attn_core::PreparedGqaRowsDevice<MiniPreparedGqaShape, hipStream_t> {
    using Status = hipError_t;
    static constexpr auto success = hipSuccess;
    static constexpr auto invalid = hipErrorInvalidValue;

    Status linear(const float* input, unsigned width, unsigned rows, std::span<const MiniAttentionLinear> views) const {
        return core::for_each_bf16_projection_group<core::fma_rows::kGroupMax>(views, success,
            [&](std::span<const MiniAttentionLinear> group) {
                return enqueue_mini_hip_dense_rows(input, width, rows, group, stream);
            });
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
    if (!valid_mini_visit(visit)) return hipErrorInvalidValue;
    return attn_core::enqueue_prepared_gqa<MiniPreparedGqaShape>(detail::MiniPreparedHipBackend{{stream}},
        is_full_attention(visit.physical_layer), first_position, rows, inputs, weights, work, output);
}

inline hipError_t enqueue_mini_hip_attention_output(
    unsigned rows, const std::uint16_t* output_weight, const float* attended, const float* gate_logits,
    float* gated, float* output, hipStream_t stream) {
    return attn_core::enqueue_prepared_gqa_output<MiniPreparedGqaShape>(detail::MiniPreparedHipBackend{{stream}},
                                                   rows, output_weight, attended, gate_logits, gated, output);
}

inline hipError_t enqueue_mini_hip_post_convolution(
    unsigned rows, const float* weights, const float* input,
    float* history, float* output, hipStream_t stream) {
    return attn_core::enqueue_prepared_gqa_post_convolution<MiniPreparedGqaShape>(detail::MiniPreparedHipBackend{{stream}},
                                                  rows, weights, input, history, output);
}

// KV belongs to the same visit as the two histories in work. On an enqueue
// failure the caller rolls back the visit or discards the transaction.
inline hipError_t enqueue_mini_hip_attention_forward(
    LayerVisit visit, unsigned first_position, unsigned rows, const MiniProjectionInputs& inputs,
    const MiniProjectionWeights& weights, const MiniProjectionWorkspace& work, const MiniPreparedAttention& prepared,
    const MiniAttentionOutputRows& output, const MiniHipAttentionCache& cache, unsigned* failure, hipStream_t stream,
    std::span<float> workspace = {}, unsigned splits = 1) {
    const MiniHipAttentionRows attention{prepared.query, prepared.key, prepared.value, prepared.relative_logits,
                                        prepared.sinks, output.attended, first_position, rows};
    if (!can_enqueue_mini_hip_attention(visit, attention, cache, failure, workspace.size(), splits))
        return hipErrorInvalidValue;
    return attn_core::enqueue_prepared_gqa_forward<MiniPreparedGqaShape>(
        detail::MiniPreparedHipBackend{{stream}}, is_full_attention(visit.physical_layer), first_position, rows,
        inputs, weights, work, prepared, output, failure, [&] {
            return enqueue_mini_hip_attention<true>(visit, attention, cache, failure, stream, workspace, splits);
        });
}

}  // namespace strix::inkling2
