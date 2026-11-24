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

// The resident backend's prefill projections: each call goes through the shared routed entry, which stores the call's
// BF16 operand image once in operand_image (rows x widest input u16) when its whole-tile projections pay for it
// (same bits as linear() above; other calls make the launches linear() makes).
struct MiniPreparedHipImageBackend : MiniPreparedHipBackend {
    std::span<std::uint16_t> operand_image;

    Status linear(const float* input, unsigned width, unsigned rows, std::span<const MiniAttentionLinear> views) const {
        return core::fma_rows::enqueue_bf16_rows_routed(input, width, rows, views, kVocabulary, operand_image, stream);
    }

    // The output gate stores the gated rows as that image when the output projection reads its whole call from it
    // (attn_core::PreparedGqaGateImage; same bits as gate() + linear()). The plan is admitted before anything is
    // enqueued: the image shares no byte with the gate's inputs, the weights or the outputs.
    core::fma_rows::Bf16ImageCallPlan output_image_plan(unsigned width, unsigned rows,
                                                        std::span<const MiniAttentionLinear> views, const float* input,
                                                        const float* logits) const {
        const float* sources[2] = {input, logits};
        return core::fma_rows::plan_bf16_rows_from_image(width, rows, views, kVocabulary, operand_image, sources);
    }
    Status gate_image(const float* input, const float* logits, std::size_t elements) const {
        if (elements > operand_image.size()) return hipErrorInvalidValue;
        namespace prep = core::row_transforms;
        return prep::elements(elements, prep::Bf16SigmoidProductImage{input, logits, operand_image.data()}, stream);
    }
    Status linear_image(const core::fma_rows::Bf16ImageCallPlan& plan, unsigned width, unsigned rows,
                        std::span<const MiniAttentionLinear> views) const {
        return core::fma_rows::enqueue_bf16_rows_from_image(plan, operand_image, width, rows, views, kVocabulary,
                                                            stream);
    }
};

// The same backend for a visit whose flash projects the relative bias (MiniHipAttentionProjectedPolicy).
struct MiniPreparedHipProjectedBackend : MiniPreparedHipBackend {
    bool relative_bias_in_attention(unsigned) const { return true; }
};

// The resident image backend for a visit whose flash projects the relative bias.
struct MiniPreparedHipImageProjectedBackend : MiniPreparedHipImageBackend {
    bool relative_bias_in_attention(unsigned) const { return true; }
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
    float* gated, float* output, hipStream_t stream, std::size_t output_row_bytes = 0U) {
    return attn_core::enqueue_prepared_gqa_output<MiniPreparedGqaShape>(detail::MiniPreparedHipBackend{{stream}},
                                                   rows, output_weight, attended, gate_logits, gated, output, output_row_bytes);
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
    std::span<float> workspace = {}, unsigned splits = 1, const std::uint32_t* device_position = nullptr,
    bool project_bias = true) {
    const MiniHipAttentionRows attention{prepared.query, prepared.key, prepared.value, prepared.relative_logits,
                                        prepared.sinks, output.attended, first_position, rows};
    if (!can_enqueue_mini_hip_attention(visit, attention, cache, failure, workspace.size(), splits))
        return hipErrorInvalidValue;
    const auto* bias_weights = weights.projection(MiniAttentionProjection::relative_bias);
    if (project_bias &&
        mini_hip_bias_in_attention(rows, splits, device_position, work.relative_features, bias_weights)) {
        // The flash projects the bias: the prepare stage skips the image and its producer (same logits, same flag).
        return attn_core::enqueue_prepared_gqa_forward<MiniPreparedGqaShape>(
            detail::MiniPreparedHipProjectedBackend{{{stream}}}, is_full_attention(visit.physical_layer),
            first_position, rows, inputs, weights, work, prepared, output, failure, [&] {
                return enqueue_mini_hip_attention_projected<true>(visit, attention, cache, work.relative_features,
                                                                  bias_weights, failure, stream);
            }, device_position);
    }
    return attn_core::enqueue_prepared_gqa_forward<MiniPreparedGqaShape>(
        detail::MiniPreparedHipBackend{{stream}}, is_full_attention(visit.physical_layer), first_position, rows,
        inputs, weights, work, prepared, output, failure, [&] {
            return enqueue_mini_hip_attention<true>(visit, attention, cache, failure, stream, workspace, splits,
                                                   device_position);
        }, device_position);
}

}  // namespace strix::inkling2
