#pragma once

#include "core/bf16_lm_head.hpp"
#include "cuda/core/short_convolution.cuh"
#include "inkling2_mini_attention_prepare_device.hpp"
#include "inkling2_mini_dense_cuda.hpp"

namespace strix::inkling2 {

using MiniCudaProjectionInputs = MiniProjectionInputs;
using MiniCudaProjectionWeights = MiniProjectionWeights;
using MiniCudaProjectionWorkspace = MiniProjectionWorkspace;
using MiniCudaPreparedAttention = MiniPreparedAttention;

template <class Resident>
[[nodiscard]] MiniCudaProjectionWeights mini_cuda_projection_weights(const Resident& resident, LayerVisit visit) {
    return mini_device_projection_weights(resident, visit);
}

namespace detail {

struct MiniPreparedCudaBackend : MiniPreparedRowsDevice<cudaStream_t> {
    using Status = cudaError_t;
    static constexpr auto success = cudaSuccess;
    static constexpr auto invalid = cudaErrorInvalidValue;
    cuda_core::DenseArith arithmetic;
    cuda_core::BigCtx* big_context = nullptr;

    Status linear(const float* input, unsigned width, unsigned rows, std::span<const MiniAttentionLinear> views) const {
        return core::for_each_bf16_projection_group<cuda_core::kMaxBf16RowProjections>(views, success,
            [&](std::span<const MiniAttentionLinear> group) {
                return enqueue_mini_cuda_dense_rows(input, width, rows, group, stream, arithmetic, big_context);
            });
    }

    Status convolve(const core::ShortConvolution& plan, unsigned rows, const float* input, const float* lag_weights,
                    const float* dynamic, float* history, float* output) const {
        return cuda_core::short_conv::launch(plan, {rows, input, lag_weights, dynamic, history, output}, stream);
    }

    Status rms_values(float* value, unsigned rows) const {
        const cuda_core::GroupedView layout{rows, kKvHeads, kHeadWidth};
        return cuda_core::rmsnorm_bf16(value, layout, nullptr, value, layout, kMiniAttentionEpsilon, stream);
    }
};

}  // namespace detail

inline cudaError_t enqueue_mini_cuda_attention_prepare(
    LayerVisit visit, unsigned first_position, unsigned rows, const MiniCudaProjectionInputs& inputs,
    const MiniCudaProjectionWeights& weights, const MiniCudaProjectionWorkspace& work,
    const MiniCudaPreparedAttention& output, cudaStream_t stream,
    cuda_core::DenseArith arithmetic = cuda_core::DenseArith::mma3, cuda_core::BigCtx* big_context = nullptr) {
    return enqueue_mini_prepared_attention(detail::MiniPreparedCudaBackend{{stream}, arithmetic, big_context},
                                            visit, first_position, rows, inputs, weights, work, output);
}

inline cudaError_t enqueue_mini_cuda_attention_output(
    unsigned rows, const std::uint16_t* output_weight, const float* attended, const float* gate_logits,
    float* gated, float* output, cudaStream_t stream, cuda_core::DenseArith arithmetic = cuda_core::DenseArith::mma3, cuda_core::BigCtx* big_context = nullptr) {
    return enqueue_mini_prepared_attention_output(detail::MiniPreparedCudaBackend{{stream}, arithmetic, big_context},
                                                   rows, output_weight, attended, gate_logits, gated, output);
}

inline cudaError_t enqueue_mini_cuda_post_convolution(
    unsigned rows, const std::uint16_t* weights, const float* input, float* lag_weights,
    float* history, float* output, cudaStream_t stream) {
    return enqueue_mini_prepared_post_convolution(detail::MiniPreparedCudaBackend{{stream}, cuda_core::DenseArith::mma3},
                                                  rows, weights, input, lag_weights, history, output);
}

inline constexpr unsigned kMiniCudaOutputVocabulary = kOutputVocabulary;

inline cudaError_t enqueue_mini_cuda_logits(
    const std::uint16_t* output_weight, const float* normalized_hidden, float* logits, unsigned rows,
    cudaStream_t stream, cuda_core::DenseArith arithmetic = cuda_core::DenseArith::mma3) {
    return core::enqueue_bf16_softcap_head(rows, {output_weight, kOutputVocabulary, logits}, kLogitSoftcap, stream,
        [&](std::span<const MiniCudaDenseProjection> projections) {
            return enqueue_mini_cuda_dense_rows(normalized_hidden, kHiddenWidth, rows, projections, stream, arithmetic);
        });
}

}  // namespace strix::inkling2
