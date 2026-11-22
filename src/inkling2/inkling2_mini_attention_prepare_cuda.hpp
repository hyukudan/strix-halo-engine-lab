#pragma once

#include "core/bf16_lm_head.hpp"
#include "inkling2_mini_attention_prepare_device.hpp"
#include "inkling2_mini_dense_cuda.hpp"

namespace strix::inkling2 {

using MiniCudaProjectionInputs = MiniProjectionInputs;
using MiniCudaProjectionWeights = MiniProjectionWeights;
using MiniCudaProjectionWorkspace = MiniProjectionWorkspace;
using MiniCudaPreparedAttention = MiniPreparedAttention;

namespace detail {

struct MiniPreparedCudaBackend : attn_core::PreparedGqaRowsDevice<MiniPreparedGqaShape, cudaStream_t> {
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

    Status rms_values(float* value, unsigned rows) const {
        const cuda_core::GroupedView layout{rows, kKvHeads, kHeadWidth};
        return cuda_core::rmsnorm_bf16(value, layout, nullptr, value, layout, MiniPreparedGqaShape::kAttentionEpsilon, stream);
    }
};

// The shared preparation issues all linear calls before the first round,
// which consumes raw K. Retain the projection descriptors until that boundary.
struct MiniPreparedCudaProjectionBatch : MiniPreparedCudaBackend {
    unsigned row_count;
    const device::Profile& profile;
    mutable std::array<MiniAttentionLinear, core::rows::kMaxJobs> projections{};
    mutable std::array<cuda_core::Bf16DenseRowGroup, core::rows::kMaxJobs> groups{};
    mutable std::size_t projection_count = 0, group_count = 0;

    Status linear(const float* input, unsigned width, unsigned rows, std::span<const MiniAttentionLinear> views) const {
        if (rows != row_count || views.size() > projections.size() - projection_count) return invalid;
        return core::for_each_bf16_projection_group<cuda_core::kMaxBf16RowProjections>(views, success,
            [&](std::span<const MiniAttentionLinear> group) {
                auto stored = std::span(projections).subspan(projection_count, group.size());
                std::ranges::copy(group, stored.begin());
                groups[group_count++] = {input, width, stored};
                projection_count += stored.size();
                return success;
            });
    }

    Status round(float* row, std::size_t elements, unsigned* failure = nullptr) const {
        if (group_count != 0) {
            const auto status = cuda_core::enqueue_bf16_dense_row_groups(
                std::span(groups).first(group_count), row_count, profile, stream);
            if (status != success) return status;
            group_count = 0;
            projection_count = 0;
        }
        return MiniPreparedCudaBackend::round(row, elements, failure);
    }
};

}  // namespace detail

inline cudaError_t enqueue_mini_cuda_attention_prepare(
    LayerVisit visit, unsigned first_position, unsigned rows, const MiniCudaProjectionInputs& inputs,
    const MiniCudaProjectionWeights& weights, const MiniCudaProjectionWorkspace& work,
    const MiniCudaPreparedAttention& output, cudaStream_t stream,
    cuda_core::DenseArith arithmetic = cuda_core::DenseArith::mma3, cuda_core::BigCtx* big_context = nullptr,
    const std::uint32_t* device_position = nullptr, const device::Profile* rows_profile = nullptr) {
    if (!valid_mini_visit(visit)) return cudaErrorInvalidValue;
    const auto enqueue = [&](const auto& backend) {
        return attn_core::enqueue_prepared_gqa<MiniPreparedGqaShape>(backend, is_full_attention(visit.physical_layer),
            first_position, rows, inputs, weights, work, output, nullptr, device_position);
    };
    const detail::MiniPreparedCudaBackend backend{{stream}, arithmetic, big_context};
    if (rows_profile && rows == 1U && arithmetic == cuda_core::DenseArith::mma3)
        return enqueue(detail::MiniPreparedCudaProjectionBatch{backend, rows, *rows_profile});
    return enqueue(backend);
}

inline cudaError_t enqueue_mini_cuda_attention_output(
    unsigned rows, const std::uint16_t* output_weight, const float* attended, const float* gate_logits,
    float* gated, float* output, cudaStream_t stream, cuda_core::DenseArith arithmetic = cuda_core::DenseArith::mma3, cuda_core::BigCtx* big_context = nullptr) {
    return attn_core::enqueue_prepared_gqa_output<MiniPreparedGqaShape>(detail::MiniPreparedCudaBackend{{stream}, arithmetic, big_context},
                                                   rows, output_weight, attended, gate_logits, gated, output);
}

inline cudaError_t enqueue_mini_cuda_post_convolution(
    unsigned rows, const float* weights, const float* input,
    float* history, float* output, cudaStream_t stream) {
    return attn_core::enqueue_prepared_gqa_post_convolution<MiniPreparedGqaShape>(detail::MiniPreparedCudaBackend{{stream}, cuda_core::DenseArith::mma3},
                                                  rows, weights, input, history, output);
}

// One decode row per lane of a replayed multi-request step, each lane's history resolved on the device.
inline cudaError_t enqueue_mini_cuda_post_convolution_lanes(
    unsigned lanes, const float* weights, const float* input, state::LaneState<float> history, float* output,
    cudaStream_t stream) {
    return attn_core::enqueue_prepared_gqa_post_convolution_lanes<MiniPreparedGqaShape>(
        detail::MiniPreparedCudaBackend{{stream}, cuda_core::DenseArith::mma3}, lanes, weights, input, history, output);
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
