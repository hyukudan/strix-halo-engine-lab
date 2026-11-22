#pragma once

#include "inkling2_mini_config.hpp"
#include "cuda/core/dense/bf16_rows.hpp"

namespace strix::inkling2 {

using MiniCudaDenseProjection = core::Bf16RowProjection;

inline cuda_core::DenseArith mini_cuda_dense_group_arith(std::span<const MiniCudaDenseProjection> projections,
                                                          cuda_core::DenseArith arithmetic) {
    const bool fast = arithmetic == cuda_core::DenseArith::mma1 || arithmetic == cuda_core::DenseArith::mma2;
    return fast && !projections.empty() && projections.front().round_bf16 ? cuda_core::DenseArith::mma3 : arithmetic;
}

inline cudaError_t enqueue_mini_cuda_dense_rows(
    const float* input, unsigned input_width, unsigned rows, std::span<const MiniCudaDenseProjection> projections,
    cudaStream_t stream, cuda_core::DenseArith arithmetic = cuda_core::DenseArith::mma3,
    cuda_core::BigCtx* context = nullptr) {
    // The BF16-rounded epilogue exists only on the three-term chain: a group whose outputs round to BF16 (one flag
    // per group, checked by the shared entry) keeps mma3 under the FAST dense prefill class.
    const auto group = mini_cuda_dense_group_arith(projections, arithmetic);
    return cuda_core::enqueue_bf16_dense_rows(input, input_width, rows, projections, stream, group, context);
}

inline cudaError_t enqueue_mini_cuda_rmsnorm(const float* input, const std::uint16_t* gain, float* output,
                                            unsigned rows, cudaStream_t stream) {
    const cuda_core::GroupedView layout{rows, 1U, kHiddenWidth};
    return cuda_core::rmsnorm_bf16(input, layout, gain, output, layout, 1.0e-6F, stream);
}

inline cudaError_t enqueue_mini_cuda_embeddings(const std::uint16_t* table, const std::uint16_t* gain,
                                                const std::uint32_t* ids, float* output, unsigned rows,
                                                cudaStream_t stream) {
    return cuda_core::enqueue_bf16_embeddings_rmsnorm(table, kVocabulary, kHiddenWidth, gain, 1.0e-6F,
                                                      ids, output, rows, stream);
}

}  // namespace strix::inkling2
