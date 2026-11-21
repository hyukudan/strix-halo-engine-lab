#pragma once

#include "inkling2_mini_config.hpp"
#include "cuda_core/dense/bf16_rows.hpp"

namespace strix::inkling2 {

using MiniCudaDenseProjection = core::Bf16RowProjection;

inline cudaError_t enqueue_mini_cuda_dense_rows(
    const float* input, unsigned input_width, unsigned rows, std::span<const MiniCudaDenseProjection> projections,
    cudaStream_t stream, cuda_core::DenseArith arithmetic = cuda_core::DenseArith::mma3,
    cuda_core::BigCtx* context = nullptr) {
    return cuda_core::enqueue_bf16_dense_rows(input, input_width, rows, projections, stream, arithmetic, context);
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
