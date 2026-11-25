#pragma once

#include "inkling2_mini_controller_operands.hpp"
#include "cuda/core/dense/bf16_rows.hpp"

namespace strix::inkling2 {

struct MiniControllerDenseLaunch {
    cudaError_t status;
    core::Bf16Operand operand;
};

// The CUDA controller dense dispatch (MiniResidentBackend::dense with a declared operand class).
inline MiniControllerDenseLaunch enqueue_controller_dense(const MiniControllerCertificate& certificate, const float* input,
                                                         unsigned width, unsigned rows,
                                                         std::span<const core::Bf16RowProjection> projections,
                                                         cudaStream_t stream, cuda_core::BigCtx* context,
                                                         core::Bf16Operand declared) {
    const auto operand = certificate.route(declared);
    return {cuda_core::enqueue_bf16_dense_rows(input, width, rows, projections, stream, cuda_core::DenseArith::mma3,
                                              context, nullptr, operand), operand};
}
}  // namespace strix::inkling2
