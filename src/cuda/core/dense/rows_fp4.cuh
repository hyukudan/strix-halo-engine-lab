#pragma once

#include "strix/core/rows.hpp"
#include <cuda_runtime_api.h>

namespace strix::core::rows { class RowsPlan; }

namespace strix::cuda_core {

// FP32 activation and output. NVFP4 uses the default exact split identity
// and the tensor scale from ViewRef. MXFP8 uses FP32 block dots followed by
// E8M0 scaling and ordered split reduction. Both support the residual chain.
// A caller-owned device workspace can be reused by ordered calls on a
// stream. During capture it must outlive all graph executions.
std::size_t rows_split_k_workspace(const core::rows::Plan& plan);
cudaError_t cuda_rows_grouped(const core::rows::Plan& plan, float* workspace, std::size_t workspace_bytes,
                              cudaStream_t stream);

// Run one MXFP8 launch from the explicitly declared split plan.
std::size_t mxfp8_rows_workspace(const core::rows::RowsPlan&, std::size_t launch_index);
cudaError_t cuda_mxfp8_rows(const core::rows::RowsPlan&, std::size_t launch_index, float* workspace,
                            std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace strix::cuda_core
