#pragma once

#include "rows_bf16_narrow_plan.hpp"
#include "strix/cuda_core/dense.hpp"

namespace strix::cuda_core {

// R1, one BF16 projection, MMA3, FP32 image rounded to BF16 on store.
[[nodiscard]] cudaError_t dense_bf16_mma3_r1_output_split(const DenseWeight& weight, float* output,
                                                        const float* input, cudaStream_t stream);

} // namespace strix::cuda_core
