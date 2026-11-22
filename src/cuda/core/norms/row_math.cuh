#pragma once

#include <cuda_runtime.h>

namespace strix::cuda::core::row_math {

__device__ __forceinline__ float l2_inverse(float square_sum, float epsilon) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(square_sum, epsilon)));
}

__device__ __forceinline__ float rotate_half(float x, float partner, float cosine, float sine, bool add) {
    const float a = __fmul_rn(x, cosine);
    const float b = __fmul_rn(partner, sine);
    return add ? __fadd_rn(a, b) : __fsub_rn(a, b);
}

}  // namespace strix::cuda::core::row_math

// Keep the exact operand evaluation at each call site. An inline function
// changes operand loading and register allocation in the fused chain.
#define STRIX_CUDA_RMS_INVERSE_RN(square_sum, width, epsilon) \
    __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn((square_sum), (width)), (epsilon))))
