#pragma once

#include "../../cuda_core/common.cuh"

namespace strix::cuda::recur {

__device__ __forceinline__ float delta_softplus(float x) {
    return x > 20.0f ? x : cuda_core::cc_log1pf(cuda_core::cc_expf(x));
}

// The schedules keep their shared-memory operands local to the kernel. For
// the S1 schedule, passing key arrays through a pointer extends their liveness
// across the reload fence and spills the 128-key instantiations.
__device__ __forceinline__ float add_product(float sum, float x, float y) {
    return __fadd_rn(sum, __fmul_rn(x, y));
}

}  // namespace strix::cuda::recur
