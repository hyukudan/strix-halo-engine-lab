#pragma once
#include <hip/hip_runtime.h>

namespace strix::glm::detail {

// The same R2 tree over 128 participants with 2 barriers instead of 9 (kTr, 2a''): participants store s[idx]; barrier;
// wave 0 lane l forms levels 64 and 32 exactly as the halving loop does, (s[l] + s[l + 64]) + (s[l + 32] + s[l + 96]),
// then levels 16 .. 1 by shuffle-down (lane l < h: x_l + x_{l + h}, lower index left); lane 0 stores the sum to *res;
// barrier; everyone reads *res. Reuse: the next call's input stores to s precede its first barrier but follow this
// call's second barrier, before which wave 0 finished reading s; the next call's *res store follows its own first
// barrier, which every reader of this call's *res has passed (review).
__device__ __forceinline__ float tree128_w0(float* s, float* res, unsigned idx, bool on, float part) {
    if (on) s[idx] = part;
    __syncthreads();
    if (threadIdx.x < 32) {
        const unsigned l = threadIdx.x;
        float x = (s[l] + s[l + 64]) + (s[l + 32] + s[l + 96]);
#pragma unroll
        for (unsigned h = 16; h > 0; h >>= 1U) {
            const float o = __shfl_down(x, h, 32);
            x = x + o;
        }
        if (l == 0) *res = x;
    }
    __syncthreads();
    return *res;
}

}  // namespace strix::glm::detail
