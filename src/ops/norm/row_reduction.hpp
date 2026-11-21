#pragma once

// Device-only row reductions shared by the portable HIP/CUDA operators.
namespace strix::ops::norm {

// The LDS halving tree of ReductionOrder::lane_strided_fma_lds_halving over one block of L lanes: s[l] = part, then
// s[l] = s[l] + s[l + h] for h = L/2 .. 1 (a barrier after each level); every lane returns s[0].
template <unsigned L>
__device__ __forceinline__ float lds_halving_sum(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

}  // namespace strix::ops::norm
