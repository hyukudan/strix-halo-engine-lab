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

// The same tree at an explicit virtual lane l, for a fused body whose block is not L lanes along x. Lanes l >= L only
// join the barriers (every thread of the block must call it); lanes l < L add exactly the pairs of
// lds_halving_sum<L>, so the result has the same bits.
template <unsigned L>
__device__ __forceinline__ float lds_halving_sum_lanes(float* s, float part, unsigned l) {
    if (l < L) s[l] = part;
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
