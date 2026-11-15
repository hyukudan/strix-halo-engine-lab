#pragma once
// Private helpers shared by the cuda_core kernels. Build flags: --fmad=false --prec-div=true --prec-sqrt=true --ftz=false.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strix::cuda_core {

// Single points for transcendentals (a HIP twin swaps these for its own bit-matched versions).
__device__ __forceinline__ float cc_expf(float x) { return expf(x); }
__device__ __forceinline__ float cc_log1pf(float x) { return log1pf(x); }

__device__ __forceinline__ float cc_silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, cc_expf(-x))); }
__device__ __forceinline__ float cc_sigmoid(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, cc_expf(-x))); }

__device__ __forceinline__ float bf16_lo(std::uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(std::uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// 32-lane xor butterfly (offsets 16,8,4,2,1): every lane ends with the same bits.
__device__ __forceinline__ float warp_sum_xor(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Block of exactly 256 threads: warp butterfly, then the 8 warp sums added in order 0..7. smem: 8 floats.
__device__ __forceinline__ float block_sum256(float v, float* smem) {
    v = warp_sum_xor(v);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) smem[w] = v;
    __syncthreads();
    float s = smem[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) s = __fadd_rn(s, smem[i]);
    __syncthreads();
    return s;
}

}  // namespace strix::cuda_core
