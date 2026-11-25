#pragma once

#include <cuda_runtime.h>

#include <cstddef>

// Project-owned MiniMax donor, Apache-2.0. Frozen from e123e646f,
// src/cuda/models/minimax/mm_rtx_kernels.cu:54–77 and the width in mm_rtx_kernels.cuh:22.
namespace mmidx_frozen {

constexpr unsigned kHidden = 6144;

__global__ void dense_t_kernel(const float* x, const float* wt, float* y, std::size_t rows, unsigned n, unsigned ys) {
    __shared__ float xs[16][64];
    const std::size_t t0 = static_cast<std::size_t>(blockIdx.x) * 16;
    const unsigned o = blockIdx.y * 128 + threadIdx.x;
    float acc[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) acc[t] = 0.0F;
    for (unsigned i0 = 0; i0 < kHidden; i0 += 64) {
        __syncthreads();
        for (unsigned j = threadIdx.x; j < 16 * 64; j += 128) {
            const std::size_t t = j / 64;
            xs[t][j % 64] = t0 + t < rows ? x[(t0 + t) * kHidden + i0 + j % 64] : 0.0F;
        }
        __syncthreads();
        if (o < n)
            for (unsigned i = 0; i < 64; ++i) {
                const float w = wt[static_cast<std::size_t>(i0 + i) * n + o];
#pragma unroll
                for (int t = 0; t < 16; ++t) acc[t] = __fmaf_rn(xs[t][i], w, acc[t]);
            }
    }
    if (o < n)
        for (std::size_t t = 0; t < 16 && t0 + t < rows; ++t) y[(t0 + t) * ys + o] = acc[t];
}

}  // namespace mmidx_frozen
