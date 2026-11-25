#pragma once

#include "core/dense_f32_input_major_plan.hpp"

#include <cuda_runtime.h>

namespace strix::cuda_core {

struct F32InputMajorRows {
    static constexpr unsigned rows = 16, outputs = 128, k = 64;
};

[[nodiscard]] constexpr core::F32InputMajorGrid plan_f32_input_major_rows(
    std::size_t rows, unsigned in, unsigned out, unsigned out_stride) noexcept {
    return core::plan_f32_input_major_rows<F32InputMajorRows>(
        rows, in, out, out_stride, 0x7fffffffU, 65535U);
}

// An explicitly selected dense rows form. x is [rows][in], w is [in][out],
// and y starts at the first live column of a [rows][out_stride] image.
[[nodiscard]] cudaError_t dense_f32_input_major_rows(const float* x, const float* w, float* y,
    std::size_t rows, unsigned in, unsigned out, unsigned out_stride, cudaStream_t stream);

namespace dense_detail {

// Unlike the warp-owned rows forms, threads here share only the input tile.
// Every output thread retains the full FP32 accumulators across all K tiles.
template <class Tile>
__global__ void f32_input_major_rows(const float* x, const float* w, float* y, std::size_t rows,
                                     unsigned in, unsigned out, unsigned out_stride) {
    __shared__ float xs[Tile::rows][Tile::k];
    const auto first = static_cast<std::size_t>(blockIdx.x) * Tile::rows;
    const unsigned o = blockIdx.y * Tile::outputs + threadIdx.x;
    float acc[Tile::rows]{};
    for (unsigned k0 = 0; k0 < in; k0 += Tile::k) {
        __syncthreads();
        for (unsigned j = threadIdx.x; j < Tile::rows * Tile::k; j += Tile::outputs) {
            const auto r = first + j / Tile::k;
            const auto k = k0 + j % Tile::k;
            xs[j / Tile::k][j % Tile::k] = r < rows ? x[r * in + k] : 0.0F;
        }
        __syncthreads();
        if (o < out) {
            for (unsigned k = 0; k < Tile::k; ++k) {
                const float weight = w[static_cast<std::size_t>(k0 + k) * out + o];
#pragma unroll
                for (unsigned r = 0; r < Tile::rows; ++r)
                    acc[r] = __fmaf_rn(xs[r][k], weight, acc[r]);
            }
        }
    }
    if (o < out) {
        for (unsigned r = 0; r < Tile::rows && first + r < rows; ++r)
            y[(first + r) * out_stride + o] = acc[r];
    }
}

}  // namespace dense_detail
}  // namespace strix::cuda_core
