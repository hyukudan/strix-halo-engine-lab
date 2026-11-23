#pragma once

#include "strix/moe/route_policy.hpp"

#include <cuda_runtime.h>

namespace strix::cuda::core {
namespace detail {

template <class Policy, unsigned Stride>
__global__ void route_score_planes_kernel(moe::RouteScorePlanes<Stride> planes, std::size_t rows,
                                          std::uint32_t* ids, float* weights) {
    using Selection = moe::RouteSerialScorePlanes<Policy>;
    const auto row = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const auto output = row * Selection::top_k;
    Selection::select_row(planes, row, ids + output, weights + output);
}

}  // namespace detail

// F3's serial selection schedule, one thread per row. Pass the same score
// view used by the projection writer. EXACT requires precise FP32 division
// and no FTZ or reassociation. Zero rows do not inspect pointers or launch.
template <class Policy, unsigned Stride>
cudaError_t launch_route_score_planes(moe::RouteScorePlanes<Stride> planes, std::size_t rows,
                                      std::uint32_t* ids, float* weights, cudaStream_t stream) {
    using Selection = moe::RouteSerialScorePlanes<Policy>;
    constexpr unsigned threads = 128U;
    constexpr std::size_t max_rows = std::size_t{0x7fffffffU} * threads;
    if (rows == 0U) return cudaSuccess;
    if (!planes.unbiased || !planes.biased || planes.unbiased == planes.biased || !ids || !weights ||
        rows > max_rows || rows > std::numeric_limits<std::size_t>::max() / Stride / Selection::experts)
        return cudaErrorInvalidValue;
    const auto blocks = static_cast<unsigned>((rows - 1U) / threads + 1U);
    detail::route_score_planes_kernel<Policy><<<blocks, threads, 0, stream>>>(planes, rows, ids, weights);
    return cudaGetLastError();
}

}  // namespace strix::cuda::core
