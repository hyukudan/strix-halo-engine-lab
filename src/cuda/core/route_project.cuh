#pragma once

#include "cuda/core/reduce.cuh"
#include "strix/moe/route_policy.hpp"

namespace strix::cuda::core {

struct RouteStridedFmaXor32 : moe::RouteStridedFmaXor32 {
    template <unsigned Hidden>
    __device__ __forceinline__ static float dot(const float* hidden, const float* weight, unsigned lane) {
        float acc = 0.0F;
        for (std::size_t j = lane; j < Hidden; j += lanes) acc = __fmaf_rn(hidden[j], weight[j], acc);
        return butterfly(acc);
    }
};

namespace detail {

template <unsigned Hidden, unsigned Experts, class Dot, class Math, unsigned Stride>
__global__ void route_sigmoid_projection_kernel(const float* hidden, const float* weight, const float* bias,
                                                std::size_t rows, moe::RouteScorePlanes<Stride> out) {
    const std::size_t pair = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / Dot::lanes;
    if (pair >= rows * Experts) return;
    const unsigned lane = threadIdx.x % Dot::lanes;
    const std::size_t row = pair / Experts, expert = pair % Experts;
    const float logit = Dot::template dot<Hidden>(hidden + row * Hidden, weight + expert * Hidden, lane);
    if (lane == 0U) {
        const auto scores = moe::route_sigmoid_score_pair<Math>(logit, bias[expert]);
        out.unbiased[pair * Stride] = scores.unbiased;
        out.biased[pair * Stride] = scores.biased;
    }
}

}  // namespace detail

// F3 projection and score stage. Inputs are dense FP32 [rows][Hidden],
// [Experts][Hidden] and [Experts]. The output views must address distinct
// elements. The caller selects the top-k stage and owns all buffer storage.
// Use IEEE division, no FTZ and no implicit contraction for EXACT bindings.
template <unsigned Hidden, unsigned Experts, class Dot, class Math, unsigned Stride>
cudaError_t launch_route_sigmoid_projection(const float* hidden, const float* weight, const float* bias,
                                             std::size_t rows, moe::RouteScorePlanes<Stride> out, cudaStream_t stream) {
    static_assert(Hidden > 0U && Experts > 0U && Dot::lanes == 32U);
    constexpr unsigned threads = 256U;
    constexpr unsigned pairs_per_block = threads / Dot::lanes;
    constexpr std::size_t max_pairs = std::size_t{0x7fffffffU} * pairs_per_block;
    if (rows == 0U) return cudaSuccess;
    if (!hidden || !weight || !bias || !out.unbiased || !out.biased || out.unbiased == out.biased ||
        rows > max_pairs / Experts)
        return cudaErrorInvalidValue;
    const auto blocks = static_cast<unsigned>((rows * Experts - 1U) / pairs_per_block + 1U);
    detail::route_sigmoid_projection_kernel<Hidden, Experts, Dot, Math>
        <<<blocks, threads, 0, stream>>>(hidden, weight, bias, rows, out);
    return cudaGetLastError();
}

}  // namespace strix::cuda::core
