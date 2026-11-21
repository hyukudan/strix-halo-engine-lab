#pragma once

#include "moe_core/route.hpp"
#include "strix/moe/route_policy.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strix::cuda::core::route_wave {

namespace detail {

inline constexpr unsigned kRowsPerBlock = 4U;
inline constexpr unsigned kThreads = 32U * kRowsPerBlock;

template <class Traits>
struct Select {
    static constexpr unsigned wave = 32U;
    static constexpr unsigned candidates = Traits::policy.experts / wave;
    unsigned lane;
    unsigned* picks;

    template <class T>
    __device__ static T xor32(T value, int mask) { return __shfl_xor_sync(0xffffffffU, value, mask, 32); }

    __device__ static bool better(float a, unsigned ai, float b, unsigned bi) {
        if constexpr (Traits::policy.key == moe::RouteKey::float_serial) return a > b || (a == b && ai < bi);
        return moe::route_better(Traits::policy.key, a, ai, b, bi);
    }

    __device__ void record(unsigned slot, bool, float, unsigned id) const {
        if (lane == 0) picks[slot] = id;
    }
};

__device__ __forceinline__ bool finite(float value) {
    return (weight::scalar_float::f2u(value) & 0x7f800000U) != 0x7f800000U;
}

template <class Traits>
__global__ __launch_bounds__(kThreads) void kernel(const float* logits, const float* bias, unsigned* ids,
                                                  float* weights, unsigned* status, unsigned rows) {
    using Selection = Select<Traits>;
    constexpr auto Policy = Traits::policy;
    constexpr unsigned experts = Policy.experts;
    constexpr unsigned top_k = Policy.top_k;
    static_assert(Policy.score == moe::RouteScore::sigmoid && Policy.selection_bias && Policy.renormalize &&
                  (top_k > 1 || Policy.renorm_single) && Policy.order == moe::RouteOrder::selection);
    static_assert(experts % Selection::wave == 0 && Selection::candidates <= 32);
    constexpr unsigned all = Selection::candidates == 32 ? ~0U : (1U << Selection::candidates) - 1U;
    __shared__ float score_rows[kRowsPerBlock][experts];
    __shared__ unsigned pick_rows[kRowsPerBlock][top_k];
    const auto warp = threadIdx.x / 32U;
    const auto row = blockIdx.x * kRowsPerBlock + warp;
    if (row >= rows) return;
    const auto lane = threadIdx.x % 32U;
    auto* scores = score_rows[warp];
    auto* picks = pick_rows[warp];
    const auto* input = logits + std::size_t{row} * experts;
    float choices[Selection::candidates];
    bool invalid = false;
#pragma unroll
    for (unsigned i = 0; i < Selection::candidates; ++i) {
        const unsigned e = lane + Selection::wave * i;
        const float logit = input[e];
        const float correction = bias[e];
        const float score = moe::sigmoid_score<typename Traits::Math>(logit);
        scores[e] = score;
        choices[i] = score + correction;
        invalid = invalid || !finite(logit) || !finite(correction) || !finite(choices[i]);
    }

    if (__any_sync(0xffffffffU, invalid)) {
        if (lane == 0) {
            status[row] = 1U;
            for (unsigned k = 0; k < top_k; ++k) {
                ids[std::size_t{row} * top_k + k] = experts;
                weights[std::size_t{row} * top_k + k] = 0.0F;
            }
        }
        return;
    }
    moe_core::route_wave_topk(choices, all, experts, top_k, lane, Selection{lane, picks});
    __syncwarp();
    if (lane != 0) return;

    status[row] = 0U;
    float denominator = 0.0F;
    for (unsigned k = 0; k < top_k; ++k) denominator += scores[picks[k]];
    denominator += Policy.renorm_eps;
    for (unsigned k = 0; k < top_k; ++k) {
        const auto at = std::size_t{row} * top_k + k;
        ids[at] = picks[k];
        weights[at] = (scores[picks[k]] / denominator) * Policy.scale;
    }
}

} // namespace detail

// Contiguous FP32 [row][experts] logits and layer-wide correction bias.
// Result rows preserve score-selection order. Invalid rows get status and
// out-of-range expert IDs so the device plan rejects them before use.
template <class Traits>
inline cudaError_t launch(const float* logits, const float* bias, unsigned* ids, float* weights,
                          unsigned* status, unsigned rows, cudaStream_t stream) {
    static_assert(moe::validate(Traits::policy) == nullptr);
    if (rows == 0) return cudaSuccess;
    if (!logits || !bias || !ids || !weights || !status || rows > 0x7fffffffU) return cudaErrorInvalidValue;
    const auto blocks = (rows - 1U) / detail::kRowsPerBlock + 1U;
    detail::kernel<Traits><<<blocks, detail::kThreads, 0, stream>>>(logits, bias, ids, weights, status, rows);
    return cudaGetLastError();
}

} // namespace strix::cuda::core::route_wave
