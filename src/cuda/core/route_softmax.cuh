#pragma once

#include "moe_core/route.hpp"
#include "strix/moe/route_policy.hpp"
#include "strix/numeric/math_gfx1151.hpp"

namespace strix::cuda::core {
namespace detail {

struct Gfx1151RoutePick {
    static constexpr unsigned wave = 32U;
    static constexpr unsigned candidates = moe::kGfx1151RouteExperts / wave;
    unsigned lane;
    std::uint32_t* ids;
    float* scores;

    template <class T>
    __device__ static T xor32(T v, int mask) { return __shfl_xor_sync(0xffffffffU, v, mask); }

    __device__ static bool better(float a, unsigned ai, float b, unsigned bi) {
        return a > b || (a == b && ai < bi);
    }

    __device__ void record(unsigned slot, bool, float best, std::uint32_t id) const {
        if (lane == 0U) {
            ids[slot] = id;
            scores[slot] = best;
        }
    }
};

// One 256-thread CTA per row. The warp exchange changes the schedule while
// preserving every pair and operand order of the 128-to-1 halving sum.
template <moe::RouteMath Math>
__global__ __launch_bounds__(moe::kGfx1151RouteLanes) void route_softmax_kernel(
    const float* logits, unsigned top_k, bool normalize, std::uint32_t* ids, float* weights,
    moe::RouteStatus* status, numeric::gfx1151::Tables tables) {
    static_assert(Math == moe::RouteMath::gfx1151_v1);
    constexpr unsigned lanes = moe::kGfx1151RouteLanes;
    constexpr unsigned experts = moe::kGfx1151RouteExperts;
    constexpr unsigned wave = Gfx1151RoutePick::wave;
    const unsigned row = blockIdx.x, tid = threadIdx.x, lane = tid % wave, warp = tid / wave;
    __shared__ float partial[lanes];
    __shared__ float probabilities[experts];
    __shared__ float warp_max[lanes / wave];
    __shared__ float denominator;
    __shared__ float best_scores[moe::kGfx1151RouteMaxTopK];
    __shared__ std::uint32_t best_ids[moe::kGfx1151RouteMaxTopK];

    const auto* input = logits + std::size_t{row} * experts;
    const float x0 = input[tid], x1 = input[tid + lanes];
    float maximum = fmaxf(fmaxf(moe_core::kRouteNegativeInfinity, x0), x1);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask));
    if (lane == 0U) warp_max[warp] = maximum;
    __syncthreads();
    maximum = warp_max[0];
#pragma unroll
    for (unsigned w = 1; w < lanes / wave; ++w) maximum = fmaxf(maximum, warp_max[w]);
    const float p0 = numeric::gfx1151::amd_expf(tables, __fsub_rn(x0, maximum));
    const float p1 = numeric::gfx1151::amd_expf(tables, __fsub_rn(x1, maximum));
    partial[tid] = __fadd_rn(__fadd_rn(0.0F, p0), p1);
    __syncthreads();

    if (warp == 0U) {
        float banks[lanes / wave];
#pragma unroll
        for (unsigned b = 0; b < lanes / wave; ++b) banks[b] = partial[lane + wave * b];
#pragma unroll
        for (unsigned stride = lanes / wave / 2U; stride != 0U; stride >>= 1U) {
#pragma unroll
            for (unsigned b = 0; b < stride; ++b) banks[b] = __fadd_rn(banks[b], banks[b + stride]);
        }
        float sum = banks[0];
#pragma unroll
        for (unsigned stride = wave / 2U; stride != 0U; stride >>= 1U)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, stride));
        if (lane == 0U) denominator = sum;
    }
    __syncthreads();
    probabilities[tid] = __fdiv_rn(p0, denominator);
    probabilities[tid + lanes] = __fdiv_rn(p1, denominator);
    __syncthreads();
    if (warp != 0U) return;

    float values[Gfx1151RoutePick::candidates];
    unsigned available = 0U;
#pragma unroll
    for (unsigned i = 0; i < Gfx1151RoutePick::candidates; ++i) {
        values[i] = probabilities[lane + wave * i];
        if (!isnan(values[i])) available |= 1U << i;
    }
    moe_core::route_wave_topk(values, available, experts, top_k, lane,
                              Gfx1151RoutePick{lane, best_ids, best_scores});
    __syncwarp();
    if (lane != 0U) return;

    float selected_sum = 0.0F;
    for (unsigned slot = 0; slot < top_k; ++slot) selected_sum = __fadd_rn(selected_sum, best_scores[slot]);
    if (!normalize) selected_sum = 1.0F;
    const auto base = std::size_t{row} * top_k;
    for (unsigned slot = 0; slot < top_k; ++slot) {
        ids[base + slot] = best_ids[slot];
        weights[base + slot] = __fdiv_rn(best_scores[slot], selected_sum);
    }
    if (status) status[row] = moe::route_status_from_last_id(best_ids[top_k - 1U], experts);
}

}  // namespace detail

// Buffers are token-major: logits [rows][512], ids and weights [rows][top_k],
// and optional status [rows]. The caller keeps the gfx1151 tables live.
inline cudaError_t launch_gfx1151_softmax_route(const moe::RoutePolicy& policy, const float* logits,
                                              std::uint32_t* ids, float* weights, moe::RouteStatus* status,
                                              unsigned rows, numeric::gfx1151::Tables tables, cudaStream_t stream) {
    if (policy.math != moe::RouteMath::gfx1151_v1 || moe::validate(policy)) return cudaErrorInvalidValue;
    if (rows == 0U) return cudaSuccess;
    if (!logits || !ids || !weights || rows > 0x7fffffffU) return cudaErrorInvalidValue;
    detail::route_softmax_kernel<moe::RouteMath::gfx1151_v1><<<rows, moe::kGfx1151RouteLanes, 0, stream>>>(
        logits, policy.top_k, policy.renormalize, ids, weights, status, tables);
    return cudaGetLastError();
}

}  // namespace strix::cuda::core
