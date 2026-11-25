#pragma once

#include "cuda/core/route_project.cuh"
#include "moe_core/microscaled_r1.hpp"
#include "moe_core/route_plan_r1.hpp"

namespace strix::moe_core::microscaled_r1::detail {

template <packed::MatrixFormat Format, unsigned TopK, unsigned Columns>
__global__ __launch_bounds__(kThreads) void planned_gate_up(Args args) {
    gate_up_assignment<Format, TopK, Columns>(args, blockIdx.y, blockIdx.x);
}

struct PlannedExpertRank {
    const std::uint32_t* group_by_slot;

    template <unsigned TopK>
    __device__ __forceinline__ unsigned position(const Args&, unsigned slot, unsigned) const {
        return group_by_slot[slot];
    }
};

template <packed::MatrixFormat Format, unsigned TopK, unsigned Columns>
__global__ __launch_bounds__(kThreads) void planned_down(Args args, const std::uint32_t* group_by_slot) {
    down_combine_ranked<Format, TopK, Columns, false>(args, PlannedExpertRank{group_by_slot});
}

}  // namespace strix::moe_core::microscaled_r1::detail

namespace strix::cuda::core::moe_route_plan_r1 {

namespace plan_r1 = moe_core::route_plan_r1;
namespace experts_r1 = moe_core::microscaled_r1;

struct RouteBuffers {
    std::uint32_t* ids;
    float* weights;
    std::uint32_t* group_by_slot;
};

namespace detail {

inline constexpr unsigned kRouterThreads = 256U;

template <unsigned Hidden, class Select, class Math, class Dot>
__global__ __launch_bounds__(kRouterThreads)
void route_plan(const float* input, const float* weight, const float* bias, RouteBuffers out) {
    __shared__ float scores[Select::experts];
    __shared__ float choices[Select::experts];
    const unsigned lane = threadIdx.x % Dot::lanes;
    constexpr unsigned warps = kRouterThreads / Dot::lanes;
    for (unsigned expert = threadIdx.x / Dot::lanes; expert < Select::experts; expert += warps) {
        const float logit = Dot::template dot<Hidden>(input, weight + std::size_t{expert} * Hidden, lane);
        if (lane == 0U) {
            const auto pair = moe::route_sigmoid_score_pair<Math>(logit, bias[expert]);
            scores[expert] = pair.unbiased;
            choices[expert] = pair.biased;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        Select::template select_row<1U>({scores, choices}, 0U, out.ids, out.weights);
        plan_r1::compact_groups<Select::top_k>(out.ids, out.group_by_slot);
    }
}

template <plan_r1::MatrixFormat Format, unsigned TopK, unsigned Columns>
cudaError_t launch_gate_up(const experts_r1::Args& args, cudaStream_t stream) {
    const dim3 grid(experts_r1::gate_blocks(args.intermediate, Columns), TopK);
    experts_r1::detail::planned_gate_up<Format, TopK, Columns>
        <<<grid, experts_r1::detail::kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

template <plan_r1::MatrixFormat Format, unsigned TopK, unsigned Columns>
cudaError_t launch_down(const experts_r1::Args& args, const std::uint32_t* groups, cudaStream_t stream) {
    const unsigned blocks = experts_r1::down_blocks(args.hidden, Columns);
    experts_r1::detail::planned_down<Format, TopK, Columns>
        <<<blocks, experts_r1::detail::kThreads, 0, stream>>>(args, groups);
    return cudaGetLastError();
}

template <plan_r1::MatrixFormat Format, unsigned TopK>
cudaError_t launch_experts(const plan_r1::LayerPlan& plan, const experts_r1::Args& args,
                           const std::uint32_t* groups, cudaStream_t stream) {
    if (!experts_r1::fits<Format, TopK>(args)) return cudaErrorInvalidValue;
    cudaError_t status;
    switch (plan.columns_gate_up) {
    case 1U: status = launch_gate_up<Format, TopK, 1U>(args, stream); break;
    case 2U: status = launch_gate_up<Format, TopK, 2U>(args, stream); break;
    default: status = launch_gate_up<Format, TopK, 4U>(args, stream); break;
    }
    if (status != cudaSuccess) return status;
    switch (plan.columns_down) {
    case 1U: return launch_down<Format, TopK, 1U>(args, groups, stream);
    case 2U: return launch_down<Format, TopK, 2U>(args, groups, stream);
    default: return launch_down<Format, TopK, 4U>(args, groups, stream);
    }
}

}  // namespace detail

// One token per launch. Inputs are FP32 [Hidden], [experts][Hidden] and
// [experts]. Each output buffer has TopK entries. All buffers must be
// disjoint and remain live through the stream. Use the incumbent math mode.
template <unsigned Hidden, class Policy, class Math, class Dot = RouteStridedFmaXor32>
cudaError_t launch_route_plan(const plan_r1::LayerPlan& plan, const float* input, const float* weight,
                              const float* bias, RouteBuffers out, cudaStream_t stream) {
    using Select = plan_r1::Selection<Policy>;
    static_assert(Hidden > 0U && Dot::lanes == 32U && Select::experts <= 1024U);
    if (!plan_r1::valid(plan) || plan.hidden != Hidden || !input || !weight || !bias ||
        !out.ids || !out.weights || !out.group_by_slot || out.ids == out.group_by_slot)
        return cudaErrorInvalidValue;
    detail::route_plan<Hidden, Select, Math, Dot><<<1U, detail::kRouterThreads, 0, stream>>>(input, weight, bias, out);
    return cudaGetLastError();
}

// args.ids, args.weights and groups are produced together by launch_route_plan
// on this stream. The matrix directory must have been validated for plan.format.
// The plan's geometry only affects which columns share a warp's input loads.
template <unsigned TopK>
cudaError_t launch_experts(const plan_r1::LayerPlan& plan, const experts_r1::Args& args,
                           const std::uint32_t* groups, cudaStream_t stream) {
    if (!plan_r1::valid(plan) || plan.hidden != args.hidden || plan.intermediate != args.intermediate || !groups)
        return cudaErrorInvalidValue;
    if (plan.format == plan_r1::MatrixFormat::nvfp4)
        return detail::launch_experts<plan_r1::MatrixFormat::nvfp4, TopK>(plan, args, groups, stream);
    return detail::launch_experts<plan_r1::MatrixFormat::mxfp8, TopK>(plan, args, groups, stream);
}

}  // namespace strix::cuda::core::moe_route_plan_r1
