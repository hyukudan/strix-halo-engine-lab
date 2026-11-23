#pragma once

#include <cuda_runtime.h>

#include "attention_rowgroup.cuh"
#include "attn_rowgroup_plan.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace strix::cuda::core::rowgroup {
namespace detail {

template <LaunchPolicy P>
__global__ __launch_bounds__(kUnionThreads) void rowgroup_union_kernel(P p, Union u) { union_body(p, u); }

template <LaunchPolicy P>
__global__ __launch_bounds__(Shape<P>::Threads, 2) void rowgroup_flash_kernel(P p, Union u) { flash_body(p, u); }

inline void check_launch(cudaError_t status, const char* stage) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("rowgroup ") + stage + ": " + cudaGetErrorString(status));
}

template <LaunchPolicy P>
void enqueue_union(const P& p, const LaunchPlan& plan, Union u, cudaStream_t stream) {
    rowgroup_union_kernel<P><<<plan.tiles, kUnionThreads, Shape<P>::UnionSmem, stream>>>(p, u);
    check_launch(cudaGetLastError(), "union launch");
}

template <LaunchPolicy P>
void enqueue_flash(const P& p, const LaunchPlan& plan, Union u, cudaStream_t stream) {
    using S = Shape<P>;
    rowgroup_flash_kernel<P><<<dim3(plan.kv_heads, plan.tiles), S::Threads, S::FlashSmem, stream>>>(p, u);
    check_launch(cudaGetLastError(), "flash launch");
}

}  // namespace detail

// Call once per policy on each CUDA context before the first enqueue and outside graph capture.
// CUDA validates the requested shared memory against the selected device's limits.
template <LaunchPolicy P> void prepare_rowgroup() {
    using S = Shape<P>;
    static_assert(S::UnionSmem <= std::numeric_limits<int>::max() && S::FlashSmem <= std::numeric_limits<int>::max());
    detail::check_launch(cudaFuncSetAttribute(detail::rowgroup_union_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(S::UnionSmem)), "union smem attribute");
    // CUDA requires explicit opt-in for blocks using more than 48 KiB of shared memory.
    if constexpr (S::FlashSmem > 48U * 1024U)
        detail::check_launch(cudaFuncSetAttribute(detail::rowgroup_flash_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(S::FlashSmem)), "flash smem attribute");
}

// Prepare this policy on the current context before enqueuing. Tile count and
// workspace_bytes<P>(tiles) give the required caller-owned device scratch.
// Keep that memory live through completion of both stages.
template <LaunchPolicy P>
void enqueue_rowgroup(P p, unsigned kv_heads, void* workspace, std::size_t bytes, cudaStream_t stream) {
    if (p.rows == 0U) return;
    const auto plan = make_launch_plan<P>(p.rows, kv_heads);
    if (!plan) throw std::invalid_argument("rowgroup launch grid is out of range");
    if (workspace == nullptr || bytes < workspace_bytes<P>(plan->tiles) ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(std::uint32_t) != 0U)
        throw std::invalid_argument("rowgroup union workspace is missing, too small, or misaligned");
    const auto u = carve<P>(workspace, plan->tiles);
    detail::enqueue_union(p, *plan, u, stream);
    detail::enqueue_flash(p, *plan, u, stream);
}

}  // namespace strix::cuda::core::rowgroup
