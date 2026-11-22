#pragma once
// F9 `recur` HIP home: the gated delta decode step for one row (gated_delta_policy.hpp has the formula and the policies).
// One block per (value head, tile of kStepValueTile value columns), kStepThreads lanes, lane = key index; the state tile
// is read once into registers, the two key reductions are LDS halving trees over all lanes for the tile's columns.
// gated_delta_step_body is the former recurrent_step_kernel body (src/hip_gdn_launcher.hip), lifted verbatim; the four
// policy hooks replace its gate, lane products and update. ScalarGate keeps that kernel's expressions, so the GDN wrapper
// compiles to the same machine code; PerKeyChannelGate is the KDA step (per-key-channel decay).
#include <hip/hip_runtime.h>

#include "attn_core/recurrent_state.hpp"
#include "ops/recur/gated_delta_policy.hpp"
#include "strix/recurrent_state_type.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ops::recur {

namespace rstate = ::strix::attn_core::rstate;

// graph_status != nullptr: the step is skipped when the graph's sticky status is already set (GDN graph validation).
template <RecurrentStateType S, class Policy>
__device__ __forceinline__ void gated_delta_step_body(
    const Policy& policy,
    const float* qkv,
    const float* beta,
    const float* gate,
    float* state_raw,
    float* core,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim, std::uint32_t* graph_status) {
    auto* state = reinterpret_cast<typename rstate::Codec<S>::Elem*>(state_raw);
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto tile_count =
        head_dim /
        kStepValueTile;
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto value_head =
        block /
        tile_count;
    const auto tile =
        block %
        tile_count;
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    if (value_head >= value_heads ||
        lane >= head_dim) {
        return;
    }

    const auto key_head =
        value_head %
        key_heads;
    const auto value_base =
        tile *
        kStepValueTile;

    const auto q_base =
        static_cast<std::uint64_t>(
            key_head) *
        head_dim;
    const auto k_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim +
        q_base;
    const auto v_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim *
            2ULL +
        static_cast<std::uint64_t>(
            value_head) *
            head_dim;

    __shared__ float reduction[
        kStepThreads *
        kStepValueTile];
    // The first reduction's broadcast must remain immutable while every wave
    // reads it. Reusing reduction[column] for the next tree lets lane zero
    // overwrite a delta before another wave has consumed it.
    __shared__ float deltas[kStepValueTile];

    float retained[kStepValueTile];
    // 16-bit contracts: the thread's 32 contiguous elements load as packed 16-byte runs (one access per 8).
    float packed_run[S == RecurrentStateType::f32 ? 1U : kStepValueTile];
    if constexpr (S != RecurrentStateType::f32)
        rstate::load_run<S, kStepValueTile>(state + (static_cast<std::uint64_t>(value_head) * head_dim + lane) * head_dim + value_base,
            packed_run);
    else
        (void)packed_run;
#pragma unroll
    for (std::uint32_t column = 0;
         column < kStepValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        float stored;
        if constexpr (S == RecurrentStateType::f32) stored = state[state_index];
        else stored = packed_run[column];
        retained[column] =
            stored *
            policy.decay(gate, value_head, head_dim, lane);
        reduction[
            static_cast<std::size_t>(
                lane) *
                kStepValueTile +
            column] =
            policy.lane_term(retained[column], qkv[k_base + lane]);
    }
    __syncthreads();

    for (std::uint32_t stride =
             kStepThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kStepValueTile;
                 ++column) {
                reduction[
                    static_cast<std::size_t>(
                        lane) *
                        kStepValueTile +
                    column] +=
                    reduction[
                        static_cast<std::size_t>(
                            lane + stride) *
                            kStepValueTile +
                        column];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kStepValueTile;
             ++column) {
            const auto value_index =
                value_base +
                column;
            deltas[column] =
                (qkv[
                     v_base +
                     value_index] -
                 reduction[column]) *
                beta[value_head];
        }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t column = 0;
         column < kStepValueTile;
         ++column) {
        const auto delta =
            deltas[column];
        retained[column] = policy.update(qkv[k_base + lane], delta, retained[column]);
        policy.observe(retained[column]);
        reduction[
            static_cast<std::size_t>(
                lane) *
                kStepValueTile +
            column] =
            policy.lane_term(retained[column], qkv[q_base + lane]);
    }
    __syncthreads();

    for (std::uint32_t stride =
             kStepThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kStepValueTile;
                 ++column) {
                reduction[
                    static_cast<std::size_t>(
                        lane) *
                        kStepValueTile +
                    column] +=
                    reduction[
                        static_cast<std::size_t>(
                            lane + stride) *
                            kStepValueTile +
                        column];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kStepValueTile;
             ++column) {
            core[
                static_cast<std::uint64_t>(
                    value_head) *
                    head_dim +
                value_base +
                column] =
                reduction[column];
        }
    }

#pragma unroll
    for (std::uint32_t column = 0;
         column < kStepValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        if constexpr (S == RecurrentStateType::f32)
            state[state_index] = retained[column];
        else
            (void)state_index;
    }
    if constexpr (S != RecurrentStateType::f32)
        rstate::store_run<S, kStepValueTile>(state + (static_cast<std::uint64_t>(value_head) * head_dim + lane) * head_dim + value_base,
            retained);
}

// Launch: grid value_heads * (head_dim / kStepValueTile), block kStepThreads, head_dim == kStepThreads.
template <RecurrentStateType S, class Policy>
__global__ __launch_bounds__(kStepThreads) void gated_delta_step_kernel(
    const float* qkv, const float* beta, const float* gate, float* state_raw, float* core, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, std::uint32_t* graph_status, Policy policy) {
    gated_delta_step_body<S>(policy, qkv, beta, gate, state_raw, core, key_heads, value_heads, head_dim, graph_status);
}

}  // namespace strix::ops::recur
