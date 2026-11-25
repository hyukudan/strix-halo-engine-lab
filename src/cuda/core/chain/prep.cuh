#pragma once

#include "key.hpp"
#include "../norms/row_math.cuh"
#include "ops/chain/prep_plan.hpp"
#include "strix/attn/cache_rows.hpp"

namespace strix::cuda::core::chain {

using ops::chain::kPrepFeaturesPerLane;
using ops::chain::kPrepLanes;
using ops::chain::kPrepWidth;
using ops::chain::PrepPlan;
using ops::chain::PrepPlane;
using ops::chain::PrepSequence;

// One full warp owns one head; lanes hold four adjacent FP32 features.
struct WarpQuadRow {
    float values[kPrepFeaturesPerLane];
};

struct WarpQuadCtx {
    const float* gain = nullptr;
    const float* rotary = nullptr;  // [cosine half][sine half] at the resolved position
    unsigned rotary_half = 0;
    float epsilon = 1e-6F;
    float* raw = nullptr;
    std::uint16_t* pending = nullptr;
};

namespace detail {

__device__ __forceinline__ unsigned quad_column() {
    return (threadIdx.x % kPrepLanes) * kPrepFeaturesPerLane;
}

__device__ __forceinline__ WarpQuadRow load_quad(const float* input) {
    WarpQuadRow row;
#pragma unroll
    for (unsigned i = 0; i < kPrepFeaturesPerLane; ++i) row.values[i] = input[quad_column() + i];
    return row;
}

__device__ __forceinline__ void apply_quad(PrepNorm, WarpQuadRow& row, const WarpQuadCtx& ctx) {
    float sum = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < kPrepFeaturesPerLane; ++i)
        sum = __fmaf_rn(row.values[i], row.values[i], sum);
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffU, sum, d, kPrepLanes));
    const float inv = STRIX_CUDA_RMS_INVERSE_RN(sum, static_cast<float>(kPrepWidth), ctx.epsilon);
#pragma unroll
    for (unsigned i = 0; i < kPrepFeaturesPerLane; ++i) {
        const float gain = __fadd_rn(1.0F, ctx.gain[quad_column() + i]);
        row.values[i] = __fmul_rn(__fmul_rn(row.values[i], inv), gain);
    }
}

__device__ __forceinline__ void apply_quad(PrepRope, WarpQuadRow& row, const WarpQuadCtx& ctx) {
    const auto column = quad_column();
    const bool rotary = column < 2U * ctx.rotary_half;
    const bool upper = column >= ctx.rotary_half;
    const auto pair_column = !rotary ? column : upper ? column - ctx.rotary_half : column + ctx.rotary_half;
    const auto pair_lane = static_cast<int>(pair_column / kPrepFeaturesPerLane);
    float partner[kPrepFeaturesPerLane];
#pragma unroll
    for (unsigned i = 0; i < kPrepFeaturesPerLane; ++i)
        partner[i] = __shfl_sync(0xffffffffU, row.values[i], pair_lane, kPrepLanes);
    if (rotary) {
        const auto offset = upper ? pair_column : column;
#pragma unroll
        for (unsigned i = 0; i < kPrepFeaturesPerLane; ++i) {
            const float own = __fmul_rn(row.values[i], ctx.rotary[offset + i]);
            row.values[i] = __fmaf_rn(upper ? partner[i] : -partner[i], ctx.rotary[ctx.rotary_half + offset + i], own);
        }
    }
}

__device__ __forceinline__ void apply_quad(PrepRawStore, const WarpQuadRow& row, const WarpQuadCtx& ctx) {
#pragma unroll
    for (unsigned i = 0; i < kPrepFeaturesPerLane; ++i) ctx.raw[quad_column() + i] = row.values[i];
}

__device__ __forceinline__ void apply_quad(PrepCacheStore, const WarpQuadRow& row, const WarpQuadCtx& ctx) {
#pragma unroll
    for (unsigned i = 0; i < kPrepFeaturesPerLane; ++i)
        ctx.pending[quad_column() + i] = ::strix::attn::cache_rows::encode(row.values[i]);
}

}  // namespace detail

// The complete x-lane warp participates. Gain, rotary operands and store targets are
// prebound at the head, so a producer can also apply these links in registers.
// This numeric form is defined for finite prep arithmetic and cache images.
template <class... Steps>
__device__ __forceinline__ void apply(List<Steps...>, WarpQuadRow& row, const WarpQuadCtx& ctx) {
    static_assert(chain_scope<List<Steps...>> != Scope::unsupported);
    (detail::apply_quad(Steps{}, row, ctx), ...);
}

struct PrepSequenceView {
    PrepSequence layout;
    std::uint16_t* key;
    std::uint16_t* value;
    std::uint16_t* index_key;
};

struct PrepBindings {
    const PrepSequenceView* sequences;  // device views with the validated host layout
    const std::uint32_t* row_sequence;  // device map from each row to its owning sequence
    const float* rotary;  // [position][2][rotary_half]
    float* query;
    const float* key;
    const float* value;
    float* index_query;
    const float* index_key;
    const float* query_gain;
    const float* key_gain;
    const float* index_query_gain;
    const float* index_key_gain;
};

namespace detail {

template <class Chain>
__device__ __forceinline__ void transform_head(Chain list, const float* input, const WarpQuadCtx& ctx) {
    auto row = load_quad(input);
    apply(list, row, ctx);
}

template <class RawChain, class CacheChain>
__global__ __launch_bounds__(kPrepLanes) void prep_kernel(PrepPlan plan, PrepBindings args) {
    const unsigned row = blockIdx.x;
    const auto head = plan.head(blockIdx.y);
    const auto& seq = args.sequences[args.row_sequence[row]];
    const unsigned position = plan.position(seq.layout, row);
    const auto* rotary = args.rotary + plan.rotary_offset(position);
    WarpQuadCtx ctx{};
    ctx.rotary = rotary;
    ctx.rotary_half = plan.shape.rotary_half;
    switch (head.plane) {
    case PrepPlane::query:
        ctx.gain = args.query_gain;
        ctx.raw = args.query + plan.row_head(row, plan.shape.query_heads, head.index);
        transform_head(RawChain{}, ctx.raw, ctx);
        break;
    case PrepPlane::key_value: {
        const auto input = plan.row_head(row, plan.shape.key_heads, head.index);
        const auto output = plan.cache_head<false>(seq.layout, position, head.index);
        ctx.gain = args.key_gain;
        ctx.pending = seq.key + output;
        transform_head(CacheChain{}, args.key + input, ctx);
        ctx.pending = seq.value + output;
        transform_head(PrepValueChain{}, args.value + input, ctx);
        break;
    }
    case PrepPlane::index_query:
        ctx.gain = args.index_query_gain;
        ctx.raw = args.index_query + plan.row_head(row, plan.shape.index_query_heads, head.index);
        transform_head(RawChain{}, ctx.raw, ctx);
        break;
    case PrepPlane::index_key:
        ctx.gain = args.index_key_gain;
        ctx.pending = seq.index_key + plan.cache_head<true>(seq.layout, position, head.index);
        transform_head(CacheChain{}, args.index_key + plan.row_head(row, plan.shape.index_key_heads, head.index), ctx);
        break;
    }
}

}  // namespace detail

// Enqueue a plan returned by plan_prep. Device views, row map and disjoint
// output spans must match the plan and remain live through the stream operation.
inline cudaError_t launch_prep(const PrepPlan& plan, const PrepBindings& args, cudaStream_t stream) {
    if (plan.shape.rows == 0) return cudaSuccess;
    if (!args.sequences || !args.row_sequence || !args.rotary || !args.query || !args.key || !args.value ||
        !args.query_gain || !args.key_gain ||
        (plan.shape.index_query_heads != 0 && (!args.index_query || !args.index_query_gain)) ||
        (plan.shape.index_key_heads != 0 && (!args.index_key || !args.index_key_gain))) return cudaErrorInvalidValue;
    detail::prep_kernel<PrepRawChain, PrepCacheChain>
        <<<dim3(plan.shape.rows, plan.heads), kPrepLanes, 0, stream>>>(plan, args);
    return cudaGetLastError();
}

}  // namespace strix::cuda::core::chain
