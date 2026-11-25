#pragma once

#include "attn_core/selected_block_plan.hpp"

#include <cuda_runtime.h>

namespace strix::cuda::core::attn {
namespace selected_blocks {
using namespace ::strix::attn;

// The two score producers preserve their own operand order. In particular,
// this comparison is left-biased on ties (including signed zero) and unordered
// inputs. The staged XOR max may differ per lane on those inputs. The serial
// max starts at -inf. Keep the native exp and division behavior on nonfinites.
__device__ __forceinline__ float ordered_max(float a, float b) { return a < b ? b : a; }
__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000U); }
__device__ __forceinline__ float xor_lane(float x, int delta) {
    return __shfl_xor_sync(0xffffffffU, x, delta, kSelectedBlockLanes);
}

// Build the exact form with --fmad=false --prec-div=true --ftz=false.
static __device__ float block_exp(float x) {
    if (x < -87.0F) return 0.0F;
    if (x > 88.0F) return __uint_as_float(0x7f800000U);
    const float n = rintf(x * 1.44269504088896341F);
    float r = __fmaf_rn(n, -0.693145751953125F, x);
    r = __fmaf_rn(n, -1.42860676533018e-06F, r);
    float p = 2.48015873e-05F;
    p = __fmaf_rn(p, r, 1.98412698e-04F);
    p = __fmaf_rn(p, r, 1.38888889e-03F);
    p = __fmaf_rn(p, r, 8.33333333e-03F);
    p = __fmaf_rn(p, r, 4.16666667e-02F);
    p = __fmaf_rn(p, r, 1.66666667e-01F);
    p = __fmaf_rn(p, r, 0.5F);
    p = __fmaf_rn(p, r, 1.0F);
    p = __fmaf_rn(p, r, 1.0F);
    return p * __uint_as_float(static_cast<std::uint32_t>(static_cast<int>(n) + 127) << 23U);
}

struct LaneValues {
    float v[kSelectedBlockLaneWidth];
};

__device__ __forceinline__ LaneValues load_scalar(const std::uint16_t* p) {
    LaneValues values{};
#pragma unroll
    for (unsigned i = 0; i < kSelectedBlockLaneWidth; ++i)
        values.v[i] = __uint_as_float(static_cast<std::uint32_t>(p[i]) << 16U);
    return values;
}

__device__ __forceinline__ LaneValues load_packed(const std::uint16_t* p) {
    const uint2 b = *reinterpret_cast<const uint2*>(p);
    return {{__uint_as_float(b.x << 16U), __uint_as_float(b.x & 0xffff0000U),
             __uint_as_float(b.y << 16U), __uint_as_float(b.y & 0xffff0000U)}};
}

__device__ __forceinline__ float dot4(const float* q, LaneValues k) {
    float score = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < kSelectedBlockLaneWidth; ++i) score = __fmaf_rn(q[i], k.v[i], score);
    return score;
}

struct OnlineBlock {
    float m, l;
    LaneValues a;

    __device__ __forceinline__ void begin_tile(float tm, bool first) {
        const float mn = ordered_max(m, tm);
        const float corr = first ? 0.0F : block_exp(m - mn);
        l = l * corr;
#pragma unroll
        for (unsigned i = 0; i < kSelectedBlockLaneWidth; ++i) a.v[i] = a.v[i] * corr;
        m = mn;
    }

    __device__ __forceinline__ void add(float probability, LaneValues value) {
        l = l + probability;
#pragma unroll
        for (unsigned i = 0; i < kSelectedBlockLaneWidth; ++i) a.v[i] = __fmaf_rn(probability, value.v[i], a.v[i]);
    }
};

template <class Shape>
__device__ __forceinline__ void finish(const OnlineBlock (&states)[Shape::list_capacity], unsigned count, float* out) {
    float big = neg_inf();
    for (unsigned c = 0; c < count; ++c) big = ordered_max(big, states[c].m);
    float l = 0.0F, weights[Shape::list_capacity];
    for (unsigned c = 0; c < count; ++c) {
        weights[c] = block_exp(states[c].m - big);
        l = __fmaf_rn(states[c].l, weights[c], l);
    }
    if (!out) return;
#pragma unroll
    for (unsigned i = 0; i < kSelectedBlockLaneWidth; ++i) {
        float a = 0.0F;
        for (unsigned c = 0; c < count; ++c) a = __fmaf_rn(states[c].a.v[i], weights[c], a);
        out[i] = a / l;
    }
}

template <class Shape> struct RowGroup {
    const std::uint16_t *k, *v;
    const std::uint32_t* blocks;
    std::uint64_t visible;
    unsigned count, groups;

    __device__ __forceinline__ RowGroup(const SelectedBlockInput& in, std::size_t row, unsigned g, unsigned n)
        : groups(n) {
        const auto seq = in.sequences[in.row_sequence[row]];
        const auto base = (static_cast<std::size_t>(in.layer) * seq.capacity) * n + g;
        k = seq.k + base * Shape::qk_dim;
        v = seq.v + base * Shape::value_dim;
        const auto i = row * n + g;
        blocks = in.blocks + i * Shape::list_capacity;
        count = in.counts[i];
        visible = static_cast<std::uint64_t>(row - seq.row0) + seq.pos0 + 1U;
    }

    __device__ __forceinline__ std::uint64_t start(unsigned c) const {
        return std::uint64_t{blocks[c]} * Shape::block_keys;
    }
    __device__ __forceinline__ unsigned keys(std::uint64_t lo) const {
        const auto remaining = visible - lo;
        return remaining < Shape::block_keys ? static_cast<unsigned>(remaining) : Shape::block_keys;
    }
    __device__ __forceinline__ const std::uint16_t* key(std::uint64_t t) const {
        return k + t * groups * Shape::qk_dim;
    }
    __device__ __forceinline__ const std::uint16_t* value(std::uint64_t t) const {
        return v + t * groups * Shape::value_dim;
    }
};

template <class Shape>
__global__ void warp_kernel(SelectedBlockInput in, unsigned rows, unsigned groups) {
    const auto warp = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kSelectedBlockLanes;
    const unsigned d = (threadIdx.x % kSelectedBlockLanes) * kSelectedBlockLaneWidth;
    const auto heads = static_cast<std::size_t>(groups) * Shape::heads_per_group;
    const auto row = warp / heads;
    if (row >= rows) return;
    const auto g = static_cast<unsigned>((warp % heads) / Shape::heads_per_group);
    const RowGroup<Shape> kv(in, row, g, groups);
    LaneValues q{};
    if (d < Shape::qk_dim) {
#pragma unroll
        for (unsigned i = 0; i < kSelectedBlockLaneWidth; ++i) q.v[i] = in.query[warp * Shape::qk_dim + d + i];
    }
    OnlineBlock states[Shape::list_capacity];
    for (unsigned c = 0; c < kv.count; ++c) {
        OnlineBlock state{neg_inf(), 0.0F, {}};
        const auto lo = kv.start(c);
        const auto keys = kv.keys(lo);
        for (unsigned t0 = 0; t0 < keys; t0 += kSelectedBlockTile) {
            const unsigned tn = min(keys - t0, kSelectedBlockTile);
            float s[kSelectedBlockTile], tm = neg_inf();
            for (unsigned j = 0; j < tn; ++j) {
                float part = 0.0F;
                if (d < Shape::qk_dim) part = dot4(q.v, load_scalar(kv.key(lo + t0 + j) + d));
#pragma unroll
                for (int delta = kSelectedBlockLanes / 2; delta >= 1; delta >>= 1) part = part + xor_lane(part, delta);
                s[j] = part * in.scale;
                tm = ordered_max(tm, s[j]);
            }
            state.begin_tile(tm, t0 == 0U);
            for (unsigned j = 0; j < tn; ++j) {
                const float p = block_exp(s[j] - state.m);
                LaneValues value{};
                if (d < Shape::value_dim) value = load_scalar(kv.value(lo + t0 + j) + d);
                state.add(p, value);
            }
        }
        states[c] = state;
    }
    finish<Shape>(states, kv.count, d < Shape::value_dim ? in.output + warp * Shape::value_dim + d : nullptr);
}

template <class Shape>
__device__ __forceinline__ void stage(const RowGroup<Shape>& kv, std::uint64_t lo, unsigned keys,
                                      std::uint16_t* k, std::uint16_t* v) {
    constexpr unsigned k_vecs = Shape::qk_dim / 8U, v_vecs = Shape::value_dim / 8U;
    constexpr unsigned vecs = k_vecs > v_vecs ? k_vecs : v_vecs;
    for (unsigned i = threadIdx.x; i < keys * vecs; i += Shape::staged_threads) {
        const unsigned j = i / vecs, part = i % vecs;
        if (part < k_vecs) {
            const uint4 packed = reinterpret_cast<const uint4*>(kv.key(lo + j))[part];
            auto* dst = reinterpret_cast<uint2*>(k + j * Shape::k_stride + 8U * part);
            dst[0] = make_uint2(packed.x, packed.y);
            dst[1] = make_uint2(packed.z, packed.w);
        }
        if (part < v_vecs) reinterpret_cast<uint4*>(v)[j * v_vecs + part] =
            reinterpret_cast<const uint4*>(kv.value(lo + j))[part];
    }
}

template <class Shape>
__global__ __launch_bounds__(Shape::staged_threads) void staged_kernel(SelectedBlockInput in, unsigned groups) {
    extern __shared__ uint4 storage[];
    auto* k = reinterpret_cast<std::uint16_t*>(storage);
    // V starts at a 16-byte boundary even for an odd number of padded K rows.
    auto* v = k + Shape::k_elements;
    auto* q = reinterpret_cast<float*>(v + Shape::block_keys * Shape::value_dim);
    const auto row = blockIdx.x, g = blockIdx.y;
    const unsigned w = threadIdx.x / kSelectedBlockLanes, lane = threadIdx.x % kSelectedBlockLanes;
    const unsigned d = lane * kSelectedBlockLaneWidth;
    const auto head = (static_cast<std::size_t>(row) * groups + g) * Shape::heads_per_group + w;
    const RowGroup<Shape> kv(in, row, g, groups);
    float* qh = q + w * Shape::qk_dim;
    for (unsigned i = lane; i < Shape::qk_dim; i += kSelectedBlockLanes) qh[i] = in.query[head * Shape::qk_dim + i];
    OnlineBlock states[Shape::list_capacity];
    for (unsigned c = 0; c < kv.count; ++c) {
        const auto lo = kv.start(c);
        const unsigned keys = kv.keys(lo);
        __syncthreads();
        stage(kv, lo, keys, k, v);
        __syncthreads();
        OnlineBlock state{neg_inf(), 0.0F, {}};
        for (unsigned g0 = 0; g0 < keys; g0 += kSelectedBlockLanes) {
            const unsigned key = g0 + lane;
            float score = neg_inf();
            if (key < keys) {
                const auto* kr = k + key * Shape::k_stride;
                float partial[kSelectedBlockLanes]{};
#pragma unroll
                for (unsigned ll = 0; ll < Shape::qk_dim; ll += kSelectedBlockLaneWidth)
                    partial[ll / kSelectedBlockLaneWidth] = dot4(qh + ll, load_packed(kr + ll));
#pragma unroll
                for (unsigned delta = kSelectedBlockLanes / 2; delta >= 1U; delta >>= 1U) {
#pragma unroll
                    for (unsigned ll = 0; ll < delta; ++ll) partial[ll] = partial[ll] + partial[ll + delta];
                }
                score = partial[0] * in.scale;
            }
            for (unsigned half = 0; half < 2U; ++half) {
                const unsigned t0 = g0 + kSelectedBlockTile * half;
                if (t0 >= keys) break;
                const unsigned tn = min(keys - t0, kSelectedBlockTile);
                const bool active = lane / kSelectedBlockTile == half && lane % kSelectedBlockTile < tn;
                float tm = active ? score : neg_inf();
#pragma unroll
                for (int delta = kSelectedBlockLanes / 2; delta >= 1; delta >>= 1) tm = ordered_max(tm, xor_lane(tm, delta));
                state.begin_tile(tm, t0 == 0U);
                const float p = active ? block_exp(score - state.m) : 0.0F;
                for (unsigned j = 0; j < tn; ++j) {
                    const auto src = static_cast<int>(kSelectedBlockTile * half + j);
                    const float weight = __shfl_sync(0xffffffffU, p, src, kSelectedBlockLanes);
                    LaneValues value{};
                    if (d < Shape::value_dim) value = load_packed(v + (t0 + j) * Shape::value_dim + d);
                    state.add(weight, value);
                }
            }
        }
        states[c] = state;
    }
    finish<Shape>(states, kv.count, d < Shape::value_dim ? in.output + head * Shape::value_dim + d : nullptr);
}
}  // namespace selected_blocks

// Prepare the staged instance once per CUDA context, outside graph capture.
template <class Shape>
cudaError_t prepare_selected_blocks() {
    if constexpr (Shape::staged_bytes > 48U * 1024U) {
        return cudaFuncSetAttribute(selected_blocks::staged_kernel<Shape>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(Shape::staged_bytes));
    } else {
        return cudaSuccess;
    }
}

template <class Shape>
cudaError_t enqueue_selected_blocks(::strix::attn::SelectedBlockSchedule schedule, unsigned rows, unsigned groups,
                                    ::strix::attn::SelectedBlockInput in, cudaStream_t stream) {
    using ::strix::attn::SelectedBlockSchedule;
    const auto plan = ::strix::attn::make_selected_block_plan<Shape>(schedule, rows, groups);
    if (!plan) return cudaErrorInvalidValue;
    if (rows == 0U) return cudaSuccess;
    if (!in.sequences || !in.row_sequence || !in.query || !in.blocks || !in.counts || !in.output)
        return cudaErrorInvalidValue;
    if (schedule == SelectedBlockSchedule::warp)
        selected_blocks::warp_kernel<Shape><<<plan->grid_x, plan->threads, 0, stream>>>(in, rows, groups);
    else
        selected_blocks::staged_kernel<Shape><<<dim3(plan->grid_x, plan->grid_y), plan->threads,
                                                  plan->shared_bytes, stream>>>(in, groups);
    return cudaGetLastError();
}

}  // namespace strix::cuda::core::attn
