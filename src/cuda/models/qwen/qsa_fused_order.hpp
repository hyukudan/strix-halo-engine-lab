// Softmax denominator order of the exact QSA fused attention (qwen_qsa.cu attention_fused_kernel), shared with its host
// test. The reference is softmax_kernel_body: 256 lanes, lane v sums e[i] for i = v, v + 256, ... ascending from 0.0,
// then an LDS tree scratch[v] += scratch[v + stride] for stride 128..1. One warp of the fused kernel owns a head: warp
// lane lam holds the reference lanes v = lam + 32 j (j < 8). The chains are unchanged. Tree strides 128, 64, 32 pair
// v with v + 128 / 64 / 32, i.e. (j, j + 4 / 2 / 1) inside the thread. Strides 16..1 pair lanes lam and lam + stride
// (a shuffle-down tree). Every level adds the same operand pairs as the reference, so the denominator bits are equal.
#pragma once

#if defined(__CUDACC__)
#define QSA_ORDER_FN __host__ __device__ __forceinline__
#else
#define QSA_ORDER_FN inline
#endif

namespace strix::cuda::qwen_qsa::fused_order {

inline constexpr unsigned kRefLanes = 256, kWarpLanes = 32, kPerLane = kRefLanes / kWarpLanes;

// The in-thread part for warp lane `lam`: 8 reference chains, then tree strides 128, 64, 32. `add(a, b)` must be
// IEEE single round-to-nearest (device __fadd_rn); `e(i)` yields the element i < count (it may also store it).
template <class Add, class Elem>
QSA_ORDER_FN float lane_partial(unsigned lam, unsigned count, Add add, Elem e) {
    float x[kPerLane];
#pragma unroll
    for (unsigned j = 0; j < kPerLane; ++j) {
        float s = 0.0F;
        for (unsigned i = lam + kWarpLanes * j; i < count; i += kRefLanes) s = add(s, e(i));
        x[j] = s;
    }
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) x[j] = add(x[j], x[j + 4U]);
    x[0] = add(x[0], x[2]);
    x[1] = add(x[1], x[3]);
    return add(x[0], x[1]);
}

}  // namespace strix::cuda::qwen_qsa::fused_order
