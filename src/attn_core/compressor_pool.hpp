#pragma once
// Shared HIP bodies of the projected-compressor pool of a sliding-window layer with pooled history rows
// (host contract: attn_core/hybrid_reference.hpp projected_ape_pool_reference in the declared order; key list:
// attn_core/window_pool_keys.hpp). The layer projects every row to a value and a gate logit (width D) and keeps the last
// R rows in a ring (row position % R); when position % R == R - 1 the group of the R rows pools, per channel c:
//   logit_j = gate_j[c] + ape[j][c];  pooled[c] = group_indexer::gated_pool_channel<R>(logits, values, declared_exp)
// (max, e_j = exp(logit_j - max), ascending sum, p_j = e_j / sum, ascending sum of p_j v_j; no contraction). The caller
// norms, rotates and rounds the pooled rows with the shared row ops and stores them as key/value rows.
#include <hip/hip_runtime.h>

#include "strix/attn/group_indexer.hpp"
#include "strix/numeric/declared.hpp"

#include <cstddef>

namespace strix::attn_core::compressor_pool {

struct DeclaredExp {
    __device__ __forceinline__ float operator()(float x) const {
#pragma clang fp contract(off)
        return numeric::declared_exp(x);
    }
};

// Source rows of one launch: the n rows of the current window of rows (positions pos0 .. pos0 + n - 1; value/logit
// [n][dim]) and the ring [R][dim] (the last R rows of earlier windows, slot = position % R).
struct PoolArgs {
    const float *value, *logit, *ape;  // ape [R][dim]
    float *ring_value, *ring_logit;
    float* out;                                                   // [groups][dim]
    unsigned pos0, n, first_group, dim;
};

// grid (groups, ceil(dim / 128)), block 128: lane = channel. Group g (positions g R .. g R + R - 1) closes inside the window.
template <unsigned R>
__global__ void __launch_bounds__(128) pool_kernel(PoolArgs a) {
#pragma clang fp contract(off)
    const unsigned c = blockIdx.y * 128U + threadIdx.x;
    if (c >= a.dim) return;
    const unsigned g = a.first_group + blockIdx.x;
    float logits[R], values[R];
    for (unsigned j = 0; j < R; ++j) {
        const unsigned q = g * R + j;
        const bool fresh = q >= a.pos0;
        const std::size_t at = (fresh ? std::size_t{q - a.pos0} : std::size_t{j}) * a.dim + c;
        logits[j] = (fresh ? a.logit[at] : a.ring_logit[at]) + a.ape[std::size_t{j} * a.dim + c];
        values[j] = fresh ? a.value[at] : a.ring_value[at];
    }
    a.out[std::size_t{blockIdx.x} * a.dim + c] = attn::group_indexer::gated_pool_channel<R>(logits, values, DeclaredExp{});
}

// ring[position % R] = row of the latest window row with that residue (no write race when n > R). grid (R), block 128.
template <unsigned R>
__global__ void __launch_bounds__(128) carry_store_kernel(const float* value, const float* logit, float* ring_value,
                                                          float* ring_logit, unsigned pos0, unsigned n, unsigned dim) {
    const unsigned slot = blockIdx.x, first = (slot + R - pos0 % R) % R;
    if (first >= n) return;
    const unsigned row = first + ((n - 1U - first) / R) * R;
    for (unsigned c = threadIdx.x; c < dim; c += 128U) {
        ring_value[std::size_t{slot} * dim + c] = value[std::size_t{row} * dim + c];
        ring_logit[std::size_t{slot} * dim + c] = logit[std::size_t{row} * dim + c];
    }
}

// Groups that close in rows [pos0, pos0 + n): g in [first_group = pos0 / R, (pos0 + n) / R).
template <unsigned R>
inline hipError_t pool(const PoolArgs& a, hipStream_t stream) {
    const unsigned groups = (a.pos0 + a.n) / R - a.first_group;
    if (groups == 0U) return hipSuccess;
    pool_kernel<R><<<dim3(groups, (a.dim + 127U) / 128U), 128, 0, stream>>>(a);
    return hipGetLastError();
}
template <unsigned R>
inline hipError_t carry_store(const PoolArgs& a, hipStream_t stream) {
    carry_store_kernel<R><<<R, 128, 0, stream>>>(a.value, a.logit, a.ring_value,
                                                  a.ring_logit, a.pos0, a.n, a.dim);
    return hipGetLastError();
}

}  // namespace strix::attn_core::compressor_pool
