#pragma once
// Shared HIP group-pool core of the compressed group indexer (contract strix/attn/group_indexer.hpp, per-lane
// statements strix/attn/group_pool.hpp). One work-group of head_dim lanes pools one COMPLETE group; lane = channel.
//   GatedSoftmax     (GLM): the engine kernels keep their slot addressing / flags and call gated_bf16_channel or
//                           group_indexer::gated_pool_channel directly (no LDS, no barrier).
//   GatedSoftmaxApe: pool_gated_ape below (projections need every token-key element: LDS + two barriers).
// A group key is pooled once when its group closes and cached; never re-pool closed groups per step.
#include <hip/hip_runtime.h>

#include "strix/attn/group_pool.hpp"

#include <cstdint>

namespace strix::attn_core::group_pool {
namespace gp = strix::attn::group_pool;

// Caller-owned LDS of pool_gated_ape (declaration order / size stay in the kernel, like select.hpp storages).
template <unsigned R, unsigned D> struct ApeLds {
    float keys[R * D];
    float pooled[D];
};

// Block core, blockDim.x == D. key(j, i) -> the cached token key element (as float) of slot j of the group.
// Returns lane's bf16-valued normalised channel (before RoPE). Every lane must call it (two barriers).
template <unsigned R, unsigned D, class Key, class Wt, class Exp>
__device__ __forceinline__ float pool_gated_ape(unsigned lane, const Key& key, const Wt& w, Exp exp_fn, ApeLds<R, D>& lds) {
    for (unsigned j = 0; j < R; ++j) lds.keys[j * D + lane] = key(j, lane);
    __syncthreads();
    const float x = gp::gated_ape_channel<R>(lds.keys, D, w, lane, exp_fn);
    lds.pooled[lane] = x;
    __syncthreads();
    return gp::rms_out(w.norm(lane), x, gp::rms_inv(lds.pooled, D, w.eps()));
}

// Generic pool launch: block b pools group v.group(b). The view V supplies
//   static constexpr unsigned kRatio, kDim;  bool live(b) (uniform);  unsigned group(b);
//   float key(g, j, i);  weights() (v/g/ape/norm/eps members);  float exp(float);
//   void store(g, lane, const float* out)  after a barrier, out [kDim] = the bf16-valued pooled key (RoPE is the
//   view's: a rotary pair reads out[pair]).
template <class V>
__global__ __launch_bounds__(V::kDim) void pool_groups_kernel(V v) {
    __shared__ ApeLds<V::kRatio, V::kDim> lds;
    __shared__ float out[V::kDim];
    const unsigned b = blockIdx.x, lane = threadIdx.x;
    if (!v.live(b)) return;
    const unsigned g = v.group(b);
    const auto key = [&](unsigned j, unsigned i) { return v.key(g, j, i); };
    const auto exp_fn = [&](float x) { return v.exp(x); };
    out[lane] = pool_gated_ape<V::kRatio, V::kDim>(lane, key, v.weights(), exp_fn, lds);
    __syncthreads();
    v.store(g, lane, out);
}

}  // namespace strix::attn_core::group_pool
