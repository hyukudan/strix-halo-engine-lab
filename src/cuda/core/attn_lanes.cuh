#pragma once

// Internal: the one-lane binding of a lane-batched prepared append (strix/attn/prepared_lanes.hpp).
// The flash kernel, the cache store and the host admission all derive a lane's arguments here, so
// they are exactly those of append_attention_prepared_bf16 on that lane's row alone.

#include "strix/cuda_core/attention.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace strix::cuda_core::lanes_detail {

struct LaneBinding {
    AttentionDims dims;          // max_positions = the lane's cache slots
    PreparedBf16Flash prepared;  // the lane's bias / sinks rows and wrap rule
    const float* q;
    const float* k;
    const float* v;
    float* out;
    __nv_bfloat16* k_cache;
    __nv_bfloat16* v_cache;
    std::uint32_t pos0;
};

__host__ __device__ inline LaneBinding bind_lane(AttentionDims d, PreparedBf16Flash shape,
                                                 const PreparedBf16LaneImages& images,
                                                 const attn::lanes::PreparedBf16Lane& lane, std::size_t row) {
    const std::size_t query = row * d.heads * d.head_dim, kv = row * d.kv_heads * d.head_dim;
    d.max_positions = lane.cache_slots;
    shape.relative_bias = images.relative_bias ? images.relative_bias + row * d.heads * shape.relative_extent : nullptr;
    shape.sinks = images.sinks ? images.sinks + row * d.heads : nullptr;
    shape.wrapped = lane.wrapped != 0U;
    return {d,
            shape,
            images.q ? images.q + query : nullptr,
            images.k ? images.k + kv : nullptr,
            images.v ? images.v + kv : nullptr,
            images.out ? images.out + query : nullptr,
            reinterpret_cast<__nv_bfloat16*>(lane.keys),
            reinterpret_cast<__nv_bfloat16*>(lane.values),
            lane.first_position};
}

// Flash stage of one launch block (attn_flash.cu); lanes already admitted and stored.
cudaError_t launch_prepared_bf16_lanes_flash(const AttentionDims& d, const PreparedBf16Flash& shape,
                                             const PreparedBf16LaneImages& images,
                                             const attn::lanes::LaneBlock& block, cudaStream_t stream,
                                             FlashPipe pipe);

}  // namespace strix::cuda_core::lanes_detail
