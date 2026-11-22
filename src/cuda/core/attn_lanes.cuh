#pragma once

// Internal: the one-lane binding of a lane-batched prepared append (strix/attn/prepared_lanes.hpp).
// The flash kernel, the cache store and the host admission all derive a lane's arguments here, so
// they are exactly those of append_attention_prepared_bf16 on that lane's row alone.

#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/attention_step_lanes.hpp"

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

// Lane l of a replayed step: its caches from the slot directory, its position from the lane column.
__host__ __device__ inline attn::lanes::PreparedBf16Lane resolve_step_lane(const PreparedBf16StepLanes& step,
                                                                         unsigned lane) {
    return {state::lane_plane<std::uint16_t>(step.lanes, step.keys, lane),
            state::lane_plane<std::uint16_t>(step.lanes, step.values, lane),
            state::lane_position(step.lanes, lane), step.cache_slots, step.wrapped};
}

// Flash stage of every lane of a replayed step (attn_flash.cu); geometry admitted, valid lanes stored.
cudaError_t launch_prepared_bf16_step_lanes_flash(const AttentionDims& d, const PreparedBf16Flash& shape,
                                                  const PreparedBf16LaneImages& images,
                                                  const PreparedBf16StepLanes& lanes, cudaStream_t stream,
                                                  FlashPipe pipe);

// Flash stage of one launch block (attn_flash.cu); lanes already admitted and stored.
cudaError_t launch_prepared_bf16_lanes_flash(const AttentionDims& d, const PreparedBf16Flash& shape,
                                             const PreparedBf16LaneImages& images,
                                             const attn::lanes::LaneBlock& block, cudaStream_t stream,
                                             FlashPipe pipe);

}  // namespace strix::cuda_core::lanes_detail
