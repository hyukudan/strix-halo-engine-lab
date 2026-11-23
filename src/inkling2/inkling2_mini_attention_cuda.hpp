#pragma once

#include "attn_core/prepared_gqa_attention.hpp"
#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/attention_step_lanes.hpp"
#include "inkling2_mini_attention_prepare.hpp"
#include "inkling2_mini_state.hpp"

#include <span>

namespace strix::inkling2 {

using MiniCudaAttentionRows = attn_core::PreparedGqaAttentionRows;

inline cudaError_t enqueue_mini_cuda_attention(
    LayerVisit visit, const MiniCudaAttentionRows& rows, MiniVisitKvCache cache,
    cudaStream_t stream, cuda_core::FlashPipe pipe = cuda_core::FlashPipe::pad2, unsigned* failure = nullptr,
    const std::uint32_t* device_position = nullptr) {
    if (!valid_mini_visit(visit) || rows.first_position >= kMaxContext ||
        rows.count > kMaxContext - rows.first_position || !rows.keys || !rows.values ||
        !rows.relative_logits || !rows.sinks) return cudaErrorInvalidValue;
    const unsigned window = is_full_attention(visit.physical_layer) ? 0U : MiniPreparedGqaShape::kSlidingWindow;
    const cuda_core::AttentionDims dims{kQueryHeads, kKvHeads, kHeadWidth,
                                        MiniPreparedGqaShape::kScoreScale, cache.slots};
    const cuda_core::PreparedBf16Flash logits{rows.relative_logits, rows.sinks, kRelativeExtent, window, cache.wrapped};
    static_assert(sizeof(__nv_bfloat16) == sizeof(std::uint16_t));
    return cuda_core::append_attention_prepared_bf16(
        dims, logits, rows.queries, rows.keys, rows.values, reinterpret_cast<__nv_bfloat16*>(cache.keys),
        reinterpret_cast<__nv_bfloat16*>(cache.values), rows.first_position, rows.count, rows.output,
        stream, pipe, failure, device_position);
}

// One-row windows of several lanes as one lane-batched append (shared contract strix/attn/prepared_lanes.hpp):
// lane l is packed row l of `rows` with its own cache and position. Each lane's bits are those of
// enqueue_mini_cuda_attention on that row alone. Host positions only.
inline cudaError_t enqueue_mini_cuda_attention_lanes(
    LayerVisit visit, const MiniCudaAttentionRows& rows, std::span<const attn::lanes::PreparedBf16Lane> lanes,
    cudaStream_t stream, unsigned* failure = nullptr) {
    if (!valid_mini_visit(visit) || lanes.empty() || rows.count != lanes.size() || !rows.keys || !rows.values ||
        !rows.relative_logits || !rows.sinks) return cudaErrorInvalidValue;
    for (const auto& lane : lanes)
        if (lane.first_position >= kMaxContext) return cudaErrorInvalidValue;
    const unsigned window = is_full_attention(visit.physical_layer) ? 0U : MiniPreparedGqaShape::kSlidingWindow;
    const cuda_core::AttentionDims dims{kQueryHeads, kKvHeads, kHeadWidth, MiniPreparedGqaShape::kScoreScale, 0U};
    const cuda_core::PreparedBf16Flash shape{nullptr, nullptr, kRelativeExtent, window, false};
    const cuda_core::PreparedBf16LaneImages images{rows.queries, rows.keys, rows.values, rows.relative_logits,
                                                  rows.sinks, rows.output};
    return cuda_core::append_attention_prepared_bf16_lanes(dims, shape, images, lanes, stream,
                                                           cuda_core::FlashPipe::pad2, failure);
}

// The lanes of a replayed multi-request decode step (shared strix/cuda_core/attention_step_lanes.hpp):
// lane l is packed row l; its position and its K/V caches are resolved on the device from the step block
// and the slot directory. Each valid lane's bits are those of enqueue_mini_cuda_attention on that row alone.
inline cudaError_t enqueue_mini_cuda_attention_step_lanes(
    LayerVisit visit, const MiniCudaAttentionRows& rows, const cuda_core::PreparedBf16StepLanes& lanes,
    cudaStream_t stream, unsigned* failure = nullptr) {
    if (!valid_mini_visit(visit) || rows.count != lanes.lanes.lanes || !rows.keys || !rows.values ||
        !rows.relative_logits || !rows.sinks) return cudaErrorInvalidValue;
    const unsigned window = is_full_attention(visit.physical_layer) ? 0U : MiniPreparedGqaShape::kSlidingWindow;
    const cuda_core::AttentionDims dims{kQueryHeads, kKvHeads, kHeadWidth, MiniPreparedGqaShape::kScoreScale, 0U};
    const cuda_core::PreparedBf16Flash shape{nullptr, nullptr, kRelativeExtent, window, false};
    const cuda_core::PreparedBf16LaneImages images{rows.queries, rows.keys, rows.values, rows.relative_logits,
                                                  rows.sinks, rows.output};
    return cuda_core::append_attention_prepared_bf16_step_lanes(dims, shape, images, lanes, stream,
                                                                cuda_core::FlashPipe::pad2, failure);
}

inline cudaError_t enqueue_mini_cuda_attention(
    LayerVisit visit, const MiniCudaAttentionRows& rows, const MiniSessionStatePlan& session,
    MiniSessionKvPlanes planes, cudaStream_t stream, cuda_core::FlashPipe pipe = cuda_core::FlashPipe::pad2) {
    if (!session.admits(rows.first_position, rows.count)) return cudaErrorInvalidValue;
    return enqueue_mini_cuda_attention(visit, rows, mini_visit_kv_cache(session, visit, planes), stream, pipe);
}

}  // namespace strix::inkling2
