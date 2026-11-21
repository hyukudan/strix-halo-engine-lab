#pragma once

#include "attn_core/prepared_gqa_cuda.hpp"
#include "inkling2_mini_attention_prepare.hpp"
#include "inkling2_mini_state.hpp"

namespace strix::inkling2 {

using MiniCudaAttentionRows = attn_core::PreparedGqaAttentionRows;

inline cudaError_t enqueue_mini_cuda_attention(
    LayerVisit visit, const MiniCudaAttentionRows& rows, MiniVisitKvCache cache,
    cudaStream_t stream, cuda_core::FlashPipe pipe = cuda_core::FlashPipe::pad2, unsigned* failure = nullptr) {
    if (!valid_mini_visit(visit)) return cudaErrorInvalidValue;
    const unsigned window = is_full_attention(visit.physical_layer) ? 0U : MiniPreparedGqaShape::kSlidingWindow;
    return attn_core::enqueue_prepared_gqa_cuda<MiniPreparedGqaShape>(
        window, rows, {cache.keys, cache.values, cache.slots, cache.wrapped}, stream, pipe, failure);
}

inline cudaError_t enqueue_mini_cuda_attention(
    LayerVisit visit, const MiniCudaAttentionRows& rows, const MiniSessionStatePlan& session,
    MiniSessionKvPlanes planes, cudaStream_t stream, cuda_core::FlashPipe pipe = cuda_core::FlashPipe::pad2) {
    if (!session.admits(rows.first_position, rows.count)) return cudaErrorInvalidValue;
    return enqueue_mini_cuda_attention(visit, rows, mini_visit_kv_cache(session, visit, planes), stream, pipe);
}

}  // namespace strix::inkling2
