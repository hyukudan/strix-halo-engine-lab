#pragma once

#include "attn_core/prepared_gqa_hip.hpp"
#include "inkling2_mini_attention_prepare.hpp"
#include "inkling2_mini_state.hpp"

namespace strix::inkling2 {

using MiniHipAttentionRows = attn_core::PreparedGqaAttentionRows;
using MiniHipAttentionCache = attn_core::PreparedGqaHipCache;

inline unsigned mini_prepared_gqa_window(LayerVisit visit) {
    return is_full_attention(visit.physical_layer) ? 0U : MiniPreparedGqaShape::kSlidingWindow;
}

inline bool can_enqueue_mini_hip_attention(
    LayerVisit visit, const MiniHipAttentionRows& rows, const MiniHipAttentionCache& cache,
    unsigned* failure, std::size_t workspace_size, unsigned splits) {
    return valid_mini_visit(visit) && attn_core::can_enqueue_prepared_gqa_hip<MiniPreparedGqaShape>(
        mini_prepared_gqa_window(visit), rows, cache, failure, workspace_size, splits);
}

inline hipError_t enqueue_mini_hip_attention(
    LayerVisit visit, const MiniHipAttentionRows& rows, const MiniHipAttentionCache& cache, unsigned* failure,
    hipStream_t stream, std::span<float> workspace = {}, unsigned splits = 1) {
    if (!valid_mini_visit(visit)) return hipErrorInvalidValue;
    return attn_core::enqueue_prepared_gqa_hip<MiniPreparedGqaShape>(
        mini_prepared_gqa_window(visit), rows, cache, failure, stream, workspace, splits);
}

inline hipError_t enqueue_mini_hip_attention(
    LayerVisit visit, const MiniHipAttentionRows& rows, const MiniSessionStatePlan& session,
    MiniSessionKvPlanes planes, const std::uint16_t* zero_padding, unsigned* failure,
    hipStream_t stream, std::span<float> workspace = {}, unsigned splits = 1) {
    if (!session.admits(rows.first_position, rows.count)) return hipErrorInvalidValue;
    const auto cache = mini_visit_kv_cache(session, visit, planes);
    if (cache.slots == 0U) return hipErrorInvalidValue;
    return enqueue_mini_hip_attention(visit, rows,
        {cache.keys, cache.values, zero_padding, cache.slots, cache.wrapped}, failure, stream, workspace, splits);
}

}  // namespace strix::inkling2
