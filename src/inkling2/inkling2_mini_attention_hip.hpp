#pragma once

#include "attn_core/cache_rows.hpp"
#include "attn_core/prepared_bf16_attention.hpp"
#include "inkling2_mini_config.hpp"
#include "inkling2_mini_state.hpp"

#include <cstdint>
#include <span>

namespace strix::inkling2 {

struct MiniHipAttentionRows {
    const float* queries;          // [row][query head][dim] FP32
    const float* keys;             // [row][KV head][dim] FP32
    const float* values;           // [row][KV head][dim] FP32
    const float* relative_logits;  // [row][query head][backward distance] FP32, natural units
    const float* sinks;            // [row][query head] FP32, natural units
    float* output;                 // [row][query head][dim] FP32
    unsigned first_position;
    unsigned count;
};

struct MiniHipAttentionCache {
    std::uint16_t* keys;    // [slot][KV head][dim] BF16 for this visit
    std::uint16_t* values;  // [slot][KV head][dim] BF16 for this visit
    const std::uint16_t* zero_padding;  // One zero BF16 head row for tile loads.
    unsigned slots;
    bool wrapped;
};

namespace detail {
struct MiniPreparedHipAttention : attn_core::dense::PreparedBf16Policy<kHeadWidth, kQueryHeads / kKvHeads, true, true> {
    static_assert(kHeadWidth == 128U && kQueryHeads % kKvHeads == 0U);
    static constexpr float kScaleLog2 = 0.08838834764831844F * 1.4426950408889634F;
};

inline attn_core::dense::PreparedBf16Args mini_hip_prepared_args(
    LayerVisit visit, const MiniHipAttentionRows& rows, const MiniHipAttentionCache& cache, unsigned* failure) {
    const unsigned window = is_full_attention(visit.physical_layer) ? 0U : kSlidingWindow;
    return {rows.queries, rows.output, cache.keys, cache.values, cache.zero_padding, failure,
            rows.count, kKvHeads, rows.first_position, window, cache.slots, cache.wrapped,
            {rows.relative_logits, kQueryHeads, kRelativeExtent}, {rows.sinks, kQueryHeads}};
}
}  // namespace detail

inline bool can_enqueue_mini_hip_attention(
    LayerVisit visit, const MiniHipAttentionRows& rows, const MiniHipAttentionCache& cache,
    unsigned* failure, std::size_t workspace_size, unsigned splits) {
    if (!valid_mini_visit(visit) || rows.first_position >= kMaxContext ||
        rows.count > kMaxContext - rows.first_position || !rows.keys || !rows.values) return false;
    return attn_core::dense::can_launch_prepared_bf16<detail::MiniPreparedHipAttention>(
        detail::mini_hip_prepared_args(visit, rows, cache, failure), workspace_size, splits);
}

// The caller owns the prepared buffers and the committed KV prefix for this
// logical visit. Keep them alive and disjoint through stream completion. Q, K/V
// cache, zero padding and split workspace follow the 16-byte alignment contract.
// On success this enqueues both cache appends before attention on the same stream.
// If an enqueue or device operation fails, the caller discards or restores the visit.
inline hipError_t enqueue_mini_hip_attention(LayerVisit visit, const MiniHipAttentionRows& rows,
                                             const MiniHipAttentionCache& cache, unsigned* failure,
                                             hipStream_t stream, std::span<float> workspace = {},
                                             unsigned splits = 1) {
    namespace dense = attn_core::dense;
    using Policy = detail::MiniPreparedHipAttention;
    if (!can_enqueue_mini_hip_attention(visit, rows, cache, failure, workspace.size(), splits)) {
        return hipErrorInvalidValue;
    }
    const auto attention = detail::mini_hip_prepared_args(visit, rows, cache, failure);

    const attn::cache_rows::Rows layout{kKvWidth, cache.wrapped ? cache.slots : 0U};
    const auto status = attn_core::cache_rows::store_pair(
        {rows.keys, kKvWidth, nullptr, rows.first_position, rows.count, layout, cache.keys, failure},
        {rows.values, kKvWidth, nullptr, rows.first_position, rows.count, layout, cache.values, failure}, stream);
    if (status != hipSuccess) return status;
    return dense::launch_prepared_bf16<Policy>(attention, stream, workspace, splits);
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
