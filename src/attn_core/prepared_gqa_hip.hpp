#pragma once

#include "attn_core/cache_rows.hpp"
#include "attn_core/prepared_bf16_attention.hpp"
#include "attn_core/prepared_finite_device.hpp"
#include "attn_core/prepared_gqa_attention.hpp"

namespace strix::attn_core {

struct PreparedGqaHipCache {
    std::uint16_t* keys;    // [slot][KV head][dim] BF16 for this visit
    std::uint16_t* values;  // [slot][KV head][dim] BF16 for this visit
    const std::uint16_t* zero_padding;  // One zero BF16 head row for tile loads.
    unsigned slots;
    bool wrapped;
};

namespace detail {
template <class Shape>
struct PreparedGqaHipPolicy
    : dense::PreparedBf16Policy<Shape::kHeadWidth, Shape::kQueryHeads / Shape::kKvHeads, true, true> {
    static_assert(Shape::kQueryHeads % Shape::kKvHeads == 0U);
    static constexpr float kScaleLog2 = Shape::kScoreScale * 1.4426950408889634F;
};

template <class Shape>
inline dense::PreparedBf16Args prepared_gqa_hip_args(
    unsigned window, const PreparedGqaAttentionRows& rows, const PreparedGqaHipCache& cache, unsigned* failure) {
    return {rows.queries, rows.output, cache.keys, cache.values, cache.zero_padding, failure,
            rows.count, Shape::kKvHeads, rows.first_position, window, cache.slots, cache.wrapped,
            {rows.relative_logits, Shape::kQueryHeads, Shape::kRelativeExtent}, {rows.sinks, Shape::kQueryHeads}};
}
}  // namespace detail

template <class Shape>
inline bool can_enqueue_prepared_gqa_hip(
    unsigned window, const PreparedGqaAttentionRows& rows, const PreparedGqaHipCache& cache,
    unsigned* failure, std::size_t workspace_size, unsigned splits) {
    if (rows.first_position >= Shape::kMaxContext ||
        rows.count > Shape::kMaxContext - rows.first_position || !rows.keys || !rows.values) return false;
    return dense::can_launch_prepared_bf16<detail::PreparedGqaHipPolicy<Shape>>(
        detail::prepared_gqa_hip_args<Shape>(window, rows, cache, failure), workspace_size, splits);
}

// The caller owns the prepared buffers and the committed KV prefix for this
// logical visit. Keep them alive and disjoint through stream completion. Q, K/V
// cache, zero padding and split workspace follow the 16-byte alignment contract.
// On success this enqueues both cache appends before attention on the same stream.
// If an enqueue or device operation fails, the caller discards or restores the visit.
template <class Shape, bool ProducersCheckBiasAndSinks = false>
inline hipError_t enqueue_prepared_gqa_hip(unsigned window, const PreparedGqaAttentionRows& rows,
                                           const PreparedGqaHipCache& cache, unsigned* failure,
                                             hipStream_t stream, std::span<float> workspace = {},
                                             unsigned splits = 1) {
    namespace dense = attn_core::dense;
    using Policy = detail::PreparedGqaHipPolicy<Shape>;
    if (!can_enqueue_prepared_gqa_hip<Shape>(window, rows, cache, failure, workspace.size(), splits)) {
        return hipErrorInvalidValue;
    }
    const auto attention = detail::prepared_gqa_hip_args<Shape>(window, rows, cache, failure);

    const attn::cache_rows::Rows layout{Shape::kKvWidth, cache.wrapped ? cache.slots : 0U};
    const auto status = attn_core::cache_rows::store_pair(
        {rows.keys, Shape::kKvWidth, nullptr, rows.first_position, rows.count, layout, cache.keys, failure},
        {rows.values, Shape::kKvWidth, nullptr, rows.first_position, rows.count, layout, cache.values, failure}, stream);
    if (status != hipSuccess) return status;
    const auto launched = dense::launch_prepared_bf16<Policy>(attention, stream, workspace, splits);
    if (launched != hipSuccess) return launched;
    // Cache stores check the appended K/V, and Flash checks BF16 Q and its
    // output (including Split-K combine). Only checked producers may omit
    // the separate scan of relative logits and sinks, including masked entries.
    if constexpr (!ProducersCheckBiasAndSinks) {
        enqueue_validate_prepared_finite(
            {rows.queries, rows.keys, rows.values, rows.relative_logits, rows.sinks, rows.output,
             rows.count, Shape::kQueryHeads, Shape::kKvHeads, Shape::kHeadWidth, Shape::kRelativeExtent, failure}, stream);
        return hipGetLastError();
    }
    return hipSuccess;
}

}  // namespace strix::attn_core
