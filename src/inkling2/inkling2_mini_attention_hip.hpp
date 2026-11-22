#pragma once

#include "attn_core/dense_flash_keysplit.hpp"
#include "attn_core/prepared_bf16_attention.hpp"
#include "inkling2_mini_attention_prepare.hpp"
#include "inkling2_mini_state.hpp"

namespace strix::inkling2 {

using MiniHipAttentionRows = attn_core::PreparedGqaAttentionRows;
using MiniHipAttentionCache = attn_core::dense::PreparedBf16Cache;

namespace detail {

struct MiniHipAttentionPolicy
    : attn_core::dense::PreparedBf16Policy<kHeadWidth, kQueryHeads / kKvHeads, true, true> {
    static_assert(kQueryHeads % kKvHeads == 0U);
    static constexpr float kScaleLog2 = MiniPreparedGqaShape::kScoreScale * 1.4426950408889634F;
};

// Prefill chunks (more rows than the key-split bound, one split): 64 positions per work-group, so each staged key and
// value tile serves 4x the query rows. Every row still visits the same key tiles in the same order (each wave skips
// the tiles outside its own 16-row range), so the outputs are the bits of the 16-position tile.
struct MiniHipAttentionPrefillPolicy
    : attn_core::dense::PreparedBf16Policy<kHeadWidth, kQueryHeads / kKvHeads, true, true, 1U, 64U> {
    static constexpr float kScaleLog2 = MiniHipAttentionPolicy::kScaleLog2;
};

// The prefill form with the relative bias projected inside the flash from the clipped features and the BF16
// projection (no [row][head][distance] image: 805 MB written and read per layer at 8K rows). Same logits and failure
// bit as the image (dense::DenseProjectedBias), so the outputs are the bits of the prefill policy above.
struct MiniHipAttentionProjectedPrefillPolicy
    : attn_core::dense::PreparedBf16ProjectedBias<
          attn_core::dense::PreparedBf16Policy<kHeadWidth, kQueryHeads / kKvHeads, false, true, 1U, 64U>,
          kRelativeWidth, kRelativeExtent> {
    static constexpr float kScaleLog2 = MiniHipAttentionPolicy::kScaleLog2;
};

inline bool mini_hip_prefill_rows(unsigned rows, unsigned splits) {
    return rows > attn_core::dense::kKeySplitMaxRows && splits <= 1U;
}

}  // namespace detail

inline unsigned mini_prepared_gqa_window(LayerVisit visit) {
    return is_full_attention(visit.physical_layer) ? 0U : MiniPreparedGqaShape::kSlidingWindow;
}

inline bool valid_mini_hip_attention_rows(LayerVisit visit, const MiniHipAttentionRows& rows) {
    return valid_mini_visit(visit) && rows.first_position < kMaxContext &&
           rows.count <= kMaxContext - rows.first_position && rows.keys && rows.values;
}

inline bool can_enqueue_mini_hip_attention(
    LayerVisit visit, const MiniHipAttentionRows& rows, const MiniHipAttentionCache& cache,
    unsigned* failure, std::size_t workspace_size, unsigned splits) {
    if (!valid_mini_hip_attention_rows(visit, rows)) return false;
    namespace dense = attn_core::dense;
    using Policy = detail::MiniHipAttentionPolicy;
    const auto attention = dense::bind_prepared_bf16<Policy>(
        kKvHeads, mini_prepared_gqa_window(visit), kRelativeExtent, rows, cache, failure);
    return dense::can_launch_prepared_bf16<Policy>(attention, workspace_size, splits);
}

template <bool ProducersCheckBiasAndSinks = false>
inline hipError_t enqueue_mini_hip_attention(
    LayerVisit visit, const MiniHipAttentionRows& rows, const MiniHipAttentionCache& cache, unsigned* failure,
    hipStream_t stream, std::span<float> workspace = {}, unsigned splits = 1,
    const std::uint32_t* device_position = nullptr, attn_core::RelativeBiasProjection projection = {}) {
    if (!valid_mini_hip_attention_rows(visit, rows)) return hipErrorInvalidValue;
    if (projection.features) {  // prefill rows only: the caller skipped the image
        if (!detail::mini_hip_prefill_rows(rows.count, splits) || device_position) return hipErrorInvalidValue;
        return attn_core::dense::append_prepared_bf16<detail::MiniHipAttentionProjectedPrefillPolicy,
                                                      ProducersCheckBiasAndSinks>(
            kKvHeads, mini_prepared_gqa_window(visit), kRelativeExtent, rows, cache, failure, stream, workspace, splits,
            device_position, projection);
    }
    if (detail::mini_hip_prefill_rows(rows.count, splits))
        return attn_core::dense::append_prepared_bf16<detail::MiniHipAttentionPrefillPolicy, ProducersCheckBiasAndSinks>(
            kKvHeads, mini_prepared_gqa_window(visit), kRelativeExtent, rows, cache, failure, stream, workspace, splits,
            device_position);
    return attn_core::dense::append_prepared_bf16<detail::MiniHipAttentionPolicy, ProducersCheckBiasAndSinks>(
        kKvHeads, mini_prepared_gqa_window(visit), kRelativeExtent, rows, cache, failure, stream, workspace, splits,
        device_position);
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
