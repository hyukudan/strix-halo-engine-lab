#pragma once

// Decode lanes of a multi-request step for the prepared BF16 flash (prepared_bf16_attention.hpp,
// append_prepared_bf16_lanes). Backend-neutral planning: no device headers.
//
// A lane is one query row of its own sequence: its own K/V cache, its own absolute position. The packed
// images (queries, keys, values, outputs, relative logits, sinks) hold lane l at row l. One launch set (store, flash,
// combine) serves up to kMaxPreparedBf16Lanes lanes; each work-group binds its lane and runs the single-row body.
//
// Batch invariance (EXACT): a lane's split count is the single-row rule of its OWN position
// (small_row_key_splits(position, 1, window)), its key-tile partition is that split's share, its partials live at the
// same offsets relative to its own workspace base, and the merge is the ascending combine. None of these depends on
// the number of lanes or on another lane, so a lane's bits equal the one-lane append_prepared_bf16 bits.

#include "dense_flash_keysplit.hpp"
#include "strix/attn/prepared_lanes.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::attn_core::dense {

struct PreparedBf16Lane {
    std::uint16_t* keys;    // [slot][KV head][dim] BF16, this lane's cache
    std::uint16_t* values;  // [slot][KV head][dim] BF16, this lane's cache
    unsigned first_position;
    unsigned cache_slots;
    unsigned wrapped;       // nonzero: slot = absolute position % cache_slots
};

inline constexpr unsigned kMaxPreparedBf16Lanes = 16U;

// Workspace floats one lane owns: one row's partials and statistics at the largest split count.
constexpr std::size_t lane_workspace_floats(unsigned heads, unsigned dim) {
    return workspace_floats(1U, heads, dim, kKeySplitMaxChunks);
}

// Lanes one launch can serve from a Split-K workspace of `workspace` floats (0: the workspace holds no lane).
constexpr unsigned lanes_per_launch(std::size_t workspace, unsigned heads, unsigned dim) {
    const std::size_t per = lane_workspace_floats(heads, dim);
    if (per == 0U) return 0U;
    const std::size_t fit = workspace / per;
    return fit < kMaxPreparedBf16Lanes ? static_cast<unsigned>(fit) : kMaxPreparedBf16Lanes;
}

// The split count of a lane: the single-row rule of its own position, never of the cohort width.
#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
inline unsigned lane_key_splits(unsigned position, unsigned window) {
    return small_row_key_splits(position, 1U, window);
}

// Grid depth of one launch: the largest lane split count (lanes above their own count exit).
inline unsigned max_lane_key_splits(std::span<const PreparedBf16Lane> lanes, unsigned window) {
    unsigned splits = 1U;
    for (const auto& lane : lanes) {
        const unsigned s = lane_key_splits(lane.first_position, window);
        splits = s > splits ? s : splits;
    }
    return splits;
}

// Every lane writes its own caches: two lanes on one cache would see each other's store before their flash, which the
// one-lane sequence orders differently.
inline bool distinct_lane_caches(std::span<const PreparedBf16Lane> lanes) {
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        if (!lanes[i].keys || !lanes[i].values || lanes[i].keys == lanes[i].values) return false;
        for (std::size_t j = 0; j < i; ++j) {
            const auto* k = lanes[j].keys;
            const auto* v = lanes[j].values;
            if (lanes[i].keys == k || lanes[i].keys == v || lanes[i].values == k || lanes[i].values == v) return false;
        }
    }
    return true;
}

// Lanes of one sequence on one cache (verify windows): the one backend-neutral rule of the lane contract
// (strix/attn/prepared_lanes.hpp sequence_lane_caches), which the CUDA lanes admission binds as well.
inline bool sequence_lane_caches(std::span<const PreparedBf16Lane> lanes, unsigned window) {
    return ::strix::attn::lanes::sequence_lane_caches<PreparedBf16Lane>(lanes, window);
}

// Lane `index` of the packed images in a one-lane argument block (PreparedBf16Args or a struct with its fields):
// query/output rows at index x heads x dim, relative logits at index x heads x extent, sinks at index x heads.
template <class Args>
#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
inline Args prepared_bf16_lane_args(const Args& base, const PreparedBf16Lane& lane, unsigned index, unsigned heads,
                                    unsigned dim) {
    Args a = base;
    const std::size_t row = std::size_t(index) * heads * dim;
    a.queries = base.queries + row;
    a.output = base.output + row;
    a.keys = lane.keys;
    a.values = lane.values;
    a.rows = 1U;
    a.first_position = lane.first_position;
    a.cache_slots = lane.cache_slots;
    a.wrapped = lane.wrapped != 0U;
    if (base.relative_bias.logits)
        a.relative_bias.logits = base.relative_bias.logits +
                                 std::size_t(index) * base.relative_bias.heads * base.relative_bias.extent;
    if (base.sinks.logits) a.sinks.logits = base.sinks.logits + std::size_t(index) * base.sinks.heads;
    return a;
}

}  // namespace strix::attn_core::dense
