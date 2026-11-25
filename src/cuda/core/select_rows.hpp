#pragma once

#include "attn_core/select_spec.hpp"
#include "strix/exec/descriptor.hpp"

#if defined(__CUDACC__)
#define STRIX_SELECT_ROW_HD __host__ __device__
#else
#define STRIX_SELECT_ROW_HD
#endif

namespace strix::cuda::core::select {

struct RowStrides {
    std::size_t query, scores, selected;  // In elements, regardless of type.
};

struct RowBases {
    const float* query;
    const float* keys;  // Shared complete-key prefix, each key 128 floats.
    float* scores;
    std::uint32_t* selected;
};

namespace detail {

struct VisibleRow {
    RowBases row;
    std::uint64_t position;  // Zero-based, including this row in the visible prefix.

    STRIX_SELECT_ROW_HD constexpr ScoreRow score(unsigned ratio) const {
        ScoreRow result;
        result.query = row.query;
        result.keys = row.keys;
        result.visible = static_cast<unsigned>((position + 1U) / ratio);
        result.scores = row.scores;
        return result;
    }

    STRIX_SELECT_ROW_HD constexpr PickRow pick(unsigned ratio) const {
        PickRow result;
        result.groups = static_cast<unsigned>((position + 1U) / ratio);
        result.tail = static_cast<unsigned>((position + 1U) % ratio);
        result.scores = result.mask_scores = row.scores;
        result.selected = row.selected;
        return result;
    }
};

STRIX_SELECT_ROW_HD constexpr RowBases offset_rows(RowBases base, RowStrides stride, std::size_t row) {
    if (base.query) base.query += row * stride.query;
    if (base.scores) base.scores += row * stride.scores;
    if (base.selected) base.selected += row * stride.selected;
    return base;
}

} // namespace detail

// The caller owns the buffer extents and orders their lifetime on the stream.
// Writable rows must be disjoint. Positions and their visible lengths fit in
// uint32_t. The scorer writes only complete groups above the top-k threshold.
struct AffineRows {
    RowBases base;
    RowStrides stride;
    std::uint32_t first_position;

    STRIX_SELECT_ROW_HD constexpr bool load(unsigned index, detail::VisibleRow& out) const {
        out = {detail::offset_rows(base, stride, index), std::uint64_t{first_position} + index};
        return true;
    }
};

struct SlotRowBase {
    RowBases base;
    std::uint32_t first_position;  // Position corresponding to each slot's scratch row zero.
};

// A valid X3 row selects a slot and a position-relative offset in that slot's
// scratch buffers. Each live (slot, position) must map to a distinct writable
// row. The descriptor and slot bases are read from device memory at replay.
struct SlotRows {
    const exec::StepBlock* step;
    const SlotRowBase* slots;
    unsigned slot_count;
    RowStrides stride;

    STRIX_SELECT_ROW_HD constexpr bool load(unsigned index, detail::VisibleRow& out) const {
        if (!exec::step_row_valid(*step, index)) return false;
        const auto& row = exec::step_row(*step, index);
        if (row.slot >= slot_count) return false;
        const auto& slot = slots[row.slot];
        if (row.position < slot.first_position) return false;
        out = {detail::offset_rows(slot.base, stride, row.position - slot.first_position), row.position};
        return true;
    }
};

} // namespace strix::cuda::core::select

#undef STRIX_SELECT_ROW_HD
