#pragma once

#include "cuda/core/select_rows.hpp"
#include "cuda/core/select_score.cuh"
#include "cuda/core/select_relu_schedules.cuh"

#include <concepts>

namespace strix::cuda::core::select {
namespace detail {

template <class Rows>
concept IndexedRows = std::same_as<Rows, AffineRows> || std::same_as<Rows, SlotRows>;

template <IndexedRows Rows>
__global__ void score_rows_kernel(Rows rows, ScoreSpec spec) {
    VisibleRow input;
    if (rows.load(blockIdx.y, input)) score_row<ScoreForm::relu_tree>(input.score(spec.block_width), spec);
}

template <class Rows>
__global__ __launch_bounds__(pick_threads) void pick_rows_kernel(Rows rows, PickSpec spec) {
    __shared__ PickScratch scratch;
    VisibleRow input;
    if (!rows.load(blockIdx.x, input)) return;
    const auto row = input.pick(spec.expansion);
    constexpr auto Order = KeyOrder::eligible_float;
#include "cuda/core/select_pick_body.inc"
}

} // namespace detail

// block_width is the pool ratio. max_groups bounds the complete visible key
// prefix of every live row. Pass the same ratio and top_k to the picker.
template <ScoreForm Form, detail::IndexedRows Rows>
    requires (Form == ScoreForm::relu_tree)
cudaError_t enqueue_scores(Rows rows, unsigned row_count, unsigned max_groups, ScoreSpec spec, cudaStream_t stream) {
    if (row_count == 0 || max_groups <= spec.top_k) return cudaSuccess;
    if (row_count > 65535U || spec.dim != 128 || spec.heads == 0 || spec.block_width == 0)
        return cudaErrorInvalidValue;
    if constexpr (std::same_as<Rows, AffineRows>) {
        if (!rows.base.query || !rows.base.keys || !rows.base.scores) return cudaErrorInvalidValue;
    } else {
        if (!rows.step || !rows.slots || row_count > exec::kStepMaxRows) return cudaErrorInvalidValue;
    }
    constexpr unsigned threads = detail::score_threads;
    detail::score_rows_kernel<<<dim3(1U + (max_groups - 1U) / threads, row_count), threads, 0, stream>>>(rows, spec);
    return cudaGetLastError();
}

// The eligible picker aliases its score row when masking. A row at or below
// top_k reads and writes no scores, so its score base may be null.
template <detail::IndexedRows Rows>
cudaError_t enqueue_pick(Rows rows, unsigned row_count, PickSpec spec, cudaStream_t stream) {
    if (row_count == 0) return cudaSuccess;
    if (spec.order != KeyOrder::eligible_float || spec.force_newest || spec.top_k == 0 || spec.expansion == 0)
        return cudaErrorInvalidValue;
    if constexpr (std::same_as<Rows, AffineRows>) {
        if (row_count > 2147483647U || !rows.base.selected ||
            (!rows.base.scores && (std::uint64_t{rows.first_position} + row_count) / spec.expansion > spec.top_k))
            return cudaErrorInvalidValue;
    } else {
        if (!rows.step || !rows.slots || row_count > exec::kStepMaxRows) return cudaErrorInvalidValue;
    }
    detail::pick_rows_kernel<<<row_count, detail::pick_threads, 0, stream>>>(rows, spec);
    return cudaGetLastError();
}

// Segment descriptors are packed before the eager launch or each replay.
// Inactive capture rows are skipped by the same producer used for scoring.
// Use the packed shape's row bound and the scorer's ratio and top_k.
template <class Map>
cudaError_t enqueue_pick(SegmentRows<Map> rows, unsigned row_count, PickSpec spec, cudaStream_t stream) {
    if (row_count == 0) return cudaSuccess;
    if (row_count > 2147483647U || spec.order != KeyOrder::eligible_float || spec.force_newest ||
        spec.top_k == 0 || spec.expansion == 0 || !rows.base.selected) return cudaErrorInvalidValue;
    if constexpr (rows.replay) {
        if (!rows.map || !rows.base.scores) return cudaErrorInvalidValue;
    } else {
        if (rows.map.count > rows.capacity) return cudaErrorInvalidValue;
        if (rows.map.count == 0) return cudaSuccess;
        if (!rows.base.scores && rows.map.segments[rows.map.count - 1U].wave_end != 0) return cudaErrorInvalidValue;
    }
    detail::pick_rows_kernel<<<row_count, detail::pick_threads, 0, stream>>>(rows, spec);
    return cudaGetLastError();
}

} // namespace strix::cuda::core::select
