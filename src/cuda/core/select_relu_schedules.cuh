#pragma once

#include "cuda/core/select_score.cuh"
#include "cuda/core/select_segments.hpp"

#include <concepts>

namespace strix::cuda::core::select {
namespace detail {

// All rows of a tile share a key prefix. The producer resolves actual
// workspace rows and positions, including holes in an X3 slot.
struct AffineScoreSpan {
    RowBases base;
    RowStrides stride;
    unsigned count, groups;
    std::uint64_t first_position;

    __device__ __forceinline__ bool load(unsigned i, VisibleRow& out) const {
        if (i >= count) return false;
        out = {offset_rows(base, stride, i), first_position + i};
        return true;
    }

    __device__ __forceinline__ const float* keys() const { return base.keys; }
};

struct SlotScoreSpan {
    SlotRows rows;
    const float* key_base;
    unsigned mask, count, groups;

    __device__ __forceinline__ bool load(unsigned i, VisibleRow& out) const {
        return i < count && ((mask >> i) & 1U) && rows.load(i, out);
    }

    __device__ __forceinline__ const float* keys() const { return key_base; }
};

__device__ __forceinline__ bool slot_score_span(SlotRows rows, unsigned count, unsigned anchor,
                                               unsigned max_groups, ScoreSpec spec, SlotScoreSpan& span) {
    VisibleRow first;
    if (!rows.load(anchor, first)) return false;
    const auto slot = exec::step_row(*rows.step, anchor).slot;
    const unsigned live = min(count, rows.step->h.rows);
    for (unsigned i = 0; i < anchor; ++i) {
        if (exec::step_row_valid(*rows.step, i) && exec::step_row(*rows.step, i).slot == slot &&
            exec::step_row(*rows.step, i).position >= rows.slots[slot].first_position) return false;
    }
    unsigned mask = 0;
    std::uint64_t groups = 0;
    for (unsigned i = anchor; i < live; ++i) {
        VisibleRow input;
        if (!rows.load(i, input) || exec::step_row(*rows.step, i).slot != slot) continue;
        mask |= 1U << i;
        const auto visible = (input.position + 1U) / spec.block_width;
        if (visible > groups) groups = visible;
    }
    if (groups <= spec.top_k || groups > max_groups) return false;
    span = {rows, first.row.keys, mask, live, static_cast<unsigned>(groups)};
    return true;
}

__device__ __forceinline__ unsigned relu_limit(const VisibleRow& row, ScoreSpec spec) {
    const auto complete = static_cast<unsigned>((row.position + 1U) / spec.block_width);
    return complete > spec.top_k ? complete : 0U;
}

inline constexpr unsigned full_warp = 0xffffffffU;

// Both schedules use the relu_tree pairing over strided groups of four
// products, followed by lane distances 16, 8, 4, 2, 1 and an ordered head sum.
__device__ __forceinline__ float relu_partial(float q0, float q1, float q2, float q3,
                                             float k0, float k1, float k2, float k3) {
    return __fadd_rn(__fadd_rn(__fmul_rn(q0, k0), __fmul_rn(q2, k2)),
                     __fadd_rn(__fmul_rn(q1, k1), __fmul_rn(q3, k3)));
}

template <unsigned Heads, unsigned Step = 0>
__device__ __forceinline__ void scatter_relu_heads(float (&v)[Heads], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned distance = 16U >> Step;
        constexpr unsigned active = Heads >> Step;
        if constexpr (active >= 2U) {
            constexpr unsigned half = active / 2U;
            const bool upper = (lane & distance) != 0;
#pragma unroll
            for (unsigned h = 0; h < half; ++h) {
                const float low = __shfl_xor_sync(full_warp, v[h], distance);
                const float high = __shfl_xor_sync(full_warp, v[h + half], distance);
                v[h] = upper ? __fadd_rn(v[h + half], high) : __fadd_rn(v[h], low);
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(full_warp, v[0], distance));
        }
        scatter_relu_heads<Heads, Step + 1U>(v, lane);
    }
}

template <class Span>
__device__ __forceinline__ void shuffle_relu_tile(Span span, unsigned wave, ScoreSpec spec) {
    const unsigned lane = threadIdx.x & 31U;
    const unsigned chunks = score_tiles(span.groups, shuffle_groups);
    const unsigned row_tile = wave / chunks;
    if (row_tile >= score_tiles(span.count, shuffle_rows)) return;
    const unsigned r0 = row_tile * shuffle_rows;
    const unsigned base = wave % chunks * shuffle_groups;
    float q[shuffle_rows][relu_heads][4];
    float* scores[shuffle_rows];
    unsigned limits[shuffle_rows];
    bool any = false;
#pragma unroll
    for (unsigned rr = 0; rr < shuffle_rows; ++rr) {
        VisibleRow row;
        const bool live = span.load(r0 + rr, row);
        limits[rr] = live ? relu_limit(row, spec) : 0U;
        scores[rr] = live ? row.row.scores : nullptr;
        any = any || base < limits[rr];
#pragma unroll
        for (unsigned h = 0; h < relu_heads; ++h) {
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j)
                q[rr][h][j] = live ? row.row.query[h * relu_dim + lane + 32U * j] : 0.0F;
        }
    }
    if (!any) return;
    float result[shuffle_rows]{};
    const unsigned last = min(shuffle_groups, span.groups - base);
    for (unsigned i = 0; i < last; ++i) {
        const auto* k = span.keys() + std::size_t{base + i} * relu_dim + lane;
        const float k0 = k[0], k1 = k[32], k2 = k[64], k3 = k[96];
#pragma unroll
        for (unsigned rr = 0; rr < shuffle_rows; ++rr) {
            float partial[relu_heads];
#pragma unroll
            for (unsigned h = 0; h < relu_heads; ++h)
                partial[h] = relu_partial(q[rr][h][0], q[rr][h][1], q[rr][h][2], q[rr][h][3], k0, k1, k2, k3);
            scatter_relu_heads(partial, lane);
            float total = 0.0F;
#pragma unroll
            for (unsigned h = 0; h < relu_heads; ++h)
                total = __fadd_rn(total, fmaxf(__shfl_sync(full_warp, partial[0], static_cast<int>(8U * h)), 0.0F));
            if (lane == i) result[rr] = __fdiv_rn(total, __fsqrt_rn(static_cast<float>(relu_dim)));
        }
    }
#pragma unroll
    for (unsigned rr = 0; rr < shuffle_rows; ++rr)
        if (lane < last && base + lane < limits[rr]) scores[rr][base + lane] = result[rr];
}

__device__ __forceinline__ bool aligned_score_vector(const float* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 15U) == 0U;
}

__device__ __forceinline__ float4 load_score_vector(const float* ptr, bool aligned) {
    if (aligned) return *reinterpret_cast<const float4*>(ptr);
    return make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

template <class Span>
__device__ __forceinline__ void tree_relu_tile(Span span, unsigned tile_x, unsigned tile_y, ScoreSpec spec) {
    constexpr unsigned key_tile = 32, threads = tree_rows * relu_heads;
    __shared__ __align__(16) float sk[key_tile * relu_dim];
    __shared__ float ss[tree_rows][key_tile + 1U];
    const unsigned tid = threadIdx.x, lane = tid & 31U, rr = tid / relu_heads, h = tid % relu_heads;
    const unsigned r0 = tile_x * tree_rows;
    const unsigned s0 = tile_y * tree_groups;
    if (r0 >= span.count || s0 >= span.groups) return; // Uniform before the first barrier.
    const unsigned s1 = s0 + min(tree_groups, span.groups - s0);
    if constexpr (std::same_as<Span, AffineScoreSpan>) {
        VisibleRow last;
        span.load(r0 + min(tree_rows, span.count - r0) - 1U, last);
        if (s0 >= relu_limit(last, spec)) return;
    }

    VisibleRow row;
    const bool live = span.load(r0 + rr, row);
    const bool query_aligned = live && aligned_score_vector(row.row.query);
    const bool key_aligned = aligned_score_vector(span.keys());
    float q[relu_dim];
#pragma unroll
    for (unsigned d = 0; d < relu_dim; d += 4U) {
        const float4 v = live ? load_score_vector(row.row.query + h * relu_dim + d, query_aligned)
                              : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q[d] = v.x; q[d + 1U] = v.y; q[d + 2U] = v.z; q[d + 3U] = v.w;
    }
    for (unsigned b0 = s0; b0 < s1; b0 += key_tile) {
        const unsigned nb = min(key_tile, s1 - b0);
        __syncthreads();
        for (unsigned e = tid; e < key_tile * relu_dim / 4U; e += threads) {
            const unsigned block = e / (relu_dim / 4U);
            reinterpret_cast<float4*>(sk)[e] = block < nb
                ? load_score_vector(span.keys() + std::size_t{b0} * relu_dim + 4U * e, key_aligned)
                : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
        __syncthreads();
        for (unsigned i = 0; i < nb; ++i) {
            const float* k = sk + i * relu_dim;
            float a[16];
#pragma unroll
            for (unsigned g = 0; g < 4U; ++g) {
                float v[8];
#pragma unroll
                for (unsigned half = 0; half < 2U; ++half) {
                    const unsigned l = 4U * g + 16U * half;
                    const float4 k0 = *reinterpret_cast<const float4*>(k + l);
                    const float4 k1 = *reinterpret_cast<const float4*>(k + l + 32U);
                    const float4 k2 = *reinterpret_cast<const float4*>(k + l + 64U);
                    const float4 k3 = *reinterpret_cast<const float4*>(k + l + 96U);
                    const float kk0[4] = {k0.x, k0.y, k0.z, k0.w}, kk1[4] = {k1.x, k1.y, k1.z, k1.w};
                    const float kk2[4] = {k2.x, k2.y, k2.z, k2.w}, kk3[4] = {k3.x, k3.y, k3.z, k3.w};
#pragma unroll
                    for (unsigned c = 0; c < 4U; ++c)
                        v[4U * half + c] = relu_partial(q[l + c], q[l + c + 32U], q[l + c + 64U], q[l + c + 96U],
                                                       kk0[c], kk1[c], kk2[c], kk3[c]);
                }
#pragma unroll
                for (unsigned c = 0; c < 4U; ++c) a[4U * g + c] = __fadd_rn(v[c], v[4U + c]);
            }
#pragma unroll
            for (unsigned width = 8; width > 0; width /= 2U) {
#pragma unroll
                for (unsigned l = 0; l < width; ++l) a[l] = __fadd_rn(a[l], a[l + width]);
            }
            float total = 0.0F;
#pragma unroll
            for (unsigned hh = 0; hh < relu_heads; ++hh)
                total = __fadd_rn(total, fmaxf(__shfl_sync(full_warp, a[0], static_cast<int>((lane & ~3U) | hh)), 0.0F));
            if (h == 0U) ss[rr][i] = __fdiv_rn(total, __fsqrt_rn(static_cast<float>(relu_dim)));
        }
        __syncthreads();
        for (unsigned e = tid; e < tree_rows * key_tile; e += threads) {
            const unsigned row_in = e / key_tile, i = e % key_tile;
            VisibleRow output;
            if (i < nb && span.load(r0 + row_in, output) && b0 + i < relu_limit(output, spec))
                output.row.scores[b0 + i] = ss[row_in][i];
        }
    }
}

inline constexpr unsigned relu_threads = warps_per_cta * shuffle_groups;

template <ScoreSchedule Schedule>
__device__ __forceinline__ bool indexed_score_span(AffineRows rows, unsigned count, unsigned groups, ScoreSpec,
                                                   AffineScoreSpan& span) {
    span = {rows.base, rows.stride, count, groups, rows.first_position};
    return true;
}

template <ScoreSchedule Schedule>
__device__ __forceinline__ bool indexed_score_span(SlotRows rows, unsigned count, unsigned groups, ScoreSpec spec,
                                                   SlotScoreSpan& span) {
    const unsigned anchor = Schedule == ScoreSchedule::warp_shuffle ? blockIdx.y : blockIdx.z;
    return slot_score_span(rows, count, anchor, groups, spec, span);
}

template <class Rows>
using IndexedScoreSpan = std::conditional_t<std::same_as<Rows, AffineRows>, AffineScoreSpan, SlotScoreSpan>;

template <class Rows>
__global__ __launch_bounds__(relu_threads) void shuffle_indexed_scores(Rows rows, unsigned count, unsigned groups, ScoreSpec spec) {
    IndexedScoreSpan<Rows> span;
    if (indexed_score_span<ScoreSchedule::warp_shuffle>(rows, count, groups, spec, span))
        shuffle_relu_tile(span, blockIdx.x * warps_per_cta + threadIdx.x / shuffle_groups, spec);
}

template <class Rows>
__global__ __launch_bounds__(relu_threads, 1) void tree_indexed_scores(Rows rows, unsigned count, unsigned groups, ScoreSpec spec) {
    IndexedScoreSpan<Rows> span;
    if (indexed_score_span<ScoreSchedule::row_tree>(rows, count, groups, spec, span))
        tree_relu_tile(span, blockIdx.x, blockIdx.y, spec);
}

template <class Map>
__device__ __forceinline__ AffineScoreSpan segment_score_span(const SegmentRows<Map>& rows, const ScoreSegment& segment) {
    auto base = offset_rows(rows.base, rows.stride, segment.first_row);
    base.keys = segment.keys;
    return {base, rows.stride, segment.rows, segment.groups, segment.first_position};
}

template <class Map>
__global__ __launch_bounds__(relu_threads) void shuffle_segment_scores(SegmentRows<Map> rows, ScoreSpec spec) {
    const auto& map = rows.descriptor();
    if (map.count == 0 || map.count > rows.capacity) return;
    const auto total = map.segments[map.count - 1U].wave_end;
    unsigned seg = 0, begin = 0;
    const auto step = std::uint64_t{gridDim.x} * warps_per_cta;
    for (auto wave = std::uint64_t{blockIdx.x} * warps_per_cta + threadIdx.x / shuffle_groups; wave < total; wave += step) {
        while (wave >= map.segments[seg].wave_end) begin = map.segments[seg++].wave_end;
        shuffle_relu_tile(segment_score_span(rows, map.segments[seg]), static_cast<unsigned>(wave - begin), spec);
    }
}

template <class Map>
__global__ __launch_bounds__(relu_threads, 1) void tree_segment_scores(SegmentRows<Map> rows, ScoreSpec spec) {
    const auto& map = rows.descriptor();
    if (map.count > rows.capacity || blockIdx.z >= map.count) return;
    const auto& segment = map.segments[blockIdx.z];
    if (segment.groups > spec.top_k)
        tree_relu_tile(segment_score_span(rows, segment), blockIdx.x, blockIdx.y, spec);
}

} // namespace detail

// Opt-in schedules of relu_tree for four 128-feature heads. X3 slots share
// keys within each slot. Live slot rows and buffer extents follow K5A's contract.
template <ScoreForm Form, ScoreSchedule Schedule, class Rows>
    requires (Form == ScoreForm::relu_tree && (std::same_as<Rows, AffineRows> || std::same_as<Rows, SlotRows>))
cudaError_t enqueue_scores(Rows rows, unsigned row_count, unsigned max_groups, ScoreSpec spec, cudaStream_t stream) {
    if (row_count == 0 || max_groups <= spec.top_k) return cudaSuccess;
    if (!detail::tiled_relu_spec(spec)) return cudaErrorInvalidValue;
    auto groups = max_groups;
    unsigned spans = 1;
    if constexpr (std::same_as<Rows, AffineRows>) {
        const auto visible = (std::uint64_t{rows.first_position} + row_count) / spec.block_width;
        if (visible <= spec.top_k) return cudaSuccess;
        if (visible > max_groups || !rows.base.query || !rows.base.keys || !rows.base.scores) return cudaErrorInvalidValue;
        groups = static_cast<unsigned>(visible);
    } else {
        if (!rows.step || !rows.slots || row_count > exec::kStepMaxRows) return cudaErrorInvalidValue;
        spans = row_count;
    }
    if constexpr (Schedule == ScoreSchedule::warp_shuffle) {
        const auto waves = detail::shuffle_waves(row_count, groups);
        if (waves > ~0U) return cudaErrorInvalidValue;
        const auto blocks = detail::score_tiles(static_cast<unsigned>(waves), detail::warps_per_cta);
        detail::shuffle_indexed_scores<<<dim3(blocks, spans), detail::relu_threads, 0, stream>>>(
            rows, row_count, groups, spec);
    } else {
        static_assert(Schedule == ScoreSchedule::row_tree);
        const auto tiles = detail::score_tiles(groups, detail::tree_groups);
        if (tiles > 65535U) return cudaErrorInvalidValue;
        detail::tree_indexed_scores<<<dim3(detail::score_tiles(row_count, detail::tree_rows), tiles, spans),
                                      detail::relu_threads, 0, stream>>>(rows, row_count, groups, spec);
    }
    return cudaGetLastError();
}

// shuffle_blocks is the fixed grid-stride budget. For eager work, zero uses
// the packed wave count. Replay requires an explicit budget at capture.
template <ScoreForm Form, ScoreSchedule Schedule, class Map>
    requires (Form == ScoreForm::relu_tree)
cudaError_t enqueue_scores(SegmentRows<Map> rows, SegmentScoreShape shape, ScoreSpec spec, cudaStream_t stream,
                           unsigned shuffle_blocks = 0) {
    if (shape.rows == 0 || shape.groups <= spec.top_k) return cudaSuccess;
    if (!detail::tiled_relu_spec(spec) || shape.rows_per_segment == 0 || !rows.base.query || !rows.base.scores)
        return cudaErrorInvalidValue;
    if constexpr (rows.replay) {
        if (!rows.map) return cudaErrorInvalidValue;
    } else {
        if (rows.map.count > rows.capacity) return cudaErrorInvalidValue;
        if (rows.map.count == 0 || rows.map.segments[rows.map.count - 1U].wave_end == 0) return cudaSuccess;
    }
    if constexpr (Schedule == ScoreSchedule::warp_shuffle) {
        if constexpr (!rows.replay) {
            if (shuffle_blocks == 0)
                shuffle_blocks = detail::score_tiles(rows.map.segments[rows.map.count - 1U].wave_end, detail::warps_per_cta);
        }
        if (shuffle_blocks == 0 || shuffle_blocks > 2147483647U) return cudaErrorInvalidValue;
        detail::shuffle_segment_scores<<<shuffle_blocks, detail::relu_threads, 0, stream>>>(rows, spec);
    } else {
        static_assert(Schedule == ScoreSchedule::row_tree);
        const auto tiles = detail::score_tiles(shape.groups, detail::tree_groups);
        if (tiles > 65535U) return cudaErrorInvalidValue;
        const unsigned spans = rows.replay ? rows.capacity : rows.descriptor().count;
        detail::tree_segment_scores<<<dim3(detail::score_tiles(shape.rows_per_segment, detail::tree_rows), tiles, spans),
                                      detail::relu_threads, 0, stream>>>(rows, spec);
    }
    return cudaGetLastError();
}

} // namespace strix::cuda::core::select
