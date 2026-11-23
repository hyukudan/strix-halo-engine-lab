#pragma once

#include "cuda/core/select_rows.hpp"

#include <span>
#include <type_traits>

#if defined(__CUDACC__)
#define STRIX_SELECT_SEGMENT_HD __host__ __device__
#else
#define STRIX_SELECT_SEGMENT_HD
#endif

namespace strix::cuda::core::select {

enum class ScoreSchedule : std::uint8_t { warp_shuffle, row_tree };

// Fixed launch bounds for eager or captured work. Each replay packs against
// the same bounds and ScoreSpec before publishing the map to the stream.
struct SegmentScoreShape {
    unsigned rows, rows_per_segment, groups;
};

struct ScoreSegmentInput {
    unsigned rows;
    std::uint64_t first_position;
    const float* keys;
};

namespace detail {

inline constexpr unsigned relu_heads = 4, relu_dim = 128;
inline constexpr unsigned shuffle_rows = 4, shuffle_groups = 32, warps_per_cta = 8;
inline constexpr unsigned tree_rows = 64, tree_groups = 256;

STRIX_SELECT_SEGMENT_HD constexpr unsigned score_tiles(unsigned count, unsigned width) {
    return count / width + static_cast<unsigned>(count % width != 0);
}

STRIX_SELECT_SEGMENT_HD constexpr std::uint64_t shuffle_waves(unsigned rows, unsigned groups) {
    return std::uint64_t{score_tiles(rows, shuffle_rows)} * score_tiles(groups, shuffle_groups);
}

constexpr bool tiled_relu_spec(ScoreSpec spec) {
    return spec.heads == relu_heads && spec.dim == relu_dim && spec.block_width != 0;
}

struct ScoreSegment {
    std::uint64_t first_position;
    const float* keys;
    unsigned first_row, rows, groups, wave_end;
};

} // namespace detail

template <unsigned Capacity>
struct SegmentScoreMap {
    static_assert(Capacity > 0 && Capacity <= 65535U);
    static constexpr unsigned capacity = Capacity;
    unsigned count = 0;
    detail::ScoreSegment segments[Capacity]{};
};

// The inputs are in scratch-row order. Failed admission leaves the previous
// map intact. Empty input admits an idle replay.
template <unsigned Capacity>
constexpr bool pack_score_segments(std::span<const ScoreSegmentInput> input, SegmentScoreShape shape,
                                   ScoreSpec spec, SegmentScoreMap<Capacity>& out) {
    if (input.size() > Capacity || !detail::tiled_relu_spec(spec)) return false;
    unsigned rows = 0;
    std::uint64_t waves = 0;
    SegmentScoreMap<Capacity> map{};
    for (const auto& item : input) {
        if (item.rows == 0 || item.rows > shape.rows_per_segment || item.rows > shape.rows - rows ||
            item.first_position > ~std::uint64_t{0} - item.rows) return false;
        const auto groups = (item.first_position + item.rows) / spec.block_width;
        if (groups > shape.groups) return false;
        const auto group_count = static_cast<unsigned>(groups);
        if (groups > spec.top_k) {
            if (!item.keys) return false;
            waves += detail::shuffle_waves(item.rows, group_count);
            if (waves > ~0U) return false;
        }
        map.segments[map.count++] = {item.first_position, item.keys, rows, item.rows, group_count,
                                     static_cast<unsigned>(waves)};
        rows += item.rows;
    }
    out = map;
    return true;
}

// Map is either SegmentScoreMap<N> (eager) or const SegmentScoreMap<N>* (replay).
// Query, scores and selected IDs are laid out across the entire cohort. Each
// segment supplies its own key prefix. The caller owns storage extents and
// stream order and supplies disjoint writable rows.
template <class Map>
struct SegmentRows {
    using MapType = std::remove_cv_t<std::remove_pointer_t<Map>>;
    static constexpr bool replay = std::is_pointer_v<Map>;
    static constexpr unsigned capacity = MapType::capacity;

    RowBases base;
    RowStrides stride;
    Map map;

    STRIX_SELECT_SEGMENT_HD constexpr const MapType& descriptor() const {
        if constexpr (replay) return *map;
        else return map;
    }

    STRIX_SELECT_SEGMENT_HD constexpr bool load(unsigned index, detail::VisibleRow& out) const {
        const auto& desc = descriptor();
        if (desc.count > capacity) return false;
        for (unsigned i = 0; i < desc.count; ++i) {
            const auto& segment = desc.segments[i];
            if (index < segment.first_row || index - segment.first_row >= segment.rows) continue;
            auto row = detail::offset_rows(base, stride, index);
            row.keys = segment.keys;
            out = {row, segment.first_position + (index - segment.first_row)};
            return true;
        }
        return false;
    }
};

template <class Map>
SegmentRows(RowBases, RowStrides, Map) -> SegmentRows<Map>;

} // namespace strix::cuda::core::select

#undef STRIX_SELECT_SEGMENT_HD
