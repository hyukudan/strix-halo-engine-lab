#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace strix::attn {

enum class SelectedBlockSchedule { warp, staged };

inline constexpr unsigned kSelectedBlockLanes = 32;
inline constexpr unsigned kSelectedBlockLaneWidth = 4;
inline constexpr unsigned kSelectedBlockTile = 16;

// The FP32 score tree has 32 chains of up to four FMAs. Narrower vectors
// use the low lanes and leave the other chains at +0. The staged form
// moves BF16 in 16-byte vectors and pads K for key-parallel score loads.
template <unsigned QkDim, unsigned ValueDim, unsigned BlockKeys, unsigned MaxBlocks, unsigned HeadsPerGroup>
struct SelectedBlockShape {
    static constexpr unsigned qk_dim = QkDim, value_dim = ValueDim;
    static constexpr unsigned block_keys = BlockKeys, list_capacity = MaxBlocks, heads_per_group = HeadsPerGroup;
    static constexpr unsigned k_stride = QkDim + kSelectedBlockLaneWidth;
    static constexpr std::uint64_t k_elements =
        (std::uint64_t{BlockKeys} * k_stride + 7U) & ~std::uint64_t{7U};
    static constexpr unsigned warp_threads = 8U * kSelectedBlockLanes;
    static constexpr unsigned staged_threads = HeadsPerGroup * kSelectedBlockLanes;
    static constexpr std::uint64_t staged_bytes =
        (k_elements + std::uint64_t{BlockKeys} * ValueDim) * sizeof(std::uint16_t) +
        std::uint64_t{HeadsPerGroup} * QkDim * sizeof(float);

    static_assert(QkDim > 0U && QkDim % 8U == 0U && QkDim <= kSelectedBlockLanes * kSelectedBlockLaneWidth);
    static_assert(ValueDim > 0U && ValueDim % 8U == 0U && ValueDim <= kSelectedBlockLanes * kSelectedBlockLaneWidth);
    static_assert(BlockKeys > 0U && MaxBlocks > 0U && MaxBlocks <= std::numeric_limits<std::uint16_t>::max());
    static_assert(HeadsPerGroup > 0U && HeadsPerGroup <= 1024U / kSelectedBlockLanes);
    static_assert(staged_bytes <= std::numeric_limits<int>::max());
};

struct SelectedBlockPlan {
    unsigned grid_x = 0, grid_y = 0, threads = 0;
    std::size_t shared_bytes = 0;
};

template <class Shape>
[[nodiscard]] constexpr std::optional<SelectedBlockPlan>
make_selected_block_plan(SelectedBlockSchedule schedule, std::uint32_t rows, unsigned groups) {
    constexpr auto max_grid_x = 0x7fffffffU;
    if (rows == 0U) return SelectedBlockPlan{};
    if (groups == 0U) return std::nullopt;
    switch (schedule) {
    case SelectedBlockSchedule::warp: {
        constexpr auto warps_per_block = Shape::warp_threads / kSelectedBlockLanes;
        constexpr auto max_warps = std::uint64_t{max_grid_x} * warps_per_block;
        const auto row_groups = std::uint64_t{rows} * groups;
        if (row_groups > max_warps / Shape::heads_per_group) return std::nullopt;
        const auto warps = row_groups * Shape::heads_per_group;
        const auto blocks = (warps - 1U) / warps_per_block + 1U;
        return SelectedBlockPlan{static_cast<unsigned>(blocks), 1U, Shape::warp_threads, 0U};
    }
    case SelectedBlockSchedule::staged:
        if (rows > max_grid_x || groups > 65535U) return std::nullopt;
        return SelectedBlockPlan{rows, groups, Shape::staged_threads, static_cast<std::size_t>(Shape::staged_bytes)};
    }
    return std::nullopt;
}

// Per-sequence BF16 caches are packed [layer][capacity][group][dim]. Each
// query row belongs to one sequence and sees keys [0, pos0 + row - row0].
struct SelectedBlockSequence {
    const std::uint16_t* k;
    const std::uint16_t* v;
    std::uint32_t row0, pos0, capacity;
};

// Q and output are packed [row][group][head in group][dim]. Counts are
// [row][group] and block IDs are [row][group][list_capacity]. The caller
// provides resident buffers and valid maps, layers, counts and offsets.
// Block starts must be below the visible key count. Order and duplicates
// are retained. An empty list stores the result of +0/+0. Staged K and V
// base pointers must be 16-byte aligned.
struct SelectedBlockInput {
    const SelectedBlockSequence* sequences;
    const std::uint32_t* row_sequence;
    const float* query;
    const std::uint32_t* blocks;
    const std::uint32_t* counts;
    float* output;
    unsigned layer;
    float scale;
};

}  // namespace strix::attn
