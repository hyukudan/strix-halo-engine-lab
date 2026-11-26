#pragma once

#include "core/dense_linear.hpp"
#include "core/dense_row_plan.hpp"
#include "core/f16_tile_gemm.hpp"
#include "core/gemv_rows.hpp"
#include "strix/fma_rows_policy.hpp"

namespace strix::core::fma_rows {

// The launch enqueue_bf16_rows makes for one group of projections on 16-byte aligned FP32 rows: refused, the tight-row
// body (decode rows), the shared BF16 WMMA tile, or the FP32 FMA rows (weights the tile's vector loads cannot read, or a
// grid past its launch limit). One source for enqueue_bf16_rows and for the routed entry, whose image routes stand in
// only for the WMMA tile.
enum class Bf16RowsPath : unsigned char { refused, tight_rows, wmma, fma };
enum class Bf16RowsSchedule : unsigned char { legacy, compact_narrow };

// Select only inside the incumbent WMMA path. Pitched weights stay on its pitched kernel.
constexpr bool admits_bf16_compact_projection(unsigned width, unsigned rows, const Bf16RowProjection& projection) {
    using Shape = f16_tile_gemm::Bf16TileShape<true>;
    constexpr Bf16RowLimits limits{gemv_rows::max_rows, 0U, Shape::panel, 16U, 1U};
    return rows <= 65535U * Shape::rows && projection.contiguous(width) &&
           plan_bf16_row_projection(width, projection.outputs, rows, true, limits) == Bf16RowPath::big_compact;
}

inline Bf16RowsPath bf16_rows_path(unsigned input_width, unsigned rows, std::span<const Bf16RowProjection> projections,
                                   unsigned max_outputs) noexcept {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    const auto group_limit = rows <= gemv_rows::max_rows ? dense_linear::kTightGroupMax : kGroupMax;
    if (rows > std::numeric_limits<unsigned>::max() - 128U || rows == 0U || input_width == 0U || input_width % 16U != 0U ||
        input_width > max_elements / rows || projections.empty() || projections.size() > group_limit)
        return Bf16RowsPath::refused;

    const bool round_bf16 = projections.front().round_bf16;
    const auto natural_row_bytes = std::size_t{input_width} * sizeof(std::uint16_t);
    const auto effective_pitch = [natural_row_bytes](const Bf16RowProjection& projection) {
        return projection.row_bytes != 0U ? projection.row_bytes : natural_row_bytes;
    };
    for (const auto& projection : projections) {
        const auto pitch = effective_pitch(projection);
        if (projection.round_bf16 != round_bf16 || projection.round_separate != projections.front().round_separate ||
            (projection.round_separate && !round_bf16) || !projection.weights || !projection.output || projection.outputs == 0 || projection.outputs > max_outputs ||
            projection.outputs > max_elements / rows ||
            projection.outputs > std::numeric_limits<unsigned>::max() - 128U ||
            pitch < natural_row_bytes || pitch % 16U != 0U ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / pitch)
            return Bf16RowsPath::refused;
    }
    if (rows <= gemv_rows::max_rows)  // the tight-row epilogue has the incumbent store only
        return projections.front().round_separate ? Bf16RowsPath::refused : Bf16RowsPath::tight_rows;
    const auto aligned_weights = [](const Bf16RowProjection& projection) {
        return reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U;
    };
    if (rows <= 65535U * f16_tile_gemm::kBM &&
        input_width <= std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK &&
        std::ranges::all_of(projections, aligned_weights)) {
        const auto pitch = effective_pitch(projections.front());
        return std::ranges::all_of(projections, [&](const Bf16RowProjection& projection) {
                   return effective_pitch(projection) == pitch;
               }) ? Bf16RowsPath::wmma : Bf16RowsPath::refused;
    }
    // Preserve existing admission for callers whose weights cannot use vector
    // loads or whose row grid exceeds the WMMA launch limit.
    return std::ranges::all_of(projections, [&](const Bf16RowProjection& projection) {
               return projection.contiguous(input_width);
           }) ? Bf16RowsPath::fma : Bf16RowsPath::refused;
}

inline bool bf16_rows_input_aligned(const float* input) noexcept {
    return input && reinterpret_cast<std::uintptr_t>(input) % 16U == 0U;
}

inline bool bf16_rows_call_admitted(const float* input, unsigned width, unsigned rows,
                                    std::span<const Bf16RowProjection> projections, unsigned max_outputs) {
    return bf16_rows_input_aligned(input) &&
           for_each_bf16_projection_group<kGroupMax>(projections, true, [&](auto group) {
               return bf16_rows_path(width, rows, group, max_outputs) != Bf16RowsPath::refused;
           });
}

}  // namespace strix::core::fma_rows
