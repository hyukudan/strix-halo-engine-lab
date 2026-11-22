#pragma once

#include "core/dense_linear.hpp"
#include "core/dense_row_plan.hpp"
#include "core/f16_tile_gemm.hpp"
#include "core/fma_rows.hpp"

#include <algorithm>
#include <limits>

namespace strix::core::fma_rows {

// Small NK projections use the decode sweep. Wide projections stage RNE BF16
// activations and resident BF16 weights in the shared WMMA tile.
inline hipError_t enqueue_bf16_rows(const float* input, unsigned input_width, unsigned rows,
                                    std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                    hipStream_t stream) {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!input || rows > std::numeric_limits<unsigned>::max() - 128U || !admits(input_width, rows, 16U) ||
        input_width > max_elements / rows || reinterpret_cast<std::uintptr_t>(input) % 16U != 0 ||
        projections.empty() || projections.size() > kGroupMax) return hipErrorInvalidValue;

    const bool round_bf16 = projections.front().round_bf16;
    Group group{};
    group.count = static_cast<unsigned>(projections.size());
    for (unsigned p = 0; p < group.count; ++p) {
        const auto& projection = projections[p];
        if (projection.round_bf16 != round_bf16 || !projection.weights || !projection.output || projection.outputs == 0 || projection.outputs > max_outputs ||
            projection.outputs > max_elements / rows ||
            projection.outputs > std::numeric_limits<unsigned>::max() - 128U ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / input_width)
            return hipErrorInvalidValue;
    }

    if (rows <= gemv_rows::max_rows) {
        // Same-input projections share one launch; each output keeps its single-launch instructions (bitwise).
        static_assert(kGroupMax <= dense_linear::kTightGroupMax);
        dense_linear::View views[dense_linear::kTightGroupMax]{};
        float* outputs[dense_linear::kTightGroupMax]{};
        for (unsigned p = 0; p < group.count; ++p) {
            views[p] = {reinterpret_cast<const std::uint8_t*>(projections[p].weights), input_width,
                        projections[p].outputs, std::size_t{input_width} * sizeof(std::uint16_t)};
            outputs[p] = projections[p].output;
        }
        if (group.count == 1U)
            return round_bf16 ? dense_linear::enqueue_bf16_tight_rows<true>(views[0], input, outputs[0], rows, stream)
                              : dense_linear::enqueue_bf16_tight_rows(views[0], input, outputs[0], rows, stream);
        return round_bf16
            ? dense_linear::enqueue_bf16_tight_rows_group<true>(views, outputs, group.count, input, rows, stream)
            : dense_linear::enqueue_bf16_tight_rows_group(views, outputs, group.count, input, rows, stream);
    }

    const auto aligned_weights = [](const Bf16RowProjection& projection) {
        return reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U;
    };
    if (rows <= 65535U * f16_tile_gemm::kBM &&
        input_width <= std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK &&
        std::ranges::all_of(projections, aligned_weights)) {
        f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> wmma_group{};
        for (const auto& projection : projections) wmma_group.append(projection);
        f16_tile_gemm::gemm_bf16_projections(input, rows, input_width, wmma_group, stream);
        return hipGetLastError();
    }

    // Preserve existing admission for callers whose weights cannot use vector
    // loads or whose row grid exceeds the WMMA launch limit.
    for (unsigned p = 0; p < group.count; ++p) {
        group.w[p] = projections[p].weights;
        group.y[p] = projections[p].output;
        group.n[p] = projections[p].outputs;
    }
    if (round_bf16) {
        if (admits(input_width, rows, 32U)) launch_auto<true>(input, Weight::BF16, input_width, rows, group, stream);
        else launch<128, 64, 1, 16, true, true>(input, Weight::BF16, input_width, rows, group, stream);
    } else if (admits(input_width, rows, 32U)) {
        launch_auto(input, Weight::BF16, input_width, rows, group, stream);
    } else {
        launch<128, 64, 1, 16, true>(input, Weight::BF16, input_width, rows, group, stream);
    }
    return hipGetLastError();
}

}  // namespace strix::core::fma_rows
