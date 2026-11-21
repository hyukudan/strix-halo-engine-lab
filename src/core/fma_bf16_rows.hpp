#pragma once

#include "core/dense_row_plan.hpp"
#include "core/fma_rows.hpp"

#include <limits>

namespace strix::core::fma_rows {

// Ascending-K FP32 FMA. The 16-wide fallback admits a full stage even when
// the shape cannot use the automatic forms' 32-wide admission rule.
inline hipError_t enqueue_bf16_rows(const float* input, unsigned input_width, unsigned rows,
                                    std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                    hipStream_t stream) {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!input || rows > std::numeric_limits<unsigned>::max() - 128U || !admits(input_width, rows, 16U) ||
        input_width > max_elements / rows || reinterpret_cast<std::uintptr_t>(input) % 16U != 0 ||
        projections.empty() || projections.size() > kGroupMax) return hipErrorInvalidValue;

    Group group{};
    group.count = static_cast<unsigned>(projections.size());
    for (unsigned p = 0; p < group.count; ++p) {
        const auto& projection = projections[p];
        if (!projection.weights || !projection.output || projection.outputs == 0 || projection.outputs > max_outputs ||
            projection.outputs > max_elements / rows ||
            projection.outputs > std::numeric_limits<unsigned>::max() - 128U ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / input_width)
            return hipErrorInvalidValue;
        group.w[p] = projection.weights;
        group.y[p] = projection.output;
        group.n[p] = projection.outputs;
    }

    if (admits(input_width, rows, 32U)) {
        launch_auto(input, Weight::BF16, input_width, rows, group, stream);
    } else {
        launch<128, 64, 1, 16, true>(input, Weight::BF16, input_width, rows, group, stream);
    }
    return hipGetLastError();
}

}  // namespace strix::core::fma_rows
