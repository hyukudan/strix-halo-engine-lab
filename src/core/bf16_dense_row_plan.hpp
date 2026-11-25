#pragma once

#include "core/dense_row_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

// Backend-neutral validation of a group of BF16 dense-row projections sharing one FP32 input. The geometry
// (shape, overflow and aliasing rules) is common; the operand ABI (pointer alignment, width multiple, projection
// cap, MMA tile) is supplied by the backend adapter as Bf16GroupLimits.
namespace strix::core {

struct Bf16GroupLimits {
    unsigned pointer_alignment;      // bytes; input, weights and outputs must be multiples
    unsigned width_multiple;         // input width must be a nonzero multiple
    std::size_t max_projections;     // per group
    unsigned tile_in, tile_out;      // the rounded-output (MMA) tile: width and outputs must be multiples
};

struct Bf16DenseRowGroup {
    const float* input;
    unsigned input_width;
    std::span<const Bf16RowProjection> projections;
};

[[nodiscard]] constexpr bool bf16_tile_eligible(unsigned in, unsigned out, const Bf16GroupLimits& limits) {
    return in % limits.tile_in == 0 && out % limits.tile_out == 0;
}

[[nodiscard]] inline bool valid_bf16_row_group(const Bf16DenseRowGroup& group, unsigned rows, bool allow_round_bf16,
                                               const Bf16GroupLimits& limits) {
    const auto& [input, input_width, projections] = group;
    const auto aligned = [&](const void* pointer) {
        return pointer && reinterpret_cast<std::uintptr_t>(pointer) % limits.pointer_alignment == 0;
    };
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!aligned(input) || input_width == 0 || input_width % limits.width_multiple != 0 || rows == 0 ||
        input_width > max_floats / rows || projections.empty() || projections.size() > limits.max_projections)
        return false;

    const bool round_bf16 = projections.front().round_bf16;
    if (round_bf16 && !allow_round_bf16) return false;
    std::uint64_t total_outputs = 0;
    for (const auto& projection : projections) {
        if (!projection.contiguous(input_width) || projection.round_bf16 != round_bf16 ||
            (round_bf16 && !bf16_tile_eligible(input_width, projection.outputs, limits)) ||
            projection.outputs == 0 || projection.outputs > max_floats / rows ||
            !aligned(projection.weights) || !aligned(projection.output) || projection.output == input ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / input_width)
            return false;
        total_outputs += projection.outputs;
    }
    return total_outputs <= std::numeric_limits<unsigned>::max();
}

}  // namespace strix::core
