#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace strix::shared_expert_rows_diagnostic {
enum class Arm { reference_rows, shared_q8_rows };
struct Shape {
    std::uint32_t hidden = 0, intermediate = 0, rows = 0;
    std::uint64_t input_stride_bytes = 0, output_stride_bytes = 0, workspace_stride_bytes = 0;
};
struct Span { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
struct Weights { Span gate_q8, up_q8, down_q8, shared_gate_f32; };
struct Layout {
    std::uint64_t input_bytes = 0, output_bytes = 0, workspace_bytes = 0;
    std::uint64_t gate_up_weight_bytes = 0, down_weight_bytes = 0, scalar_weight_bytes = 0;
    std::uint32_t launches = 0;
};

// BF16 activations; gate/up/down are canonical Q8_0 and scalar gate is F32.
// Per-row workspace is gate[M], up[M]. Activation consumes up before scalar
// projection reuses up[0], exactly as in the single-row shared operator.
inline Layout layout(Shape s, Arm arm) {
    if (!s.hidden || s.hidden > 10240U || s.hidden % 32U ||
        !s.intermediate || s.intermediate > 10240U || s.intermediate % 32U ||
        (s.rows != 2U && s.rows != 4U && s.rows != 8U))
        throw std::invalid_argument("shared rows diagnostic geometry is invalid");
    if (arm != Arm::reference_rows && arm != Arm::shared_q8_rows)
        throw std::invalid_argument("shared rows diagnostic arm is invalid");
    const auto extent = [rows = s.rows](std::uint64_t stride, std::uint64_t payload) {
        if (stride < payload || stride % 16U ||
            stride > (std::numeric_limits<std::uint64_t>::max() - payload) / (rows - 1U))
            throw std::invalid_argument("shared rows diagnostic stride is invalid");
        return (rows - 1U) * stride + payload;
    };
    Layout result;
    result.input_bytes = extent(s.input_stride_bytes, static_cast<std::uint64_t>(s.hidden) * 2U);
    result.output_bytes = extent(s.output_stride_bytes, static_cast<std::uint64_t>(s.hidden) * 2U);
    result.workspace_bytes = extent(s.workspace_stride_bytes, static_cast<std::uint64_t>(s.intermediate) * 8U);
    result.gate_up_weight_bytes = static_cast<std::uint64_t>(s.hidden / 32U) * 34U * s.intermediate;
    result.down_weight_bytes = static_cast<std::uint64_t>(s.intermediate / 32U) * 34U * s.hidden;
    result.scalar_weight_bytes = static_cast<std::uint64_t>(s.hidden) * 4U;
    result.launches = arm == Arm::reference_rows ? 5U * s.rows : 3U + 2U * s.rows;
    return result;
}
inline Layout validate(Shape s, Span input, Weights weights, Span workspace, Span output, Arm arm) {
    const auto result = layout(s, arm);
    const auto check = [](Span v, std::uint64_t expected) {
        if (!v.address || v.address % 16U || v.bytes != expected ||
            v.bytes > std::numeric_limits<std::uintptr_t>::max() - v.address)
            throw std::invalid_argument("shared rows diagnostic view has invalid alignment or extent");
    };
    check(input, result.input_bytes); check(output, result.output_bytes);
    check(workspace, result.workspace_bytes);
    check(weights.gate_q8, result.gate_up_weight_bytes); check(weights.up_q8, result.gate_up_weight_bytes);
    check(weights.down_q8, result.down_weight_bytes); check(weights.shared_gate_f32, result.scalar_weight_bytes);
    const auto overlap = [](Span a, Span b) {
        return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
    };
    const std::array reads{input, weights.gate_q8, weights.up_q8, weights.down_q8, weights.shared_gate_f32};
    for (const auto write : {workspace, output})
        for (const auto read : reads) if (overlap(write, read))
            throw std::invalid_argument("shared rows diagnostic writable/read alias");
    if (overlap(workspace, output))
        throw std::invalid_argument("shared rows diagnostic writable alias");
    return result;
}
} // namespace strix::shared_expert_rows_diagnostic
