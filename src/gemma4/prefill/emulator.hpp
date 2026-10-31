#pragma once
#include "prefill.hpp"
#include <array>

namespace strix::gemma4::prefill {
// One gfx1151 wave32 instruction for a single output element, including its C
// input. Half operands must be finite. The characterization fixtures bind the
// intra-instruction accumulator recipe; a host IEEE FMA chain is insufficient.
[[nodiscard]] float instruction(const std::array<std::uint16_t, 16>& a,
                                const std::array<std::uint16_t, 16>& b, float accumulator);
// Host-only bounded operator reference. It retains FP32 output and rounds both
// operands to FP16 before fixed, increasing K tiles, including masked tail tiles.
[[nodiscard]] std::vector<float> emulate_linear(LinearView weights, std::span<const float> input, unsigned rows);
} // namespace strix::gemma4::prefill
