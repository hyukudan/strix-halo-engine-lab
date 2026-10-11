#pragma once

#include <array>
#include <span>

namespace strix {
inline constexpr unsigned projection_threads = 256;
// Fixed binary tree: 128/64/32/16/8/4/2/1. The split implementation
// preserves that tree while moving the final 32 values into lane registers.
[[nodiscard]] float projection_reduce_lds(std::span<const float, projection_threads> values);
[[nodiscard]] float projection_reduce_shuffle_tail(std::span<const float, projection_threads> values);
// Explicit per-lane FMA, ascending K in strides of 256. This is the
// laboratory arithmetic reference, not a model-wide numerical contract.
[[nodiscard]] float projection_dot_reference(std::span<const float> weights,
                                             std::span<const float> activations);
} // namespace strix
