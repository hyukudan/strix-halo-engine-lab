#pragma once
#include <cstdint>
#include "glm_dims.hpp"

namespace strix::glm::state_layout {
// Serialized convolution blocks preserve q/k/v-major storage, not per-head interleaving.
inline constexpr std::uint64_t kConvFloatsPerHead = 3ULL * kKdaDim * (kConvK - 1);
constexpr std::uint64_t conv_floats(std::uint64_t heads) { return heads * kConvFloatsPerHead; }
constexpr std::uint64_t pooled_rows(std::uint64_t capacity) { return capacity / kKpool; }
static_assert(kConvFloatsPerHead == 3 * 128 * 3);
static_assert(sizeof(float) == 4);
}  // namespace strix::glm::state_layout
