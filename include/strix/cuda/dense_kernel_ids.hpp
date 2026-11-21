#pragma once

#include <cstdint>

namespace strix::cuda {

// Dispatch ID for the native Q8_0 projection with the four-chain A8 reduction.
inline constexpr std::uint32_t kDenseA8ImmaKernel = 0x515304;

} // namespace strix::cuda
