#pragma once

#include <cmath>
#include <limits>

namespace strix::cuda_core {

#if defined(_MSC_VER)
// The MSVC INFINITY macro narrows an out-of-range double in nvcc.
inline constexpr float kFlashNegativeInfinity = -std::numeric_limits<float>::infinity();
#else
inline constexpr float kFlashNegativeInfinity = -INFINITY;
#endif

}  // namespace strix::cuda_core
