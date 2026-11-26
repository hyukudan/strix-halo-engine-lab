#pragma once

#include "core/dense_bf16_narrow_plan.hpp"

namespace strix::cuda_core {

inline constexpr unsigned kBf16NarrowOutputs = 8U;
inline constexpr unsigned kBf16NarrowThreads = 256U;

// The shared plan with the m16 tile geometry of this backend.
[[nodiscard]] constexpr unsigned bf16_narrow_blocks(unsigned in, unsigned out) noexcept {
    return ::strix::core::bf16_narrow_blocks(in, out, 32U, 2U * kBf16NarrowOutputs);
}

// The shared split selection with the same geometry.
[[nodiscard]] constexpr bool bf16_narrow_selected(unsigned rows, unsigned projections, bool round_bf16, unsigned in,
                                                  unsigned out) noexcept {
    return ::strix::core::bf16_narrow_selected(rows, projections, round_bf16, in, out, 32U, 2U * kBf16NarrowOutputs);
}

} // namespace strix::cuda_core
