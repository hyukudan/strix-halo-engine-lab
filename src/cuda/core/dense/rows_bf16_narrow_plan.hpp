#pragma once

#include <cstdint>

namespace strix::cuda_core {

inline constexpr unsigned kBf16NarrowOutputs = 8U;
inline constexpr unsigned kBf16NarrowThreads = 256U;

// Split complete incumbent m16 tiles. Both halves keep their original MMA
// fragment positions. This explicitly selected form covers one activation row.
[[nodiscard]] constexpr unsigned bf16_narrow_blocks(unsigned in, unsigned out) noexcept {
    if (in == 0U || in % 32U != 0U || out == 0U || out % (2U * kBf16NarrowOutputs) != 0U)
        return 0U;
    return out / kBf16NarrowOutputs;
}

} // namespace strix::cuda_core
