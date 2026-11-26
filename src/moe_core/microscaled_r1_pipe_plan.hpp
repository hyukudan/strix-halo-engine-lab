#pragma once

#include "strix/device/launch_geometry.hpp"

namespace strix::moe_core::microscaled_r1::pipe {

inline constexpr unsigned kTop = 8U;
inline constexpr unsigned kWarp = 32U;
inline constexpr unsigned kGateWarps = 4U;
inline constexpr unsigned kDownColumns = 2U;

constexpr unsigned gate_blocks(unsigned intermediate) { return (intermediate - 1U) / kGateWarps + 1U; }
constexpr unsigned down_blocks(unsigned hidden) { return (hidden - 1U) / kDownColumns + 1U; }

constexpr device::LaunchGeometry gate_up_launch(unsigned intermediate) {
    return {{gate_blocks(intermediate), kTop, 1}, {kGateWarps * kWarp, 1, 1}, 0};
}
constexpr device::LaunchGeometry down_combine_launch(unsigned hidden) {
    return {{down_blocks(hidden), 1, 1}, {kTop * kWarp, 1, 1}, 0};
}

// The source lane holds four adjacent scale bytes in one aligned word.
struct ScaleWord {
    static constexpr unsigned kBytes = 4U;

#if defined(__CUDACC__)
    __host__ __device__
#endif
    static constexpr unsigned lane(unsigned scale) { return scale / kBytes; }

#if defined(__CUDACC__)
    __host__ __device__
#endif
    static constexpr unsigned shift(unsigned scale) { return (scale % kBytes) * 8U; }
};

}  // namespace strix::moe_core::microscaled_r1::pipe
