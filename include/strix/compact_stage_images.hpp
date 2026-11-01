#pragma once

#include "strix/qpf2k_defaults.hpp"
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace strix {

// An exact storage policy, independent of the prefill numerical identity.
constexpr bool compact_stage_images_requested(const char* value) noexcept {
    return !value || value[0] != '0';
}
inline bool compact_stage_images_enabled() {
    static const bool enabled = compact_stage_images_requested(std::getenv("STRIX_QPF2K_COMPACT_IMAGES")) &&
        compact_stage_images_requested(std::getenv("STRIX_QPF2K_DEFAULTS")) &&
        !(std::getenv("STRIX_BASELINE") && std::string_view(std::getenv("STRIX_BASELINE")) == "1");
    return enabled && !qpf2k::baseline;
}

// Store the upper bits of an ALREADY rounded FP32 result. Do not use a hardware
// BF16 cast or round again: neither may replace the producer's NaN payload rules.
constexpr std::uint16_t compact_rounded_bits(std::uint32_t bits) noexcept {
    return static_cast<std::uint16_t>(bits >> 16U);
}
constexpr std::uint32_t widen_compact_bits(std::uint16_t bits) noexcept {
    return static_cast<std::uint32_t>(bits) << 16U;
}

} // namespace strix
