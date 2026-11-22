#pragma once

#include <bit>
#include <cstdint>
#include <string_view>

namespace strix::weight {

struct RowPitchRule {
    std::uint64_t collision_stride = 0;
    std::uint64_t pad = 0;
};

constexpr bool valid_row_pitch_rule(RowPitchRule rule) {
    if (rule.collision_stride == 0U) return rule.pad == 0U;
    return std::has_single_bit(rule.collision_stride) && rule.pad % 16U == 0U && rule.pad < rule.collision_stride;
}

constexpr std::uint64_t pitched_row_bytes(std::uint64_t row_bytes, RowPitchRule rule) {
    return rule.collision_stride != 0U && row_bytes % rule.collision_stride == 0U ? row_bytes + rule.pad : row_bytes;
}

// gfx1151 BF16 GEMV at 2048-byte pitch multiples measured 121-193 GB/s, rising to 219-230 GB/s with 256-byte padding and identical outputs.
inline constexpr RowPitchRule kGfx1151RowPitch{2048, 256};

// Device data: the placement rule of a HIP architecture name without its feature suffix ("gfx1151", not
// "gfx1151:xnack-"). Architectures without a measured rule keep the natural layout.
constexpr RowPitchRule row_pitch_rule_for_arch(std::string_view arch) {
    return arch == "gfx1151" ? kGfx1151RowPitch : RowPitchRule{};
}

}  // namespace strix::weight
