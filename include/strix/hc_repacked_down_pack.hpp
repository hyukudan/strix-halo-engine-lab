#pragma once

#include "strix/hc_repacked_down_layout.hpp"

#include <span>
#include <string>

namespace strix::hc_repacked_down {

// Exact spans and canonical geometry are required. Source and destination must
// not overlap. All validation completes before any destination byte is written.
// These operations move raw little-endian Q8_0 bytes; no floating-point conversion.
void pack(const Geometry& g, std::span<const std::uint8_t> original,
          std::span<std::uint8_t> planes);
// Padding is ignored by unpack; use padding_zero to validate the complete plane.
void unpack(const Geometry& g, std::span<const std::uint8_t> planes,
            std::span<std::uint8_t> original);
[[nodiscard]] bool padding_zero(const Geometry& g,
                                std::span<const std::uint8_t> planes);
[[nodiscard]] std::string sha256_hex(std::span<const std::uint8_t> bytes);

}  // namespace strix::hc_repacked_down
