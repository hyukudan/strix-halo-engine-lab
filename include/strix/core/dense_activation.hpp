#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace strix::core::dense {

enum class Activation : std::uint8_t { native, fp4, fp8 };

[[nodiscard]] constexpr std::string_view activation_name(Activation a) {
    switch (a) {
    case Activation::native: return "native";
    case Activation::fp4: return "fp4";
    case Activation::fp8: return "fp8";
    }
    return "unknown";
}

[[nodiscard]] constexpr Activation activation(std::string_view word) {
    if (word == "native") return Activation::native;
    if (word == "fp4") return Activation::fp4;
    if (word == "fp8") return Activation::fp8;
    throw std::invalid_argument("invalid dense activation format");
}

// The current narrow MMA family consumes NVFP4 weights. Mixed-format checkpoints retain
// their usual arithmetic at sites without a compatible weight or tile shape.
[[nodiscard]] constexpr Activation select(Activation wanted, bool nvfp4_weight,
                                          std::uint32_t in, std::uint32_t out) {
    return nvfp4_weight && in && out && in % 128U == 0 && out % 16U == 0 ? wanted : Activation::native;
}

} // namespace strix::core::dense
