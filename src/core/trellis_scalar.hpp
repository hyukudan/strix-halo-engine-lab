#pragma once
#include "core/trellis_format.hpp"
#include "strix/weight/formats/ieee_half.hpp"

namespace strix::core::trellis {

// Both operators are exact in FP32. One final RNE conversion models the FP16
// FMA without relying on host half support.
STRIX_TRELLIS_HD inline float mul1_value(unsigned state) {
    using Book = Mul1Constants;
    using weight::ieee_half::decode;
    const float value = float(1024U + Book::state_sum(state)) * decode(Book::kScale) + decode(Book::kBias);
    std::uint16_t half = 0;
    (void)weight::ieee_half::encode_finite(value, half);
    return decode(half);
}

} // namespace strix::core::trellis
