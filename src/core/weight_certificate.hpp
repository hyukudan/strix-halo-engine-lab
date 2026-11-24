#pragma once

#include <algorithm>
#include <cmath>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>

namespace strix::core {

// Finite-weight certificate of the exact one-plane BF16 operand schedule (see exact_bf16_planes in
// core/dense_row_plan.hpp). The schedule is bit-identical to the three-plane split only while every product, partial
// sum and BF16 rounding stays finite: a non-finite weight meets the split's zero planes as 0 x inf, and an overflowing
// sum or rounding leaves the split's residual non-finite. The certificate is the largest weight magnitude (+infinity
// when any element is not finite), taken over the very bytes a backend stages; it is a host check made once at load.

// Largest |w| of an FP32 weight tensor, +infinity when an element is not finite.
inline float finite_magnitude_max(std::span<const float> values) {
    float top = 0.0F;
    for (const float value : values) {
        if (!std::isfinite(value)) return std::numeric_limits<float>::infinity();
        top = std::max(top, std::fabs(value));
    }
    return top;
}

// The same over a BF16 tensor (raw bits): an all-ones exponent field is infinity or NaN.
inline float finite_magnitude_max_bf16(std::span<const std::uint16_t> bits) {
    std::uint32_t top = 0U;
    for (const std::uint16_t value : bits) {
        if ((value & 0x7F80U) == 0x7F80U) return std::numeric_limits<float>::infinity();
        top = std::max<std::uint32_t>(top, value & 0x7FFFU);
    }
    // Magnitude bits are monotonic in the value, so the largest pattern is the largest magnitude.
    const std::uint32_t widened = top << 16U;
    return std::bit_cast<float>(widened);
}

// Largest sum magnitude (also bound of every partial sum) must stay below 2^126 so that no product, partial sum or BF16
// rounding of either schedule leaves the finite range. `operand_l1_max` is the largest L1 norm of one operand vector
// (the caller derives it from its normalisation, and the operands are assumed finite); false for a non-finite bound.
inline bool one_plane_sum_bounded(float operand_l1_max, float weight_max) {
    constexpr float limit = 0x1p126F;
    return std::isfinite(operand_l1_max) && std::isfinite(weight_max) && operand_l1_max * weight_max < limit;
}

}  // namespace strix::core
