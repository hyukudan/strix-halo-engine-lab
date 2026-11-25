#pragma once

#include "inkling2_mini_config.hpp"
#include "core/dense_row_plan.hpp"
#include "core/weight_certificate.hpp"

#include <algorithm>
#include <cstdint>
#include <span>

namespace strix::inkling2 {

// Operand class the resident CUDA backend may declare for its attention linears (the q/k/v/gate/sink/relative groups
// read rmsnorm_bf16 rows, the output projection reads the BF16 sigmoid-gated product). The one-plane class
// (core::Bf16Operand::bf16_exact) is the three-plane arithmetic only while every dot product, partial sum and BF16
// rounding stays finite (core/weight_certificate.hpp), so it is admitted only when the staged attention weights prove
// it; anything else (no certificate, a non-finite weight or gain, an oversized one) is the FP32 class (three planes).
//
// The proof uses the largest weight magnitude over every attention projection and the largest norm gain, taken from the
// staged bytes, and a conservative chain of operand bounds (finite residual rows assumed): a normalised and gained row
// has |x|_1 <= 1.01 width gain_max; a projection output element is at most that times weight_max; the relative-bias
// projection reads relative features (output elements of the first stage); the attended value is a convex combination of
// values, each a short convolution of kConvTaps taps (first-stage outputs or the static channel taps, weights) with first-stage outputs; the gate multiplies by a sigmoid
// (at most one). Every dot is bounded by operand L1 x weight_max < 2^126.
class MiniAttentionCertificate {
public:
    void add_projection(std::span<const std::uint16_t> bf16_bits) {
        weight_max_ = std::max(weight_max_, core::finite_magnitude_max_bf16(bf16_bits));
        ++projections_;
    }
    void add_gain(std::span<const std::uint16_t> bf16_bits) {
        gain_max_ = std::max(gain_max_, core::finite_magnitude_max_bf16(bf16_bits));
        ++gains_;
    }
    [[nodiscard]] bool bounded() const {
        if (projections_ == 0U || gains_ == 0U) return false;
        const float input_l1 = 1.01F * static_cast<float>(kHiddenWidth) * gain_max_;
        if (!core::one_plane_sum_bounded(input_l1, weight_max_)) return false;
        const float first = input_l1 * weight_max_;
        if (!core::one_plane_sum_bounded(static_cast<float>(kRelativeWidth) * first, weight_max_)) return false;
        const float value = static_cast<float>(kConvTaps) * std::max(first, weight_max_) * first;
        const float output_input = std::max(first, value);
        return core::one_plane_sum_bounded(static_cast<float>(kQueryWidth) * output_input, weight_max_);
    }
    // The operand class that actually runs for a declared one.
    [[nodiscard]] core::Bf16Operand route(core::Bf16Operand declared) const {
        return bounded() ? declared : core::Bf16Operand::fp32;
    }

private:
    float weight_max_ = 0.0F, gain_max_ = 0.0F;
    unsigned projections_ = 0, gains_ = 0;
};

}  // namespace strix::inkling2
