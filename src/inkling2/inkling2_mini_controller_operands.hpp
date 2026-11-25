#pragma once

#include "inkling2_mini_config.hpp"
#include "core/dense_row_plan.hpp"
#include "core/weight_certificate.hpp"

#include <algorithm>
#include <cstdint>
#include <span>

namespace strix::inkling2 {

// Operand class of the controller's declared BF16 operands (down: two RMS-normalised rows; gates: the latent) on the
// tall dense route. The one-plane class (core::Bf16Operand::bf16_exact) is admitted only when the certificate proves
// the bound of core::low_rank_operands_bounded over the controller weights as staged; any other state is the FP32
// class (three planes). The certificate is only filled from tensors the resident weights hold (see
// read_mini_controller_certificate), never from a second source.
class MiniControllerCertificate {
public:
    void add_down(std::span<const std::uint16_t> bf16_bits) {
        down_max_ = std::max(down_max_, core::finite_magnitude_max_bf16(bf16_bits));
        ++tensors_;
    }
    void add_gate(std::span<const std::uint16_t> bf16_bits) {
        gate_max_ = std::max(gate_max_, core::finite_magnitude_max_bf16(bf16_bits));
        ++tensors_;
    }
    [[nodiscard]] bool bounded() const {
        return tensors_ != 0U && core::low_rank_operands_bounded(kHiddenWidth, kControllerRank, down_max_, gate_max_);
    }
    // The operand class that actually runs for a declared one.
    [[nodiscard]] core::Bf16Operand route(core::Bf16Operand declared) const {
        return bounded() ? declared : core::Bf16Operand::fp32;
    }

private:
    float down_max_ = 0.0F, gate_max_ = 0.0F;
    unsigned tensors_ = 0;
};

}  // namespace strix::inkling2
