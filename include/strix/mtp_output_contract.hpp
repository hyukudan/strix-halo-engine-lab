#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

struct MtpOutputStructuralContract {
    std::uint64_t wide_input_size = 0;
    std::uint64_t mixed_hidden_size = 0;
    std::uint64_t vocab_size = 0;

    bool uses_gated_residual_head_mix = false;
    bool head_mix_has_injection = false;
    bool lm_head_required = false;

    // Formula is covered by the existing gated-residual oracle. The exact
    // checkpoint binding of the final LM-head weight remains unresolved.
    bool mixer_semantics_resolved = true;
    bool head_binding_resolved = false;
    bool output_semantics_resolved = false;
};

[[nodiscard]] MtpOutputStructuralContract
build_mtp_output_structural_contract(
    const ModelContract& model);

void validate_mtp_output_structural_contract(
    const ModelContract& model,
    const MtpOutputStructuralContract& contract);

}  // namespace strix
