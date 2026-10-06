#pragma once

#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/mtp_draft_tensor_audit.hpp"

#include <cstdint>

namespace strix {

enum class MtpLmHeadSource {
    unresolved,
    mtp_sidecar_shared_head,
    target_output_fallback,
};

struct MtpOutputStructuralContract {
    std::uint64_t wide_input_size = 0;
    std::uint64_t mixed_hidden_size = 0;
    std::uint64_t vocab_size = 0;

    bool uses_gated_residual_head_mix = false;
    bool head_mix_has_injection = false;
    bool lm_head_required = false;

    bool head_mix_binding_resolved = false;
    std::uint32_t head_norm_tensor_index = 0;
    std::uint32_t head_down_tensor_index = 0;
    std::uint32_t head_up_tensor_index = 0;

    MtpLmHeadSource lm_head_source =
        MtpLmHeadSource::unresolved;
    std::uint32_t lm_head_tensor_index = 0;

    // Formula is covered by the existing gated-residual oracle. The exact
    // checkpoint binding of the final LM-head weight remains unresolved.
    bool mixer_semantics_resolved = true;
    bool head_binding_resolved = false;
    bool output_semantics_resolved = false;
};

[[nodiscard]] MtpOutputStructuralContract
build_mtp_output_structural_contract(
    const ModelContract& model);

[[nodiscard]] MtpOutputStructuralContract
bind_mtp_output_head(
    const ModelContract& model,
    const ModelManifest& manifest,
    const MtpDraftTensorAudit& audit);

void validate_mtp_output_structural_contract(
    const ModelContract& model,
    const MtpOutputStructuralContract& contract);

}  // namespace strix
