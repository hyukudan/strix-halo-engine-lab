#pragma once

#include "strix/model_contract.hpp"
#include "strix/mtp_attention_contract.hpp"
#include "strix/mtp_draft_contract.hpp"
#include "strix/mtp_draft_state_contract.hpp"
#include "strix/mtp_draft_tensor_audit.hpp"
#include "strix/mtp_output_contract.hpp"

#include <cstdint>

namespace strix {

struct MtpForwardCompositionContract {
    std::uint64_t proposal_identity = 0;
    std::uint64_t structural_contract_identity = 0;
    std::uint64_t manifest_layout_identity = 0;

    std::uint64_t hidden_size = 0;
    std::uint64_t wide_hidden_size = 0;
    std::uint64_t attention_query_width = 0;
    std::uint64_t attention_kv_width = 0;
    std::uint64_t draft_state_width = 0;
    std::uint64_t output_wide_input_size = 0;
    std::uint64_t output_hidden_size = 0;
    std::uint64_t vocab_size = 0;

    bool tensor_families_complete = false;
    bool attention_geometry_matches_hidden = false;
    bool state_matches_wide_hidden = false;
    bool output_consumes_draft_state = false;
    bool output_head_resolved = false;
    bool semantic_subcontracts_resolved = false;
    bool composition_resolved = false;

    std::uint64_t contract_identity = 0;
};

[[nodiscard]] MtpForwardCompositionContract
build_mtp_forward_composition_contract(
    const ModelContract& model,
    const MtpDraftStructuralContract& structural,
    const MtpAttentionStructuralContract& attention,
    const MtpDraftStateContract& state,
    const MtpOutputStructuralContract& output,
    const MtpDraftTensorAudit& tensors);

void validate_mtp_forward_composition_contract(
    const ModelContract& model,
    const MtpDraftStructuralContract& structural,
    const MtpAttentionStructuralContract& attention,
    const MtpDraftStateContract& state,
    const MtpOutputStructuralContract& output,
    const MtpDraftTensorAudit& tensors,
    const MtpForwardCompositionContract& contract);

}  // namespace strix
