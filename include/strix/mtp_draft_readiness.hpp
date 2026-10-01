#pragma once

#include "strix/mtp_attention_contract.hpp"
#include "strix/mtp_draft_contract.hpp"
#include "strix/mtp_draft_state_contract.hpp"
#include "strix/mtp_draft_tensor_audit.hpp"

#include <string>
#include <vector>

namespace strix {

struct MtpDraftReadiness {
    bool tensor_families_complete = false;
    bool input_fusion_semantics_resolved = false;
    bool attention_semantics_resolved = false;
    bool draft_state_semantics_resolved = false;
    bool output_semantics_resolved = false;
    bool executable = false;
    std::vector<std::string> blockers;
};

[[nodiscard]] MtpDraftReadiness
assess_mtp_draft_readiness(
    const MtpDraftStructuralContract& structural,
    const MtpAttentionStructuralContract& attention,
    const MtpDraftStateContract& state,
    const MtpDraftTensorAudit& tensors);

}  // namespace strix
