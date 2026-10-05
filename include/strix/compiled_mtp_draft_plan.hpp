#pragma once

#include "strix/model_contract.hpp"
#include "strix/mtp_attention_contract.hpp"
#include "strix/mtp_draft_binding.hpp"
#include "strix/mtp_draft_contract.hpp"
#include "strix/mtp_draft_readiness.hpp"
#include "strix/mtp_draft_state_contract.hpp"
#include "strix/mtp_forward_contract.hpp"
#include "strix/mtp_output_contract.hpp"
#include "strix/mtp_draft_tensor_audit.hpp"

#include <cstdint>
#include <vector>

namespace strix {

enum class MtpDraftPlanStep {
    input_fusion,
    attention_hyper_mix,
    qsa_indexer,
    sparse_attention,
    attention_combine,
    mlp_hyper_mix,
    routed_and_shared_moe,
    mlp_combine,
    publish_next_state,
    output_head_mix,
    lm_head,
};

struct CompiledMtpDraftPlan {
    std::uint64_t proposal_identity = 0;
    std::uint64_t contract_identity = 0;
    std::uint64_t plan_identity = 0;

    MtpDraftStructuralContract structural;
    MtpAttentionStructuralContract attention;
    MtpDraftStateContract state;
    MtpOutputStructuralContract output;
    MtpForwardCompositionContract forward;
    MtpDraftTensorAudit tensor_audit;
    MtpDraftReadiness readiness;

    std::vector<MtpDraftPlanStep> steps;
    bool executable = false;
};

[[nodiscard]] CompiledMtpDraftPlan
compile_mtp_draft_plan(
    const ModelContract& model,
    const ModelManifest& manifest,
    const MtpDraftRuntimeBinding& binding);

void require_executable_mtp_draft_plan(
    const CompiledMtpDraftPlan& plan);

}  // namespace strix
