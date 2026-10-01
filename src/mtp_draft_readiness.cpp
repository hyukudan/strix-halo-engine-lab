#include "strix/mtp_draft_readiness.hpp"

namespace strix {

MtpDraftReadiness
assess_mtp_draft_readiness(
    const MtpDraftStructuralContract& structural,
    const MtpAttentionStructuralContract& attention,
    const MtpDraftStateContract& state,
    const MtpOutputStructuralContract& output,
    const MtpDraftTensorAudit& tensors) {
    MtpDraftReadiness result;

    result.tensor_families_complete =
        tensors.structurally_complete();
    result.input_fusion_semantics_resolved =
        structural.input_fusion_semantics_resolved;
    result.attention_semantics_resolved =
        structural.attention_semantics_resolved &&
        attention.sparse_selection_semantics_resolved;
    result.draft_state_semantics_resolved =
        structural.draft_state_semantics_resolved &&
        state.state_width != 0;
    result.output_mixer_semantics_resolved =
        structural.output_mixer_semantics_resolved &&
        output.mixer_semantics_resolved;
    result.output_head_binding_resolved =
        output.head_binding_resolved;
    result.output_semantics_resolved =
        output.output_semantics_resolved;

    if (!result.tensor_families_complete) {
        result.blockers.emplace_back(
            "required MTP tensor families are missing");
    }
    if (!result.input_fusion_semantics_resolved) {
        result.blockers.emplace_back(
            "MTP input-fusion semantics are unresolved");
    }
    if (!result.attention_semantics_resolved) {
        result.blockers.emplace_back(
            "MTP attention/QSA cutoff-tie semantics are unresolved");
    }
    if (!result.draft_state_semantics_resolved) {
        result.blockers.emplace_back(
            "MTP next-step draft-state semantics are unresolved");
    }
    if (!result.output_mixer_semantics_resolved) {
        result.blockers.emplace_back(
            "MTP output mixer semantics are unresolved");
    }
    if (!result.output_head_binding_resolved) {
        result.blockers.emplace_back(
            "MTP LM-head checkpoint binding is unresolved");
    }
    if (!result.output_semantics_resolved &&
        result.output_mixer_semantics_resolved &&
        result.output_head_binding_resolved) {
        result.blockers.emplace_back(
            "MTP output semantics are unresolved");
    }
    if (!structural.forward_semantics_resolved) {
        result.blockers.emplace_back(
            "complete MTP forward semantics gate is unresolved");
    }

    result.executable =
        result.blockers.empty();
    return result;
}

}  // namespace strix
