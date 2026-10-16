#include "strix/qualified_mtp_draft_backend.hpp"

#include <stdexcept>

namespace strix {

QualifiedMtpDraftBackend::QualifiedMtpDraftBackend(
    const QualifiedMtpDraftPlan& qualified_plan,
    const MtpDraftRuntimeBinding& binding,
    const MtpDraftStateMemoryBinding& state_memory,
    QualifiedMtpForwardBackend& backend)
    : qualified_plan_(qualified_plan),
      binding_(binding),
      state_memory_(state_memory),
      backend_(backend) {
    require_qualified_mtp_draft_plan(
        qualified_plan_);

    if (binding_.compatibility_identity == 0 ||
        qualified_plan_.base.proposal_identity !=
            binding_.compatibility_identity) {
        throw std::invalid_argument(
            "qualified MTP plan does not match draft runtime binding");
    }

    const auto memory_plan =
        plan_mtp_draft_state_memory(
            binding_.compatibility_identity,
            qualified_plan_.base.state,
            state_memory_.rows,
            state_memory_.element_bytes,
            state_memory_.alignment);
    validate_mtp_draft_state_memory_binding(
        memory_plan,
        state_memory_);
}

void QualifiedMtpDraftBackend::validate_request(
    const SerialSessionContext& target_session,
    const DraftProposalStateContext& proposal_state, DraftProposalStatePhase phase) const {
    require_qualified_mtp_draft_plan(
        qualified_plan_);

    validate_mtp_draft_proposal_phase(
        binding_, proposal_state, target_session, phase);

    if (proposal_state.proposal_identity !=
        qualified_plan_.base.proposal_identity) {
        throw std::invalid_argument(
            "MTP proposal state does not match qualified plan identity");
    }
}

std::vector<std::uint32_t>
QualifiedMtpDraftBackend::propose_candidates(
    const SerialSessionContext& target_session,
    const DraftProposalStateContext& proposal_state,
    std::uint32_t max_candidate_depth) {
    if (max_candidate_depth == 0) {
        return {};
    }

    validate_request(
        target_session,
        proposal_state, DraftProposalStatePhase::proposal_active);

    auto tokens =
        backend_.propose_mtp_candidates(
            qualified_plan_,
            binding_,
            state_memory_,
            target_session,
            proposal_state,
            max_candidate_depth);

    if (tokens.size() >
        static_cast<std::size_t>(
            max_candidate_depth)) {
        throw std::runtime_error(
            "qualified MTP forward backend exceeded proposal depth");
    }
    for (const auto token : tokens) {
        if (qualified_plan_.base.structural.vocab_size == 0 ||
            static_cast<std::uint64_t>(token) >=
                qualified_plan_.base.structural.vocab_size) {
            throw std::runtime_error(
                "qualified MTP forward backend produced out-of-vocabulary token");
        }
    }
    return tokens;
}

void QualifiedMtpDraftBackend::reconcile_after_target(
    const DraftProposalStateContext& proposal_state,
    const SerialSessionContext& target_session,
    std::span<const std::uint32_t> emitted_tokens) {
    validate_request(
        target_session,
        proposal_state, DraftProposalStatePhase::reconciliation_required);

    backend_.reconcile_mtp_after_target(
        qualified_plan_,
        binding_,
        state_memory_,
        proposal_state,
        target_session,
        emitted_tokens);
}

}  // namespace strix
