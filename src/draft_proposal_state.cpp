#include "strix/draft_proposal_state.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

void require_target_binding(
    const DraftProposalStateContext& state,
    const SerialSessionContext& target_session) {
    if (state.proposal_identity == 0 ||
        state.target_runtime_compatibility_identity == 0 ||
        state.generation == 0) {
        throw std::invalid_argument(
            "draft proposal state is not initialized");
    }
    if (target_session.runtime_compatibility_identity == 0 ||
        state.target_runtime_compatibility_identity !=
            target_session.runtime_compatibility_identity) {
        throw std::invalid_argument(
            "draft proposal state target runtime identity mismatch");
    }
}

}  // namespace

DraftProposalStateContext
make_draft_proposal_state(
    std::uint64_t proposal_identity,
    const SerialSessionContext& target_session) {
    if (proposal_identity == 0) {
        throw std::invalid_argument(
            "draft proposal identity must be non-zero");
    }
    if (target_session.runtime_compatibility_identity == 0) {
        throw std::invalid_argument(
            "draft proposal state requires target runtime compatibility identity");
    }
    if (target_session.health !=
        SerialSessionHealth::ready) {
        throw std::logic_error(
            "draft proposal state requires ready target session");
    }

    DraftProposalStateContext state;
    state.proposal_identity =
        proposal_identity;
    state.target_runtime_compatibility_identity =
        target_session.runtime_compatibility_identity;
    state.synchronized_target_position =
        target_session.next_position;
    state.synchronized_target_committed_tokens =
        target_session.committed_tokens;
    state.generation = 1;
    state.phase =
        DraftProposalStatePhase::synchronized;
    return state;
}

void validate_draft_proposal_state(
    const DraftProposalStateContext& state,
    const SerialSessionContext& target_session) {
    require_target_binding(
        state,
        target_session);

    if (target_session.health !=
        SerialSessionHealth::ready) {
        throw std::logic_error(
            "draft proposal requires ready target session");
    }
    if (state.phase !=
        DraftProposalStatePhase::synchronized) {
        throw std::logic_error(
            "draft proposal state requires reconciliation");
    }
    if (state.synchronized_target_position !=
            target_session.next_position ||
        state.synchronized_target_committed_tokens !=
            target_session.committed_tokens) {
        throw std::logic_error(
            "draft proposal state is stale relative to target session");
    }
}

void begin_draft_proposal(
    DraftProposalStateContext& state,
    const SerialSessionContext& target_session) {
    validate_draft_proposal_state(
        state,
        target_session);
    if (state.generation ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "draft proposal state generation overflow");
    }
    ++state.generation;
    state.phase =
        DraftProposalStatePhase::proposal_active;
}

void finish_draft_proposal(
    DraftProposalStateContext& state) {
    if (state.proposal_identity == 0 ||
        state.generation == 0 ||
        state.phase !=
            DraftProposalStatePhase::proposal_active) {
        throw std::logic_error(
            "draft proposal finish requires active proposal state");
    }
    state.phase =
        DraftProposalStatePhase::reconciliation_required;
}

void reconcile_draft_proposal_state(
    DraftProposalStateContext& state,
    const SerialSessionContext& target_session) {
    require_target_binding(
        state,
        target_session);
    if (target_session.health !=
        SerialSessionHealth::ready) {
        throw std::logic_error(
            "draft proposal reconciliation requires ready target session");
    }
    if (state.phase !=
            DraftProposalStatePhase::reconciliation_required &&
        state.phase !=
            DraftProposalStatePhase::poisoned) {
        throw std::logic_error(
            "draft proposal reconciliation requires dirty or poisoned state");
    }

    state.synchronized_target_position =
        target_session.next_position;
    state.synchronized_target_committed_tokens =
        target_session.committed_tokens;
    state.phase =
        DraftProposalStatePhase::synchronized;
}

void poison_draft_proposal_state(
    DraftProposalStateContext& state) noexcept {
    state.phase =
        DraftProposalStatePhase::poisoned;
}

}  // namespace strix
