#pragma once

#include "strix/serial_session.hpp"

#include <cstdint>

namespace strix {

enum class DraftProposalStatePhase {
    synchronized,
    proposal_active,
    reconciliation_required,
    poisoned,
};

struct DraftProposalStateContext {
    std::uint64_t proposal_identity = 0;
    std::uint64_t target_runtime_compatibility_identity = 0;
    std::uint64_t synchronized_target_position = 0;
    std::uint64_t synchronized_target_committed_tokens = 0;
    std::uint64_t generation = 0;
    DraftProposalStatePhase phase =
        DraftProposalStatePhase::synchronized;
};

[[nodiscard]] DraftProposalStateContext
make_draft_proposal_state(
    std::uint64_t proposal_identity,
    const SerialSessionContext& target_session);

void validate_draft_proposal_state(
    const DraftProposalStateContext& state,
    const SerialSessionContext& target_session);

void begin_draft_proposal(
    DraftProposalStateContext& state,
    const SerialSessionContext& target_session);

void finish_draft_proposal(
    DraftProposalStateContext& state);

void reconcile_draft_proposal_state(
    DraftProposalStateContext& state,
    const SerialSessionContext& target_session);

void poison_draft_proposal_state(
    DraftProposalStateContext& state) noexcept;

}  // namespace strix
