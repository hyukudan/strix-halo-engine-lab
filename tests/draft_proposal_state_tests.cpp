#include "strix/draft_proposal_state.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::SerialSessionContext target() {
    strix::SerialSessionContext session;
    session.execution_identity = 0x1111ULL;
    session.state_layout_identity = 0x2222ULL;
    session.runtime_compatibility_identity = 0x3333ULL;
    session.next_position = 10;
    session.committed_tokens = 10;
    session.health =
        strix::SerialSessionHealth::ready;
    return session;
}

}  // namespace

int main() {
    {
        auto session = target();
        auto state =
            strix::make_draft_proposal_state(
                0x4444ULL,
                session);

        strix::validate_draft_proposal_state(
            state,
            session);
        expect(state.phase ==
                   strix::DraftProposalStatePhase::synchronized &&
                   state.generation == 1,
               "new draft state is synchronized with target");

        strix::begin_draft_proposal(
            state,
            session);
        expect(state.phase ==
                   strix::DraftProposalStatePhase::proposal_active &&
                   state.generation == 2,
               "begin marks active proposal and advances generation");

        strix::finish_draft_proposal(state);
        expect(state.phase ==
                   strix::DraftProposalStatePhase::reconciliation_required,
               "finished proposal requires reconciliation");

        ++session.next_position;
        ++session.committed_tokens;
        strix::reconcile_draft_proposal_state(
            state,
            session);
        expect(state.phase ==
                   strix::DraftProposalStatePhase::synchronized &&
                   state.synchronized_target_position == 11 &&
                   state.synchronized_target_committed_tokens == 11,
               "reconciliation republishes target synchronization point");
    }

    {
        auto session = target();
        auto state =
            strix::make_draft_proposal_state(
                0x4444ULL,
                session);
        ++session.next_position;

        bool stale_rejected = false;
        try {
            strix::validate_draft_proposal_state(
                state,
                session);
        } catch (const std::logic_error&) {
            stale_rejected = true;
        }
        expect(stale_rejected,
               "stale draft synchronization point is rejected");
    }

    {
        auto session = target();
        auto state =
            strix::make_draft_proposal_state(
                0x4444ULL,
                session);
        strix::poison_draft_proposal_state(state);

        bool poisoned_rejected = false;
        try {
            strix::validate_draft_proposal_state(
                state,
                session);
        } catch (const std::logic_error&) {
            poisoned_rejected = true;
        }
        expect(poisoned_rejected,
               "poisoned draft state cannot propose before recovery");

        strix::reconcile_draft_proposal_state(
            state,
            session);
        strix::validate_draft_proposal_state(
            state,
            session);
        expect(state.phase ==
                   strix::DraftProposalStatePhase::synchronized,
               "explicit reconciliation recovers poisoned draft state");
    }

    {
        auto session = target();
        auto state =
            strix::make_draft_proposal_state(
                0x4444ULL,
                session);
        state.generation =
            std::numeric_limits<std::uint64_t>::max();

        bool overflow_rejected = false;
        try {
            strix::begin_draft_proposal(
                state,
                session);
        } catch (const std::overflow_error&) {
            overflow_rejected = true;
        }
        expect(overflow_rejected &&
                   state.phase ==
                       strix::DraftProposalStatePhase::synchronized,
               "generation overflow fails before proposal activation");
    }

    if (failures == 0) {
        std::cout << "draft proposal state tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
