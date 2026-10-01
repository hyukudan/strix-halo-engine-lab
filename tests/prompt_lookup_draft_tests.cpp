#include "strix/prompt_lookup_draft.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

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
    session.execution_identity = 0x11ULL;
    session.state_layout_identity = 0x22ULL;
    session.runtime_compatibility_identity = 0x33ULL;
    session.next_position = 7;
    session.committed_tokens = 7;
    session.health =
        strix::SerialSessionHealth::ready;
    return session;
}

}  // namespace

int main() {
    {
        const std::vector<std::uint32_t> history{
            1, 2, 3, 9, 1, 2, 3};
        strix::PromptLookupDraftBackend backend(
            {3, 32},
            history);
        auto session = target();
        auto state =
            backend.make_state(session);

        strix::begin_draft_proposal(
            state,
            session);
        const auto proposal =
            backend.propose_candidates(
                session,
                state,
                3);
        expect(proposal ==
                   std::vector<std::uint32_t>({9, 1, 2}),
               "prompt lookup uses most recent non-overlapping matching n-gram");

        strix::finish_draft_proposal(state);
        const std::vector<std::uint32_t> emitted{
            9, 8};
        backend.reconcile_after_target(
            state,
            session,
            emitted);
        strix::reconcile_draft_proposal_state(
            state,
            session);
        expect(backend.history().size() ==
                   history.size() + emitted.size() &&
                   backend.history().back() == 8,
               "prompt lookup reconciliation appends emitted continuation");
    }

    {
        const std::vector<std::uint32_t> history{
            1, 2, 3, 4, 5};
        strix::PromptLookupDraftBackend backend(
            {3, 32},
            history);
        auto session = target();
        auto state =
            backend.make_state(session);
        strix::begin_draft_proposal(
            state,
            session);
        const auto proposal =
            backend.propose_candidates(
                session,
                state,
                4);
        expect(proposal.empty(),
               "prompt lookup returns no proposal when suffix has no prior match");
    }

    {
        const std::vector<std::uint32_t> history{
            0, 1, 2, 3, 4, 5, 6};
        strix::PromptLookupDraftBackend backend(
            {2, 5},
            history);
        expect(backend.history() ==
                   std::vector<std::uint32_t>({2, 3, 4, 5, 6}),
               "prompt lookup history is bounded by configured capacity");
    }

    {
        const std::vector<std::uint32_t> history{
            1, 2, 1, 2};
        strix::PromptLookupDraftBackend first(
            {2, 16},
            history);
        strix::PromptLookupDraftBackend second(
            {3, 16},
            history);
        auto session = target();
        auto state =
            first.make_state(session);
        strix::begin_draft_proposal(
            state,
            session);

        bool rejected = false;
        try {
            static_cast<void>(
                second.propose_candidates(
                    session,
                    state,
                    2));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected,
               "prompt lookup backend rejects state from another proposal identity");
    }

    {
        bool rejected = false;
        try {
            strix::PromptLookupDraftBackend invalid(
                {3, 3});
            static_cast<void>(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected,
               "prompt lookup requires capacity for n-gram plus continuation");
    }

    if (failures == 0) {
        std::cout << "prompt lookup draft tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
