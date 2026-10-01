#include "strix/qualified_mtp_draft_backend.hpp"

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

struct FakeForward final
    : strix::QualifiedMtpForwardBackend {
    std::vector<std::uint32_t> proposed{7, 8};
    std::uint32_t proposal_calls = 0;
    std::uint32_t reconcile_calls = 0;

    std::vector<std::uint32_t>
    propose_mtp_candidates(
        const strix::QualifiedMtpDraftPlan&,
        const strix::MtpDraftRuntimeBinding&,
        const strix::SerialSessionContext&,
        const strix::DraftProposalStateContext&,
        std::uint32_t) override {
        ++proposal_calls;
        return proposed;
    }

    void reconcile_mtp_after_target(
        const strix::QualifiedMtpDraftPlan&,
        const strix::MtpDraftRuntimeBinding&,
        const strix::DraftProposalStateContext&,
        const strix::SerialSessionContext&,
        std::span<const std::uint32_t>) override {
        ++reconcile_calls;
    }
};

strix::QualifiedMtpDraftPlan qualified_plan() {
    strix::QualifiedMtpDraftPlan plan;
    plan.base.proposal_identity = 0x1111ULL;
    plan.base.plan_identity = 0x2222ULL;
    plan.base.contract_identity = 0x3333ULL;
    plan.base.structural.vocab_size = 100;
    plan.replay_policy.policy_identity = 0x4444ULL;
    plan.replay_assessment.passes_declared_policy = true;
    plan.independent_reference_identity = 0x5555ULL;
    plan.qualification_identity = 0x6666ULL;
    plan.executable = true;
    return plan;
}

strix::MtpDraftRuntimeBinding binding() {
    strix::MtpDraftRuntimeBinding value;
    value.compatibility_identity = 0x1111ULL;
    value.physical_binding_identity = 0x7777ULL;
    value.model_content_identity.bytes[0] = 1;
    value.tensor_indices = {0};
    value.total_tensor_bytes = 4096;
    return value;
}

strix::SerialSessionContext target_session() {
    strix::SerialSessionContext session;
    session.runtime_compatibility_identity = 0xaaaaULL;
    session.next_position = 10;
    session.committed_tokens = 10;
    session.health = strix::SerialSessionHealth::ready;
    return session;
}

}  // namespace

int main() {
    auto plan = qualified_plan();
    auto draft_binding = binding();
    auto target = target_session();

    auto proposal_state =
        strix::make_draft_proposal_state(
            draft_binding.compatibility_identity,
            target);

    FakeForward forward;
    strix::QualifiedMtpDraftBackend backend(
        plan,
        draft_binding,
        forward);

    const auto tokens =
        backend.propose_candidates(
            target,
            proposal_state,
            4);

    expect(tokens ==
               std::vector<std::uint32_t>({7, 8}) &&
               forward.proposal_calls == 1,
           "qualified MTP wrapper forwards valid proposal request");

    backend.reconcile_after_target(
        proposal_state,
        target,
        tokens);
    expect(forward.reconcile_calls == 1,
           "qualified MTP wrapper forwards valid reconciliation");

    const auto empty =
        backend.propose_candidates(
            target,
            proposal_state,
            0);
    expect(empty.empty() &&
               forward.proposal_calls == 1,
           "zero-depth MTP proposal bypasses forward backend");

    auto wrong_state = proposal_state;
    ++wrong_state.proposal_identity;
    bool state_rejected = false;
    try {
        static_cast<void>(
            backend.propose_candidates(
                target,
                wrong_state,
                2));
    } catch (const std::invalid_argument&) {
        state_rejected = true;
    }
    expect(state_rejected &&
               forward.proposal_calls == 1,
           "MTP wrapper rejects proposal state from another plan");

    auto unqualified = plan;
    unqualified.executable = false;
    unqualified.replay_assessment.passes_declared_policy = false;
    bool plan_rejected = false;
    try {
        strix::QualifiedMtpDraftBackend invalid(
            unqualified,
            draft_binding,
            forward);
        static_cast<void>(invalid);
    } catch (const std::logic_error&) {
        plan_rejected = true;
    }
    expect(plan_rejected,
           "MTP wrapper cannot be constructed from unqualified plan");

    auto bad_forward = forward;
    bad_forward.proposed = {101};
    strix::QualifiedMtpDraftBackend bad_backend(
        plan,
        draft_binding,
        bad_forward);
    bool vocab_rejected = false;
    try {
        static_cast<void>(
            bad_backend.propose_candidates(
                target,
                proposal_state,
                2));
    } catch (const std::runtime_error&) {
        vocab_rejected = true;
    }
    expect(vocab_rejected,
           "MTP wrapper rejects out-of-vocabulary draft token");

    if (failures == 0) {
        std::cout << "qualified MTP draft backend tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
