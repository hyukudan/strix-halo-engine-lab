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
        const strix::MtpDraftStateMemoryBinding&,
        const strix::SerialSessionContext&,
        const strix::DraftProposalStateContext&,
        std::uint32_t) override {
        ++proposal_calls;
        return proposed;
    }

    void reconcile_mtp_after_target(
        const strix::QualifiedMtpDraftPlan&,
        const strix::MtpDraftRuntimeBinding&,
        const strix::MtpDraftStateMemoryBinding&,
        const strix::DraftProposalStateContext&,
        const strix::SerialSessionContext&,
        std::span<const std::uint32_t>) override {
        ++reconcile_calls;
    }
};

strix::QualifiedMtpDraftPlan qualified_plan() {
    strix::QualifiedMtpDraftPlan plan;
    plan.base.proposal_identity = 0x1111ULL;
    plan.base.contract_identity = 0x3333ULL;
    plan.base.structural.vocab_size = 100;
    plan.base.state.hidden_size = 2560;
    plan.base.state.hc_count = 4;
    plan.base.state.state_width = 10240;
    plan.base.state.one_state_row_per_request = true;
    plan.base.plan_identity =
        strix::recompute_compiled_mtp_draft_plan_identity(
            plan.base);
    plan.replay_policy.policy_identity = 0x4444ULL;
    plan.replay_assessment.passes_declared_policy = true;
    plan.independent_reference_identity = 0x5555ULL;
    plan.qualified_plan_identity =
        plan.base.plan_identity;
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

strix::MtpDraftStateMemoryBinding state_memory() {
    strix::MtpDraftStateMemoryBinding value;
    value.proposal_identity = 0x1111ULL;
    value.state_width = 10240;
    value.rows = 1;
    value.element_bytes = 2;
    value.alignment = 256;
    value.device_base =
        static_cast<std::uintptr_t>(
            0x60000000ULL);
    value.bytes = 10240ULL * 2ULL;
    value.allocation_generation = 1;

    strix::MtpDraftStateContract contract;
    contract.hidden_size = 2560;
    contract.hc_count = 4;
    contract.state_width = 10240;
    contract.one_state_row_per_request = true;
    const auto plan =
        strix::plan_mtp_draft_state_memory(
            value.proposal_identity,
            contract,
            value.rows,
            value.element_bytes,
            value.alignment);
    value.plan_identity =
        plan.plan_identity;
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
    auto draft_state_memory = state_memory();
    auto target = target_session();

    auto proposal_state =
        strix::make_draft_proposal_state(
            draft_binding.compatibility_identity,
            target);

    FakeForward forward;
    strix::QualifiedMtpDraftBackend backend(
        plan,
        draft_binding,
        draft_state_memory,
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

    auto wrong_memory =
        draft_state_memory;
    ++wrong_memory.proposal_identity;
    bool memory_rejected = false;
    try {
        strix::QualifiedMtpDraftBackend invalid_memory(
            plan,
            draft_binding,
            wrong_memory,
            forward);
        static_cast<void>(invalid_memory);
    } catch (const std::invalid_argument&) {
        memory_rejected = true;
    }
    expect(
        memory_rejected,
        "qualified MTP wrapper rejects draft-state storage from another proposal");

    auto unqualified = plan;
    unqualified.executable = false;
    unqualified.replay_assessment.passes_declared_policy = false;
    bool plan_rejected = false;
    try {
        strix::QualifiedMtpDraftBackend invalid(
            unqualified,
            draft_binding,
            draft_state_memory,
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
        draft_state_memory,
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
