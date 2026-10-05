#include "strix/qualified_mtp_forward_runtime.hpp"

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

strix::QualifiedMtpDraftPlan make_plan() {
    strix::QualifiedMtpDraftPlan plan;
    plan.base.proposal_identity = 0x1111ULL;
    plan.base.plan_identity = 0x2222ULL;
    plan.base.contract_identity = 0x3333ULL;
    plan.base.structural =
        strix::build_mtp_draft_structural_contract(
            strix::ModelContract::
                qwen38_flash_next());
    plan.base.state =
        strix::build_mtp_draft_state_contract(
            strix::ModelContract::
                qwen38_flash_next());
    plan.base.forward.contract_identity =
        0x7777ULL;
    plan.base.forward.composition_resolved =
        true;
    plan.base.steps = {
        strix::MtpDraftPlanStep::input_fusion,
        strix::MtpDraftPlanStep::publish_next_state,
        strix::MtpDraftPlanStep::lm_head,
    };
    plan.replay_policy.policy_identity =
        0x4444ULL;
    plan.replay_assessment.passes_declared_policy =
        true;
    plan.independent_reference_identity =
        0x5555ULL;
    plan.qualification_identity =
        0x6666ULL;
    plan.executable = true;
    return plan;
}

strix::ActivatedMtpDraftPlan make_activation() {
    strix::ActivatedMtpDraftPlan value;
    value.structural_plan_identity =
        0x2222ULL;
    value.proposal_identity =
        0x1111ULL;
    value.replay_policy_identity =
        0x4444ULL;
    value.independent_reference_identity =
        0x5555ULL;
    value.activation_identity =
        0x8888ULL;
    value.semantics_qualified = true;
    return value;
}

strix::MtpDraftRuntimeBinding make_binding() {
    strix::MtpDraftRuntimeBinding value;
    value.compatibility_identity =
        0x1111ULL;
    value.physical_binding_identity =
        0x9999ULL;
    value.model_content_identity.bytes[0] = 1;
    value.tensor_indices = {0};
    value.total_tensor_bytes = 4096;
    return value;
}

strix::MtpDraftStateMemoryBinding make_memory(
    const strix::MtpDraftStateContract& state) {
    strix::MtpDraftStateMemoryBinding value;
    value.proposal_identity =
        0x1111ULL;
    value.state_width =
        state.state_width;
    value.rows = 1;
    value.element_bytes = 2;
    value.alignment = 256;
    value.device_base =
        static_cast<std::uintptr_t>(
            0x60000000ULL);
    value.bytes =
        value.state_width *
        value.element_bytes;
    value.allocation_generation = 1;
    const auto plan =
        strix::plan_mtp_draft_state_memory(
            value.proposal_identity,
            state,
            value.rows,
            value.element_bytes,
            value.alignment);
    value.plan_identity =
        plan.plan_identity;
    return value;
}

strix::SerialSessionContext make_target() {
    strix::SerialSessionContext session;
    session.runtime_compatibility_identity =
        0xaaaaULL;
    session.next_position = 10;
    session.committed_tokens = 10;
    session.health =
        strix::SerialSessionHealth::ready;
    return session;
}

struct StepBackend final
    : strix::MtpForwardExecutionBackend {
    std::uint32_t calls = 0;

    void dispatch_mtp_step(
        const strix::MtpForwardStepContext&) override {
        ++calls;
    }
};

struct InputProvider final
    : strix::MtpRuntimeCycleInputProvider {
    std::uint32_t calls = 0;
    std::uint32_t reconcile_calls = 0;
    std::uintptr_t hidden =
        static_cast<std::uintptr_t>(
            0x70000000ULL);

    strix::MtpDraftCycleInput
    make_cycle_input(
        const strix::QualifiedMtpDraftPlan&,
        const strix::ActivatedMtpDraftPlan& activated,
        const strix::MtpDraftRuntimeBinding&,
        const strix::MtpDraftStateMemoryBinding&,
        const strix::SerialSessionContext& target,
        const strix::DraftProposalStateContext&,
        std::span<const std::uint32_t> prior) override {
        strix::MtpDraftCycleInput input;
        input.proposal_identity =
            activated.proposal_identity;
        input.activation_identity =
            activated.activation_identity;
        input.target_runtime_compatibility_identity =
            target.runtime_compatibility_identity;
        input.seed_token_id =
            prior.empty()
                ? 7U
                : prior.back();
        input.seed_position =
            target.next_position +
            prior.size();
        input.hidden_source_position =
            input.seed_position == 0
                ? 0
                : input.seed_position - 1;
        input.hidden_address =
            hidden;
        input.hidden_elements =
            10240;
        input.hidden_element_bytes = 2;
        input.hidden_allocation_generation = 1;
        ++calls;
        return input;
    }

    void reconcile_after_target(
        const strix::QualifiedMtpDraftPlan&,
        const strix::ActivatedMtpDraftPlan&,
        const strix::MtpDraftRuntimeBinding&,
        const strix::MtpDraftStateMemoryBinding&,
        const strix::DraftProposalStateContext&,
        const strix::SerialSessionContext&,
        std::span<const std::uint32_t>) override {
        ++reconcile_calls;
    }
};

struct TokenReader final
    : strix::MtpRuntimeTokenReader {
    std::uint32_t calls = 0;

    std::uint32_t read_greedy_token(
        const strix::QualifiedMtpDraftPlan&,
        const strix::MtpDraftCycleInput&,
        const strix::MtpForwardExecutionResult&,
        std::uint32_t candidate_index) override {
        ++calls;
        return 20U + candidate_index;
    }
};

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();
    auto plan = make_plan();
    const auto activation =
        make_activation();
    const auto binding =
        make_binding();
    const auto memory =
        make_memory(
            plan.base.state);
    const auto target =
        make_target();
    const auto proposal_state =
        strix::make_draft_proposal_state(
            binding.compatibility_identity,
            target);

    StepBackend steps;
    InputProvider inputs;
    TokenReader tokens;
    strix::QualifiedMtpForwardRuntimeAdapter
        adapter(
            model,
            activation,
            steps,
            inputs,
            tokens);

    const auto proposed =
        adapter.propose_mtp_candidates(
            plan,
            binding,
            memory,
            target,
            proposal_state,
            2);

    expect(
        proposed ==
            std::vector<std::uint32_t>({20, 21}) &&
            inputs.calls == 2 &&
            tokens.calls == 2 &&
            steps.calls == 6,
        "runtime adapter executes qualified forward once per candidate and reads each token");

    adapter.reconcile_mtp_after_target(
        plan,
        binding,
        memory,
        proposal_state,
        target,
        proposed);
    expect(
        inputs.reconcile_calls == 1,
        "runtime adapter delegates draft-state reconciliation after target commit");

    auto foreign_activation =
        activation;
    ++foreign_activation.proposal_identity;
    bool foreign_rejected = false;
    try {
        StepBackend other_steps;
        InputProvider other_inputs;
        TokenReader other_tokens;
        strix::QualifiedMtpForwardRuntimeAdapter
            invalid(
                model,
                foreign_activation,
                other_steps,
                other_inputs,
                other_tokens);
        static_cast<void>(
            invalid.propose_mtp_candidates(
                plan,
                binding,
                memory,
                target,
                proposal_state,
                1));
    } catch (const std::invalid_argument&) {
        foreign_rejected = true;
    }
    expect(
        foreign_rejected,
        "runtime adapter rejects activation from another proposal identity");

    if (failures == 0) {
        std::cout
            << "qualified MTP forward runtime adapter tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
