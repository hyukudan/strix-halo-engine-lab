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

struct RuntimeFixture {
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding runtime;
    strix::MtpDraftRuntimeBinding binding;
};

RuntimeFixture make_runtime_fixture() {
    RuntimeFixture value;
    value.manifest.architecture = "fixture";
    strix::ManifestTensor tensor;
    tensor.name = "mtp.fixture.weight";
    tensor.dimensions = {1};
    tensor.ggml_type = 30;
    tensor.byte_size = 4096;
    tensor.absolute_offset = 4096;
    tensor.role = strix::TensorRole::mtp;
    value.manifest.tensors.push_back(tensor);
    value.manifest.total_tensor_bytes = tensor.byte_size;
    value.manifest.layout_identity =
        strix::model_manifest_layout_identity(
            value.manifest);

    value.runtime.manifest_layout_identity =
        value.manifest.layout_identity;
    value.runtime.model_content_identity.bytes[0] = 1;
    strix::RuntimeTensorBinding runtime_tensor;
    runtime_tensor.tensor_index = 0;
    runtime_tensor.storage =
        strix::RuntimeTensorStorage::resident;
    runtime_tensor.address =
        static_cast<std::uintptr_t>(
            0x50000000ULL);
    runtime_tensor.bytes = 4096;
    runtime_tensor.allocation_generation = 1;
    value.runtime.tensors.push_back(
        runtime_tensor);

    value.binding =
        strix::build_mtp_draft_runtime_binding(
            value.manifest,
            value.runtime);
    return value;
}

strix::QualifiedMtpDraftPlan make_plan(
    const strix::MtpDraftRuntimeBinding& binding) {
    strix::QualifiedMtpDraftPlan plan;
    plan.base.proposal_identity =
        binding.compatibility_identity;
    plan.base.contract_identity = 0x3333ULL;
    plan.base.structural =
        strix::build_mtp_draft_structural_contract(
            strix::ModelContract::
                qwen38_flash_next(),
            binding);
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
    plan.base.plan_identity =
        strix::recompute_compiled_mtp_draft_plan_identity(
            plan.base);
    plan.replay_policy.policy_identity =
        0x4444ULL;
    plan.replay_assessment.passes_declared_policy =
        true;
    plan.independent_reference_identity =
        0x5555ULL;
    plan.qualified_plan_identity =
        plan.base.plan_identity;
    plan.qualification_identity =
        0x6666ULL;
    plan.executable = true;
    return plan;
}

strix::ActivatedMtpDraftPlan make_activation(
    const strix::QualifiedMtpDraftPlan& plan) {
    strix::ActivatedMtpDraftPlan value;
    value.structural_plan_identity =
        plan.base.plan_identity;
    value.proposal_identity =
        plan.base.proposal_identity;
    value.replay_policy_identity =
        0x4444ULL;
    value.independent_reference_identity =
        0x5555ULL;
    value.activation_identity =
        0x8888ULL;
    value.semantics_qualified = true;
    return value;
}

strix::MtpDraftStateMemoryBinding make_memory(
    const strix::MtpDraftStateContract& state,
    std::uint64_t proposal_identity) {
    strix::MtpDraftStateMemoryBinding value;
    value.proposal_identity =
        proposal_identity;
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
    std::vector<std::uint64_t> positions;

    void dispatch_mtp_step(
        const strix::MtpForwardStepContext& context) override {
        ++calls;
        positions.push_back(
            context.target_position);
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
    auto physical =
        make_runtime_fixture();
    const auto& binding =
        physical.binding;
    auto plan =
        make_plan(
            binding);
    const auto activation =
        make_activation(plan);
    const auto memory =
        make_memory(
            plan.base.state,
            binding.compatibility_identity);
    const strix::MtpForwardPhysicalResources
        resources{
            &physical.manifest,
            &physical.runtime,
        };
    const auto target =
        make_target();
    auto proposal_state =
        strix::make_draft_proposal_state(
            binding.compatibility_identity,
            target);
    strix::begin_draft_proposal(proposal_state,target);

    StepBackend steps;
    InputProvider inputs;
    TokenReader tokens;
    strix::QualifiedMtpForwardRuntimeAdapter
        adapter(
            model,
            activation,
            resources,
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
    expect(
        steps.positions ==
            std::vector<std::uint64_t>({
                10, 10, 10,
                11, 11, 11,
            }),
        "successive MTP candidates dispatch at successive absolute seed positions");

    strix::finish_draft_proposal(proposal_state);
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
                resources,
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
