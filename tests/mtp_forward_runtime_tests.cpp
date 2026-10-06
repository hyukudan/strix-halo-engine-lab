#include "strix/mtp_forward_runtime.hpp"

#include <iostream>
#include <optional>
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

class RecordingBackend final
    : public strix::MtpForwardExecutionBackend {
public:
    void dispatch_mtp_step(
        const strix::MtpForwardStepContext& context) override {
        expect(context.binding != nullptr &&
                   context.state_memory != nullptr,
               "MTP forward step receives binding and state memory");
        expect(context.step_index == steps.size(),
               "MTP forward step index is contiguous");
        steps.push_back(context.step);
        if (fail_at.has_value() &&
            context.step_index ==
                *fail_at) {
            throw std::runtime_error(
                "injected MTP step failure");
        }
    }

    std::vector<strix::MtpDraftPlanStep> steps;
    std::optional<std::uint32_t> fail_at;
};

strix::QualifiedMtpDraftPlan qualified_fixture() {
    strix::QualifiedMtpDraftPlan plan;
    plan.base.proposal_identity = 0x1234ULL;
    plan.base.structural.vocab_size = 100;
    plan.base.state.state_width = 16;
    plan.base.state.one_state_row_per_request = true;
    plan.base.forward.contract_identity = 0x9999ULL;
    plan.base.forward.composition_resolved = true;
    plan.base.steps = {
        strix::MtpDraftPlanStep::input_fusion,
        strix::MtpDraftPlanStep::attention_hyper_mix,
        strix::MtpDraftPlanStep::qsa_indexer,
        strix::MtpDraftPlanStep::sparse_attention,
        strix::MtpDraftPlanStep::attention_combine,
        strix::MtpDraftPlanStep::mlp_hyper_mix,
        strix::MtpDraftPlanStep::routed_and_shared_moe,
        strix::MtpDraftPlanStep::mlp_combine,
        strix::MtpDraftPlanStep::publish_next_state,
        strix::MtpDraftPlanStep::output_head_mix,
        strix::MtpDraftPlanStep::lm_head,
    };
    plan.base.plan_identity =
        strix::recompute_compiled_mtp_draft_plan_identity(
            plan.base);
    plan.replay_policy.policy_identity = 1;
    plan.replay_assessment.identity_valid = true;
    plan.replay_assessment.multi_step = true;
    plan.replay_assessment.tokens_match = true;
    plan.replay_assessment.hidden_checked = true;
    plan.replay_assessment.hidden_within_policy = true;
    plan.replay_assessment.logits_checked = true;
    plan.replay_assessment.logits_within_policy = true;
    plan.replay_assessment.passes_declared_policy = true;
    plan.independent_reference_identity = 2;
    plan.qualified_plan_identity =
        plan.base.plan_identity;
    plan.qualification_identity = 3;
    plan.executable = true;
    return plan;
}

}  // namespace

int main() {
    auto plan =
        qualified_fixture();

    strix::MtpDraftRuntimeBinding binding;
    binding.compatibility_identity =
        plan.base.proposal_identity;
    binding.model_content_identity.bytes[0] =
        0x42;

    const auto memory_plan =
        strix::plan_mtp_draft_state_memory(
            binding.compatibility_identity,
            plan.base.state,
            1,
            2,
            256);
    strix::MtpDraftStateMemoryBinding memory;
    memory.plan_identity =
        memory_plan.plan_identity;
    memory.proposal_identity =
        memory_plan.proposal_identity;
    memory.state_width =
        memory_plan.state_width;
    memory.rows =
        memory_plan.rows;
    memory.element_bytes =
        memory_plan.element_bytes;
    memory.alignment =
        memory_plan.alignment;
    memory.device_base =
        static_cast<std::uintptr_t>(
            0x10000000ULL);
    memory.bytes =
        memory_plan.total_bytes;
    memory.allocation_generation = 1;

    strix::SerialSessionContext target;
    target.execution_identity = 1;
    target.runtime_compatibility_identity =
        0x8888ULL;
    target.state_layout_identity = 2;
    target.next_position = 5;
    target.committed_tokens = 5;

    auto proposal =
        strix::make_draft_proposal_state(
            binding.compatibility_identity,
            target);

    RecordingBackend backend;
    const auto result =
        strix::execute_qualified_mtp_forward(
            plan,
            binding,
            memory,
            target,
            proposal,
            7,
            backend);

    expect(result.completed_steps ==
               plan.base.steps.size() &&
               result.next_state_published &&
               result.logits_published,
           "qualified MTP forward dispatches the complete compiled step sequence");
    expect(backend.steps ==
               plan.base.steps,
           "qualified MTP runtime preserves compiled step order exactly");

    RecordingBackend failing;
    failing.fail_at = 3;
    bool failure_observed = false;
    try {
        static_cast<void>(
            strix::execute_qualified_mtp_forward(
                plan,
                binding,
                memory,
                target,
                proposal,
                7,
                failing));
    } catch (const std::runtime_error&) {
        failure_observed = true;
    }
    expect(failure_observed &&
               failing.steps.size() == 4,
           "MTP step failure stops later forward dispatch");

    auto foreign = binding;
    ++foreign.compatibility_identity;
    bool foreign_rejected = false;
    try {
        static_cast<void>(
            strix::execute_qualified_mtp_forward(
                plan,
                foreign,
                memory,
                target,
                proposal,
                7,
                backend));
    } catch (const std::invalid_argument&) {
        foreign_rejected = true;
    }
    expect(foreign_rejected,
           "qualified MTP runtime rejects foreign proposal binding");

    if (failures == 0) {
        std::cout << "MTP forward runtime tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
