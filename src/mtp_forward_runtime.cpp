#include "strix/mtp_forward_runtime.hpp"

#include <stdexcept>

namespace strix {

MtpForwardExecutionResult
execute_qualified_mtp_forward(
    const QualifiedMtpDraftPlan& plan,
    const MtpDraftRuntimeBinding& binding,
    const MtpDraftStateMemoryBinding& state_memory,
    const MtpForwardPhysicalResources& resources,
    const SerialSessionContext& target_session,
    const DraftProposalStateContext& proposal_state,
    std::uint32_t input_token,
    MtpForwardExecutionBackend& backend) {
    require_qualified_mtp_draft_plan(
        plan);

    if (plan.base.forward.contract_identity == 0 ||
        !plan.base.forward.composition_resolved) {
        throw std::logic_error(
            "qualified MTP forward requires resolved composition contract");
    }
    if (binding.compatibility_identity == 0 ||
        binding.physical_binding_identity == 0 ||
        plan.base.proposal_identity !=
            binding.compatibility_identity) {
        throw std::invalid_argument(
            "qualified MTP forward binding does not match plan");
    }
    if (resources.manifest == nullptr ||
        resources.runtime == nullptr) {
        throw std::invalid_argument(
            "qualified MTP forward requires physical model resources");
    }
    validate_mtp_draft_runtime_binding(
        *resources.manifest,
        *resources.runtime,
        binding);
    if (plan.base.structural.vocab_size == 0 ||
        static_cast<std::uint64_t>(
            input_token) >=
            plan.base.structural.vocab_size) {
        throw std::invalid_argument(
            "qualified MTP forward input token is outside vocabulary");
    }

    validate_mtp_draft_proposal_state(
        binding,
        proposal_state,
        target_session);
    if (proposal_state.proposal_identity !=
        plan.base.proposal_identity) {
        throw std::invalid_argument(
            "qualified MTP forward proposal state does not match plan");
    }

    const auto memory_plan =
        plan_mtp_draft_state_memory(
            binding.compatibility_identity,
            plan.base.state,
            state_memory.rows,
            state_memory.element_bytes,
            state_memory.alignment);
    validate_mtp_draft_state_memory_binding(
        memory_plan,
        state_memory);

    if (plan.base.steps.empty()) {
        throw std::logic_error(
            "qualified MTP forward plan contains no steps");
    }

    MtpForwardExecutionResult result;
    for (std::size_t index = 0;
         index < plan.base.steps.size();
         ++index) {
        MtpForwardStepContext context;
        context.step =
            plan.base.steps[index];
        context.step_index =
            static_cast<std::uint32_t>(
                index);
        context.input_token =
            input_token;
        context.target_position =
            target_session.next_position;
        context.proposal_identity =
            plan.base.proposal_identity;
        context.plan_identity =
            plan.base.plan_identity;
        context.binding =
            &binding;
        context.state_memory =
            &state_memory;
        context.manifest =
            resources.manifest;
        context.runtime =
            resources.runtime;

        backend.dispatch_mtp_step(
            context);
        ++result.completed_steps;

        if (context.step ==
            MtpDraftPlanStep::publish_next_state) {
            result.next_state_published =
                true;
        }
        if (context.step ==
            MtpDraftPlanStep::lm_head) {
            result.logits_published =
                true;
        }
    }

    if (!result.next_state_published ||
        !result.logits_published) {
        throw std::logic_error(
            "qualified MTP forward did not publish required outputs");
    }

    return result;
}

}  // namespace strix
