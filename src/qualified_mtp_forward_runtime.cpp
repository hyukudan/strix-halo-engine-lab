#include "strix/qualified_mtp_forward_runtime.hpp"

#include <stdexcept>

namespace strix {

QualifiedMtpForwardRuntimeAdapter::
QualifiedMtpForwardRuntimeAdapter(
    const ModelContract& model,
    const ActivatedMtpDraftPlan& activated_plan,
    MtpForwardExecutionBackend& step_backend,
    MtpRuntimeCycleInputProvider& input_provider,
    MtpRuntimeTokenReader& token_reader)
    : model_(model),
      activated_plan_(activated_plan),
      step_backend_(&step_backend),
      input_provider_(&input_provider),
      token_reader_(&token_reader) {
    if (const auto error = model_.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (!activated_plan_.semantics_qualified ||
        activated_plan_.activation_identity == 0 ||
        activated_plan_.structural_plan_identity == 0 ||
        activated_plan_.proposal_identity == 0) {
        throw std::invalid_argument(
            "qualified MTP runtime adapter requires semantic activation");
    }
}

void QualifiedMtpForwardRuntimeAdapter::
validate_plan_binding(
    const QualifiedMtpDraftPlan& qualified_plan,
    const MtpDraftRuntimeBinding& binding) const {
    require_qualified_mtp_draft_plan(
        qualified_plan);

    if (activated_plan_.structural_plan_identity !=
            qualified_plan.base.plan_identity ||
        activated_plan_.proposal_identity !=
            qualified_plan.base.proposal_identity ||
        binding.compatibility_identity !=
            qualified_plan.base.proposal_identity) {
        throw std::invalid_argument(
            "qualified MTP runtime adapter plan/activation/binding identities do not match");
    }
}

std::vector<std::uint32_t>
QualifiedMtpForwardRuntimeAdapter::
propose_mtp_candidates(
    const QualifiedMtpDraftPlan& qualified_plan,
    const MtpDraftRuntimeBinding& binding,
    const MtpDraftStateMemoryBinding& state_memory,
    const SerialSessionContext& target_session,
    const DraftProposalStateContext& proposal_state,
    std::uint32_t max_candidate_depth) {
    validate_plan_binding(
        qualified_plan,
        binding);

    std::vector<std::uint32_t> candidates;
    candidates.reserve(
        max_candidate_depth);

    for (std::uint32_t index = 0;
         index < max_candidate_depth;
         ++index) {
        const auto cycle_input =
            input_provider_->make_cycle_input(
                qualified_plan,
                activated_plan_,
                binding,
                state_memory,
                target_session,
                proposal_state,
                candidates);

        validate_mtp_draft_cycle_input(
            model_,
            activated_plan_,
            qualified_plan.base.state,
            target_session,
            cycle_input);

        const auto forward =
            execute_qualified_mtp_forward(
                qualified_plan,
                binding,
                state_memory,
                target_session,
                proposal_state,
                cycle_input.seed_token_id,
                *step_backend_);

        if (!forward.next_state_published ||
            !forward.logits_published) {
            throw std::logic_error(
                "qualified MTP forward runtime did not publish candidate outputs");
        }

        const auto token =
            token_reader_->read_greedy_token(
                qualified_plan,
                cycle_input,
                forward,
                index);

        if (qualified_plan.base.structural.vocab_size == 0 ||
            static_cast<std::uint64_t>(
                token) >=
                qualified_plan.base.structural.vocab_size) {
            throw std::runtime_error(
                "qualified MTP runtime token reader produced out-of-vocabulary token");
        }

        candidates.push_back(
            token);
    }

    return candidates;
}

void QualifiedMtpForwardRuntimeAdapter::
reconcile_mtp_after_target(
    const QualifiedMtpDraftPlan& qualified_plan,
    const MtpDraftRuntimeBinding& binding,
    const MtpDraftStateMemoryBinding& state_memory,
    const DraftProposalStateContext& proposal_state,
    const SerialSessionContext& target_session,
    std::span<const std::uint32_t> emitted_tokens) {
    validate_plan_binding(
        qualified_plan,
        binding);

    input_provider_->reconcile_after_target(
        qualified_plan,
        activated_plan_,
        binding,
        state_memory,
        proposal_state,
        target_session,
        emitted_tokens);
}

}  // namespace strix
