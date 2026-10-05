#pragma once

#include "strix/mtp_draft_activation.hpp"
#include "strix/mtp_draft_cycle_input.hpp"
#include "strix/mtp_forward_runtime.hpp"
#include "strix/qualified_mtp_draft_backend.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

class MtpRuntimeCycleInputProvider {
public:
    virtual ~MtpRuntimeCycleInputProvider() = default;

    [[nodiscard]] virtual MtpDraftCycleInput
    make_cycle_input(
        const QualifiedMtpDraftPlan& qualified_plan,
        const ActivatedMtpDraftPlan& activated_plan,
        const MtpDraftRuntimeBinding& binding,
        const MtpDraftStateMemoryBinding& state_memory,
        const SerialSessionContext& target_session,
        const DraftProposalStateContext& proposal_state,
        std::span<const std::uint32_t> prior_candidates) = 0;

    virtual void reconcile_after_target(
        const QualifiedMtpDraftPlan& qualified_plan,
        const ActivatedMtpDraftPlan& activated_plan,
        const MtpDraftRuntimeBinding& binding,
        const MtpDraftStateMemoryBinding& state_memory,
        const DraftProposalStateContext& proposal_state,
        const SerialSessionContext& target_session,
        std::span<const std::uint32_t> emitted_tokens) = 0;
};

class MtpRuntimeTokenReader {
public:
    virtual ~MtpRuntimeTokenReader() = default;

    [[nodiscard]] virtual std::uint32_t
    read_greedy_token(
        const QualifiedMtpDraftPlan& qualified_plan,
        const MtpDraftCycleInput& cycle_input,
        const MtpForwardExecutionResult& forward_result,
        std::uint32_t candidate_index) = 0;
};

class QualifiedMtpForwardRuntimeAdapter final
    : public QualifiedMtpForwardBackend {
public:
    QualifiedMtpForwardRuntimeAdapter(
        const ModelContract& model,
        const ActivatedMtpDraftPlan& activated_plan,
        MtpForwardExecutionBackend& step_backend,
        MtpRuntimeCycleInputProvider& input_provider,
        MtpRuntimeTokenReader& token_reader);

    [[nodiscard]] std::vector<std::uint32_t>
    propose_mtp_candidates(
        const QualifiedMtpDraftPlan& qualified_plan,
        const MtpDraftRuntimeBinding& binding,
        const MtpDraftStateMemoryBinding& state_memory,
        const SerialSessionContext& target_session,
        const DraftProposalStateContext& proposal_state,
        std::uint32_t max_candidate_depth) override;

    void reconcile_mtp_after_target(
        const QualifiedMtpDraftPlan& qualified_plan,
        const MtpDraftRuntimeBinding& binding,
        const MtpDraftStateMemoryBinding& state_memory,
        const DraftProposalStateContext& proposal_state,
        const SerialSessionContext& target_session,
        std::span<const std::uint32_t> emitted_tokens) override;

private:
    void validate_plan_binding(
        const QualifiedMtpDraftPlan& qualified_plan,
        const MtpDraftRuntimeBinding& binding) const;

    ModelContract model_;
    ActivatedMtpDraftPlan activated_plan_;
    MtpForwardExecutionBackend* step_backend_ = nullptr;
    MtpRuntimeCycleInputProvider* input_provider_ = nullptr;
    MtpRuntimeTokenReader* token_reader_ = nullptr;
};

}  // namespace strix
