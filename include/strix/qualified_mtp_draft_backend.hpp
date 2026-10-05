#pragma once

#include "strix/mtp_draft_binding.hpp"
#include "strix/mtp_draft_qualification.hpp"
#include "strix/mtp_draft_state_memory.hpp"
#include "strix/speculation_executor.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

class QualifiedMtpForwardBackend {
public:
    virtual ~QualifiedMtpForwardBackend() = default;

    [[nodiscard]] virtual std::vector<std::uint32_t>
    propose_mtp_candidates(
        const QualifiedMtpDraftPlan& qualified_plan,
        const MtpDraftRuntimeBinding& binding,
        const MtpDraftStateMemoryBinding& state_memory,
        const SerialSessionContext& target_session,
        const DraftProposalStateContext& proposal_state,
        std::uint32_t max_candidate_depth) = 0;

    virtual void reconcile_mtp_after_target(
        const QualifiedMtpDraftPlan& qualified_plan,
        const MtpDraftRuntimeBinding& binding,
        const MtpDraftStateMemoryBinding& state_memory,
        const DraftProposalStateContext& proposal_state,
        const SerialSessionContext& target_session,
        std::span<const std::uint32_t> emitted_tokens) = 0;
};

class QualifiedMtpDraftBackend final
    : public GreedyDraftProposalBackend {
public:
    QualifiedMtpDraftBackend(
        const QualifiedMtpDraftPlan& qualified_plan,
        const MtpDraftRuntimeBinding& binding,
        const MtpDraftStateMemoryBinding& state_memory,
        QualifiedMtpForwardBackend& backend);

    [[nodiscard]] std::vector<std::uint32_t>
    propose_candidates(
        const SerialSessionContext& target_session,
        const DraftProposalStateContext& proposal_state,
        std::uint32_t max_candidate_depth) override;

    void reconcile_after_target(
        const DraftProposalStateContext& proposal_state,
        const SerialSessionContext& target_session,
        std::span<const std::uint32_t> emitted_tokens) override;

private:
    void validate_request(
        const SerialSessionContext& target_session,
        const DraftProposalStateContext& proposal_state) const;

    const QualifiedMtpDraftPlan& qualified_plan_;
    const MtpDraftRuntimeBinding& binding_;
    const MtpDraftStateMemoryBinding& state_memory_;
    QualifiedMtpForwardBackend& backend_;
};

}  // namespace strix
