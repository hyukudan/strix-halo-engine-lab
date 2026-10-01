#pragma once

#include "strix/speculation_executor.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct PromptLookupDraftConfig {
    std::uint32_t ngram_size = 3;
    std::uint64_t history_capacity_tokens = 4096;
};

class PromptLookupDraftBackend final
    : public GreedyDraftProposalBackend {
public:
    PromptLookupDraftBackend(
        PromptLookupDraftConfig config,
        std::span<const std::uint32_t> initial_history = {});

    [[nodiscard]] std::uint64_t
    proposal_identity() const noexcept {
        return proposal_identity_;
    }

    [[nodiscard]] const std::vector<std::uint32_t>&
    history() const noexcept {
        return history_;
    }

    [[nodiscard]] DraftProposalStateContext
    make_state(
        const SerialSessionContext& target_session) const;

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
    void validate_state(
        const DraftProposalStateContext& state,
        DraftProposalStatePhase expected_phase) const;

    void trim_history();

    PromptLookupDraftConfig config_;
    std::vector<std::uint32_t> history_;
    std::uint64_t proposal_identity_ = 0;
};

}  // namespace strix
