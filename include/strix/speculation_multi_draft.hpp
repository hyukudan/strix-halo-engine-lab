#pragma once

#include "strix/token_sampling.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace strix {

namespace detail {

[[nodiscard]] double rejected_target_mass(const TokenDistribution& target, const TokenDistribution& draft);
[[nodiscard]] double binary64_rejected_target_mass(const TokenDistribution& target, const TokenDistribution& draft);

}  // namespace detail

struct SpeculationTreeNode;

struct SpeculationTreeCandidate {
    TokenDraft draft;
    const SpeculationTreeNode* child = nullptr;
};

struct SpeculationTreeNode {
    // The target row is normalized over the prefix represented by this node.
    const TokenDistribution& target;
    std::span<const SpeculationTreeCandidate> candidates;
    std::uint32_t first_draw_index = 0;
};

struct MultiDraftDecision {
    std::uint32_t token = 0;
    std::optional<std::size_t> accepted_candidate;
};

// Each draft token must be an independent proposal from its normalized q at
// absolute_position. Siblings may use different q rows or propose the same id.
// Draw index first_draw_index + i belongs to candidate i in both draft_proposal
// and draft_acceptance. The last candidate also owns that index in residual_fallback.
// At a depth with prebuilt siblings, assign their child nodes disjoint index
// ranges for proposals at the same position. An empty row uses serial_target at 0.
[[nodiscard]] MultiDraftDecision verify_multi_draft_row(
    const SpeculationTreeNode& node, const SpeculationRngIdentity& rng,
    std::uint64_t absolute_position);

// Follow accepted children, stopping at a fallback token, a leaf, EOS or budget.
// A node with no candidates samples its own target row as a bonus.
[[nodiscard]] std::vector<std::uint32_t> verify_speculative_tree(
    const SpeculationTreeNode& root, const SpeculationRngIdentity& rng,
    std::uint64_t absolute_position, std::uint32_t token_budget,
    std::span<const std::uint32_t> eos_tokens = {});

}  // namespace strix
