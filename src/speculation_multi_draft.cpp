#include "strix/speculation_multi_draft.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

template <typename Float>
double charge_coverage(double target, double draft, Float remaining, Float& covered) {
    covered = std::min(static_cast<Float>(target), covered + remaining * draft);
    return static_cast<double>(target - covered);
}

template <typename Float>
double sum_rejected_target_mass(const TokenDistribution& target, const TokenDistribution& draft) {
    const auto ids = target.tokens();
    const auto probabilities = target.probabilities();
    Float mass = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        Float covered = 0;
        mass += charge_coverage(probabilities[i], draft.probability(ids[i]), Float{1}, covered);
    }
    return static_cast<double>(mass);
}

}  // namespace

double detail::rejected_target_mass(const TokenDistribution& target, const TokenDistribution& draft) {
    return sum_rejected_target_mass<long double>(target, draft);
}

double detail::binary64_rejected_target_mass(const TokenDistribution& target, const TokenDistribution& draft) {
    return sum_rejected_target_mass<double>(target, draft);
}

MultiDraftDecision verify_multi_draft_row(
    const SpeculationTreeNode& node, const SpeculationRngIdentity& rng,
    std::uint64_t absolute_position) {
    using Draw = SpeculationRandomDrawClass;
    if (node.candidates.empty()) {
        return {decide_token(node.target, nullptr,
                    speculation_uniform01(rng, absolute_position, Draw::serial_target), 0.0, 0.0).token,
                std::nullopt};
    }
    if (node.candidates.size() - 1U > UINT32_MAX - node.first_draw_index) {
        throw std::overflow_error("multi-draft draw index overflow");
    }

    const auto ids = node.target.tokens();
    const auto original = node.target.probabilities();
    std::vector<long double> coverage;
    std::vector<double> uncovered;
    long double remaining = 1.0L;
    TokenDistribution conditioned;
    const TokenDistribution* current = &node.target;
    for (std::size_t i = 0; i < node.candidates.size(); ++i) {
        const auto& proposal = node.candidates[i].draft;
        if (proposal.position != absolute_position || proposal.distribution.empty()) {
            throw std::invalid_argument("multi-draft candidate requires a host distribution at its row position");
        }
        const auto index = node.first_draw_index + static_cast<std::uint32_t>(i);
        const bool last = i + 1U == node.candidates.size();
        const auto acceptance = speculation_uniform01(rng, absolute_position, Draw::draft_acceptance, index);
        const auto fallback = last ? speculation_uniform01(rng, absolute_position, Draw::residual_fallback, index) : 0.0;
        const auto decision = decide_token(*current, &proposal, 0.0, acceptance, fallback);
        if (decision.accepted || last) {
            return {decision.token, decision.accepted ? std::optional<std::size_t>{i} : std::nullopt};
        }

        // Keep the cumulative draft coverage in original target units.
        // If a row has survival mass S, subtracting its normalized draft q
        // is equivalent to adding S*q to coverage, then clipping at the
        // original target mass. Previously covered mass cannot reappear.
        if (coverage.empty()) {
            coverage.resize(ids.size());
            uncovered.resize(ids.size());
        }
        long double next_remaining = 0.0L;
        for (std::size_t j = 0; j < ids.size(); ++j) {
            uncovered[j] = charge_coverage(original[j], proposal.distribution.probability(ids[j]),
                remaining, coverage[j]);
            next_remaining += uncovered[j];
        }
        if (!(next_remaining > 0.0L)) {
            throw std::logic_error("rejected multi-draft candidate has no surviving target mass");
        }
        for (auto& weight : uncovered) {
            weight = static_cast<double>(weight / next_remaining);
        }
        remaining = next_remaining;
        conditioned = make_token_distribution_from_probabilities(ids, uncovered);
        current = &conditioned;
    }
    throw std::logic_error("multi-draft verification did not finish");
}

std::vector<std::uint32_t> verify_speculative_tree(
    const SpeculationTreeNode& root, const SpeculationRngIdentity& rng,
    std::uint64_t absolute_position, std::uint32_t token_budget,
    std::span<const std::uint32_t> eos_tokens) {
    std::vector<std::uint32_t> emitted;
    const auto* node = &root;
    while (node && emitted.size() < token_budget) {
        const auto picked = verify_multi_draft_row(*node, rng, absolute_position);
        emitted.push_back(picked.token);
        if (!picked.accepted_candidate || emitted.size() == token_budget ||
            std::find(eos_tokens.begin(), eos_tokens.end(), picked.token) != eos_tokens.end()) {
            break;
        }
        node = node->candidates[*picked.accepted_candidate].child;
        if (node) {
            if (absolute_position == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("speculative tree position overflow");
            }
            ++absolute_position;
        }
    }
    return emitted;
}

}  // namespace strix
