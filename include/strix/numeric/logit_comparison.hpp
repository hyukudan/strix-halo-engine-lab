#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::numeric {

struct LogitComparison {
    double kl_reference_to_candidate;
    std::size_t reference_top1, candidate_top1;
    [[nodiscard]] bool top1_agrees() const noexcept { return reference_top1 == candidate_top1; }
};

// Full-vocabulary KL(reference || candidate) and greedy IDs at the same
// input position. Ties choose the lowest token ID.
[[nodiscard]] inline LogitComparison compare_logits(std::span<const float> reference, std::span<const float> candidate) {
    const auto finite = [](float value) { return std::isfinite(value); };
    if (reference.empty() || reference.size() != candidate.size() ||
        !std::ranges::all_of(reference, finite) || !std::ranges::all_of(candidate, finite))
        throw std::invalid_argument("logit comparison needs two finite rows of equal nonzero width");
    const auto ref_max = std::ranges::max_element(reference);
    const auto cand_max = std::ranges::max_element(candidate);
    double ref_mass = 0.0, cand_mass = 0.0, weighted_delta = 0.0;
    for (std::size_t id = 0; id < reference.size(); ++id) {
        const double ref = static_cast<double>(reference[id]) - *ref_max;
        const double cand = static_cast<double>(candidate[id]) - *cand_max;
        const double mass = std::exp(ref);
        ref_mass += mass;
        cand_mass += std::exp(cand);
        weighted_delta += mass * (ref - cand);
    }
    return {std::max(0.0, weighted_delta / ref_mass + std::log(cand_mass / ref_mass)),
            static_cast<std::size_t>(ref_max - reference.begin()), static_cast<std::size_t>(cand_max - candidate.begin())};
}

struct NextTokenScore {
    double nll;
    std::uint32_t best_id;
};

// Negative log-likelihood of `target` and the greedy ID of one logits row
// (stable log-sum-exp; ties choose the lowest token ID).
[[nodiscard]] inline NextTokenScore score_next_token(std::span<const float> logits, std::uint32_t target) {
    if (target >= logits.size()) throw std::out_of_range("next-token score: target outside vocabulary");
    if (!std::ranges::all_of(logits, [](float value) { return std::isfinite(value); }))
        throw std::invalid_argument("next-token score: nonfinite logits");
    const auto best = std::ranges::max_element(logits);
    double partition = 0;
    for (float value : logits) partition += std::exp(double{value} - *best);
    return {std::log(partition) + (double{*best} - logits[target]),
            static_cast<std::uint32_t>(best - logits.begin())};
}

}  // namespace strix::numeric
