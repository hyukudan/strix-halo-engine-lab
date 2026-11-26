#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
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

}  // namespace strix::numeric
