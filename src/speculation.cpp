#include "strix/speculation.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix {
namespace {

bool is_eos(std::uint32_t token,
            std::span<const std::uint32_t> eos_tokens) noexcept {
    return std::find(eos_tokens.begin(), eos_tokens.end(), token) !=
           eos_tokens.end();
}

}  // namespace

GreedyVerificationPlan verify_greedy_candidates(
    const GreedyVerificationRequest& request) {
    if (request.target_tokens.size() !=
        request.candidate_tokens.size() + 1U) {
        throw std::invalid_argument(
            "target verification tokens must contain candidate_length + 1 entries");
    }
    if (request.token_budget == 0) {
        throw std::invalid_argument(
            "greedy verification token budget must be non-zero");
    }

    GreedyVerificationPlan plan;
    plan.candidate_length =
        static_cast<std::uint32_t>(request.candidate_tokens.size());

    std::size_t raw_matches = 0;
    while (raw_matches < request.candidate_tokens.size() &&
           request.candidate_tokens[raw_matches] ==
               request.target_tokens[raw_matches]) {
        ++raw_matches;
    }

    plan.raw_matches =
        static_cast<std::uint32_t>(raw_matches);
    plan.all_candidates_matched =
        raw_matches == request.candidate_tokens.size();

    // Canonical assisted-greedy result: all contiguous target-selected
    // matches plus exactly one target-selected bonus token at the first
    // mismatch position, or after the final candidate when all match.
    const auto uncropped_count = raw_matches + 1U;
    const auto length_limited_count =
        std::min<std::size_t>(
            uncropped_count,
            request.token_budget);

    plan.stopped_on_length =
        length_limited_count < uncropped_count;

    plan.valid_tokens.assign(
        request.target_tokens.begin(),
        request.target_tokens.begin() +
            static_cast<std::ptrdiff_t>(length_limited_count));

    for (std::size_t index = 0;
         index < plan.valid_tokens.size();
         ++index) {
        if (!is_eos(plan.valid_tokens[index], request.eos_tokens)) {
            continue;
        }
        plan.valid_tokens.resize(index + 1U);
        plan.stopped_on_eos = true;
        break;
    }

    // The assisted-generation contract recomputes n_matches from the cropped
    // number of valid tokens: every cycle keeps one target-produced token
    // beyond the accepted candidate prefix.
    plan.accepted_matches =
        plan.valid_tokens.empty()
            ? 0U
            : static_cast<std::uint32_t>(
                  plan.valid_tokens.size() - 1U);

    if (plan.accepted_matches > plan.candidate_length) {
        throw std::logic_error(
            "accepted candidate matches exceed candidate length");
    }

    plan.candidate_states_to_crop =
        plan.candidate_length - plan.accepted_matches;

    return plan;
}

}  // namespace strix
