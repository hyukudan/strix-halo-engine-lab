#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct GreedyVerificationRequest {
    // Drafted candidate tokens. Length = candidate depth.
    std::span<const std::uint32_t> candidate_tokens;

    // Target greedy selections for every candidate position plus one bonus
    // position. Length MUST be candidate depth + 1.
    std::span<const std::uint32_t> target_tokens;

    // Maximum number of continuation tokens that may be committed from this
    // verification cycle after length accounting. Must be non-zero.
    std::uint32_t token_budget = 1;

    // Any matching token terminates the returned continuation inclusively.
    std::span<const std::uint32_t> eos_tokens;
};

struct GreedyVerificationPlan {
    std::uint32_t candidate_length = 0;

    // Contiguous candidate/target matches before EOS/length cropping.
    std::uint32_t raw_matches = 0;

    // Matches retained after EOS/length cropping. This is the canonical
    // n_matches used to advance committed target state.
    std::uint32_t accepted_matches = 0;

    // Number of speculative candidate positions whose target-side state must
    // be removed/restored.
    std::uint32_t candidate_states_to_crop = 0;

    bool all_candidates_matched = false;
    bool stopped_on_eos = false;
    bool stopped_on_length = false;

    // Accepted matching candidate prefix plus one target-produced bonus token,
    // unless EOS/length cropping truncates the cycle earlier.
    std::vector<std::uint32_t> valid_tokens;
};

[[nodiscard]] GreedyVerificationPlan verify_greedy_candidates(
    const GreedyVerificationRequest& request);

}  // namespace strix
