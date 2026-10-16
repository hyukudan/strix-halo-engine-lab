#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct MtpForwardReplayPolicy {
    std::uint64_t policy_identity = 0;
    double max_hidden_abs_error = 0.0;
    double max_logit_abs_error = 0.0;
    std::uint64_t minimum_steps = 2;
    bool require_exact_tokens = true;
};

struct MtpForwardReplayObservation {
    std::uint64_t proposal_identity = 0;
    std::uint64_t plan_identity = 0;
    std::uint64_t independent_reference_identity = 0;

    std::vector<std::uint32_t> reference_tokens;
    std::vector<std::uint32_t> engine_tokens;

    std::uint64_t hidden_values_compared = 0;
    double maximum_hidden_abs_error = 0.0;

    std::uint64_t logit_values_compared = 0;
    double maximum_logit_abs_error = 0.0;
    // Zero preserves portable/legacy scope. Native entry requires an exact
    // nonzero match to its cast/reduction/kernel arithmetic profile.
    std::uint64_t native_arithmetic_profile_identity = 0;
};

struct MtpForwardReplayAssessment {
    bool identity_valid = false;
    bool multi_step = false;
    bool tokens_match = false;
    bool hidden_checked = false;
    bool hidden_within_policy = false;
    bool logits_checked = false;
    bool logits_within_policy = false;
    bool passes_declared_policy = false;
    std::uint64_t native_arithmetic_profile_identity = 0;
};

[[nodiscard]] MtpForwardReplayAssessment
assess_mtp_forward_replay(
    const MtpForwardReplayPolicy& policy,
    const MtpForwardReplayObservation& observation);

}  // namespace strix
