#include "strix/mtp_forward_replay.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strix {

MtpForwardReplayAssessment
assess_mtp_forward_replay(
    const MtpForwardReplayPolicy& policy,
    const MtpForwardReplayObservation& observation) {
    if (policy.policy_identity == 0) {
        throw std::invalid_argument(
            "MTP replay policy identity must be non-zero");
    }
    if (!std::isfinite(policy.max_hidden_abs_error) ||
        policy.max_hidden_abs_error < 0.0 ||
        !std::isfinite(policy.max_logit_abs_error) ||
        policy.max_logit_abs_error < 0.0 ||
        policy.minimum_steps == 0) {
        throw std::invalid_argument(
            "MTP replay policy limits must be finite, non-negative and non-empty");
    }
    if (!std::isfinite(
            observation.maximum_hidden_abs_error) ||
        observation.maximum_hidden_abs_error < 0.0 ||
        !std::isfinite(
            observation.maximum_logit_abs_error) ||
        observation.maximum_logit_abs_error < 0.0) {
        throw std::invalid_argument(
            "MTP replay observations must contain finite non-negative errors");
    }

    MtpForwardReplayAssessment result;
    result.native_arithmetic_profile_identity=observation.native_arithmetic_profile_identity;
    result.identity_valid =
        observation.proposal_identity != 0 &&
        observation.plan_identity != 0 &&
        observation.independent_reference_identity != 0;

    result.multi_step =
        observation.reference_tokens.size() >=
            policy.minimum_steps &&
        observation.engine_tokens.size() ==
            observation.reference_tokens.size();

    result.tokens_match =
        observation.engine_tokens ==
        observation.reference_tokens;

    result.hidden_checked =
        observation.hidden_values_compared != 0;
    result.hidden_within_policy =
        result.hidden_checked &&
        observation.maximum_hidden_abs_error <=
            policy.max_hidden_abs_error;

    result.logits_checked =
        observation.logit_values_compared != 0;
    result.logits_within_policy =
        result.logits_checked &&
        observation.maximum_logit_abs_error <=
            policy.max_logit_abs_error;

    result.passes_declared_policy =
        result.identity_valid &&
        result.multi_step &&
        (!policy.require_exact_tokens ||
         result.tokens_match) &&
        result.hidden_within_policy &&
        result.logits_within_policy;

    return result;
}

}  // namespace strix
