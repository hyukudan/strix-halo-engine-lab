#include "strix/serial_decode_replay.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace strix {

void validate_serial_decode_engine_logits(
    std::span<const float> logits,
    std::uint64_t expected_vocab_size) {
    if (expected_vocab_size == 0 ||
        expected_vocab_size >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(
            "serial replay expected vocabulary size is invalid");
    }
    if (logits.size() !=
        static_cast<std::size_t>(
            expected_vocab_size)) {
        throw std::invalid_argument(
            "serial replay engine logits do not match expected vocabulary size");
    }
    for (const float value : logits) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "serial replay engine logits contain a non-finite value");
        }
    }
}

SerialDecodeReplayAssessment
assess_serial_decode_replay(
    const SerialDecodeReplayPolicy& policy,
    const SerialDecodeReplayObservation& observation) {
    if (!policy.expected_model_content_identity.valid() ||
        policy.expected_compiled_execution_identity == 0 ||
        policy.expected_runtime_compatibility_identity == 0 ||
        policy.policy_identity == 0 ||
        policy.minimum_steps < 2 ||
        policy.minimum_logit_values_compared == 0 ||
        !std::isfinite(policy.max_logit_abs_error) ||
        policy.max_logit_abs_error < 0.0) {
        throw std::invalid_argument(
            "serial replay policy is invalid");
    }
    if (!std::isfinite(
            observation.maximum_logit_abs_error) ||
        observation.maximum_logit_abs_error < 0.0) {
        throw std::invalid_argument(
            "serial replay observation contains invalid logit error");
    }

    SerialDecodeReplayAssessment result;

    result.identity_valid =
        observation.model_content_identity ==
            policy.expected_model_content_identity &&
        observation.compiled_execution_identity ==
            policy.expected_compiled_execution_identity &&
        observation.runtime_compatibility_identity ==
            policy.expected_runtime_compatibility_identity &&
        observation.independent_reference_identity != 0 &&
        observation.independent_reference_identity ==
            policy.policy_identity;

    result.input_sequence_present =
        !observation.input_tokens.empty();

    result.multi_step =
        observation.input_tokens.size() >=
            policy.minimum_steps &&
        observation.reference_argmax_tokens.size() ==
            observation.input_tokens.size() &&
        observation.engine_argmax_tokens.size() ==
            observation.input_tokens.size();

    result.argmax_checked =
        observation.reference_argmax_tokens.size() ==
            observation.engine_argmax_tokens.size() &&
        !observation.reference_argmax_tokens.empty();

    result.argmax_match =
        result.argmax_checked &&
        observation.reference_argmax_tokens ==
            observation.engine_argmax_tokens;

    result.logits_checked =
        observation.logit_values_compared >=
            policy.minimum_logit_values_compared;

    result.logits_within_policy =
        result.logits_checked &&
        observation.maximum_logit_abs_error <=
            policy.max_logit_abs_error;

    result.passes_declared_policy =
        result.identity_valid &&
        result.input_sequence_present &&
        result.multi_step &&
        result.logits_within_policy &&
        (!policy.require_exact_argmax ||
         result.argmax_match);

    return result;
}

}  // namespace strix
