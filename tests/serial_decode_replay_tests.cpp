#include "strix/serial_decode_replay.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::SerialDecodeReplayObservation passing_observation() {
    strix::SerialDecodeReplayObservation observation;
    observation.model_content_identity.bytes[0] = 0x42;
    observation.compiled_execution_identity = 0x1001ULL;
    observation.runtime_compatibility_identity = 0x1002ULL;
    observation.independent_reference_identity = 0x1003ULL;
    observation.input_tokens = {11, 22, 33};
    observation.reference_argmax_tokens = {44, 55, 66};
    observation.engine_argmax_tokens =
        observation.reference_argmax_tokens;
    observation.logit_values_compared =
        3ULL * 248320ULL;
    observation.maximum_logit_abs_error = 5e-4;
    return observation;
}

}  // namespace

int main() {
    strix::SerialDecodeReplayPolicy policy;
    policy.policy_identity = 0x2001ULL;
    policy.max_logit_abs_error = 1e-3;
    policy.minimum_steps = 2;
    policy.require_exact_argmax = true;

    const auto observation =
        passing_observation();
    const auto passing =
        strix::assess_serial_decode_replay(
            policy,
            observation);

    expect(passing.identity_valid &&
               passing.multi_step &&
               passing.argmax_match &&
               passing.logits_within_policy &&
               passing.passes_declared_policy,
           "matching multi-step replay passes declared serial policy");

    auto token_mismatch = observation;
    token_mismatch.engine_argmax_tokens[1] = 56;
    const auto mismatch =
        strix::assess_serial_decode_replay(
            policy,
            token_mismatch);
    expect(!mismatch.argmax_match &&
               !mismatch.passes_declared_policy,
           "argmax mismatch fails exact-token replay policy");

    auto bad_logits = observation;
    bad_logits.maximum_logit_abs_error = 2e-3;
    const auto excessive_error =
        strix::assess_serial_decode_replay(
            policy,
            bad_logits);
    expect(!excessive_error.logits_within_policy &&
               !excessive_error.passes_declared_policy,
           "logit error above policy fails replay");

    auto no_logits = observation;
    no_logits.logit_values_compared = 0;
    no_logits.maximum_logit_abs_error = 0.0;
    const auto unchecked =
        strix::assess_serial_decode_replay(
            policy,
            no_logits);
    expect(!unchecked.logits_checked &&
               !unchecked.passes_declared_policy,
           "zero compared logits cannot qualify serial replay");

    auto wrong_identity = observation;
    wrong_identity.model_content_identity = {};
    const auto unidentified =
        strix::assess_serial_decode_replay(
            policy,
            wrong_identity);
    expect(!unidentified.identity_valid &&
               !unidentified.passes_declared_policy,
           "replay without checkpoint content identity cannot qualify");

    bool invalid_policy = false;
    try {
        auto broken = policy;
        broken.policy_identity = 0;
        static_cast<void>(
            strix::assess_serial_decode_replay(
                broken,
                observation));
    } catch (const std::invalid_argument&) {
        invalid_policy = true;
    }
    expect(invalid_policy,
           "serial replay rejects uninitialized policy");

    if (failures == 0) {
        std::cout << "serial decode replay tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
