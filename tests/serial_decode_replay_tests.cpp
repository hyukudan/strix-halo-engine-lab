#include "strix/serial_decode_replay.hpp"

#include <array>
#include <iostream>
#include <limits>
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
    policy.policy_identity = 0x1003ULL;
    policy.max_logit_abs_error = 1e-3;
    policy.minimum_steps = 2;
    policy.minimum_logit_values_compared = 3;
    policy.require_exact_argmax = true;

    const std::array<float, 3> finite_logits{
        1.0F,
        -2.0F,
        3.5F,
    };
    bool finite_logits_valid = true;
    try {
        strix::validate_serial_decode_engine_logits(
            finite_logits,
            finite_logits.size());
    } catch (const std::invalid_argument&) {
        finite_logits_valid = false;
    }
    expect(finite_logits_valid,
           "finite engine logits with exact vocabulary size pass preflight");

    auto non_finite_logits = finite_logits;
    non_finite_logits[1] =
        std::numeric_limits<float>::quiet_NaN();
    bool non_finite_rejected = false;
    try {
        strix::validate_serial_decode_engine_logits(
            non_finite_logits,
            non_finite_logits.size());
    } catch (const std::invalid_argument&) {
        non_finite_rejected = true;
    }
    expect(non_finite_rejected,
           "non-finite engine logits fail replay preflight");

    bool wrong_vocab_rejected = false;
    try {
        strix::validate_serial_decode_engine_logits(
            finite_logits,
            finite_logits.size() + 1U);
    } catch (const std::invalid_argument&) {
        wrong_vocab_rejected = true;
    }
    expect(wrong_vocab_rejected,
           "engine logit count must match the target vocabulary exactly");

    bool zero_vocab_rejected = false;
    try {
        strix::validate_serial_decode_engine_logits(
            finite_logits,
            0);
    } catch (const std::invalid_argument&) {
        zero_vocab_rejected = true;
    }
    expect(zero_vocab_rejected,
           "zero expected vocabulary is rejected before replay assessment");

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
    no_logits.logit_values_compared = 2;
    no_logits.maximum_logit_abs_error = 0.0;
    const auto unchecked =
        strix::assess_serial_decode_replay(
            policy,
            no_logits);
    expect(!unchecked.logits_checked &&
               !unchecked.passes_declared_policy,
           "insufficient compared logits cannot qualify serial replay");

    auto wrong_identity = observation;
    wrong_identity.model_content_identity = {};
    const auto unidentified =
        strix::assess_serial_decode_replay(
            policy,
            wrong_identity);
    expect(!unidentified.identity_valid &&
               !unidentified.passes_declared_policy,
           "replay without checkpoint content identity cannot qualify");

    auto wrong_reference = observation;
    wrong_reference.independent_reference_identity =
        policy.policy_identity + 1ULL;
    const auto foreign_reference =
        strix::assess_serial_decode_replay(
            policy,
            wrong_reference);
    expect(!foreign_reference.identity_valid &&
               !foreign_reference.passes_declared_policy,
           "replay from a different independent reference cannot qualify");

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

    bool single_step_policy_rejected = false;
    try {
        auto broken = policy;
        broken.minimum_steps = 1;
        static_cast<void>(
            strix::assess_serial_decode_replay(
                broken,
                observation));
    } catch (const std::invalid_argument&) {
        single_step_policy_rejected = true;
    }
    expect(single_step_policy_rejected,
           "serial replay policy cannot downgrade qualification to one step");

    bool zero_probe_policy_rejected = false;
    try {
        auto broken = policy;
        broken.minimum_logit_values_compared = 0;
        static_cast<void>(
            strix::assess_serial_decode_replay(
                broken,
                observation));
    } catch (const std::invalid_argument&) {
        zero_probe_policy_rejected = true;
    }
    expect(zero_probe_policy_rejected,
           "serial replay policy requires at least one compared logit");

    if (failures == 0) {
        std::cout << "serial decode replay tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
