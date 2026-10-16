#include "strix/mtp_forward_replay.hpp"

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

}  // namespace

int main() {
    strix::MtpForwardReplayPolicy policy;
    policy.policy_identity = 0x1111ULL;
    policy.max_hidden_abs_error = 1e-4;
    policy.max_logit_abs_error = 1e-4;
    policy.minimum_steps = 3;

    strix::MtpForwardReplayObservation observation;
    observation.proposal_identity = 0x2222ULL;
    observation.plan_identity = 0x3333ULL;
    observation.independent_reference_identity = 0x4444ULL;
    observation.reference_tokens = {10, 20, 30};
    observation.engine_tokens = {10, 20, 30};
    observation.hidden_values_compared = 30'720;
    observation.maximum_hidden_abs_error = 5e-5;
    observation.logit_values_compared = 744'960;
    observation.maximum_logit_abs_error = 7e-5;

    const auto pass =
        strix::assess_mtp_forward_replay(
            policy,
            observation);
    expect(pass.identity_valid &&
               pass.multi_step &&
               pass.tokens_match &&
               pass.hidden_checked &&
               pass.logits_checked &&
               pass.passes_declared_policy,
           "MTP replay assessment accepts complete evidence within declared policy");

    expect(!observation.native_arithmetic_profile_identity && !pass.native_arithmetic_profile_identity,
        "portable replay assessment retains the legacy zero profile by default");
    auto native_a=observation;native_a.native_arithmetic_profile_identity=0x4d5450410001ULL;
    auto native_b=native_a;native_b.native_arithmetic_profile_identity=0x4d5450410002ULL;
    const auto pa=strix::assess_mtp_forward_replay(policy,native_a);
    const auto pb=strix::assess_mtp_forward_replay(policy,native_b);
    expect(pa.passes_declared_policy && pb.passes_declared_policy &&
        pa.native_arithmetic_profile_identity==native_a.native_arithmetic_profile_identity &&
        pb.native_arithmetic_profile_identity==native_b.native_arithmetic_profile_identity &&
        pa.native_arithmetic_profile_identity!=pb.native_arithmetic_profile_identity,
        "assessment preserves distinct native profiles without changing replay tolerances or token policy");
    auto failed_native=native_a;failed_native.maximum_logit_abs_error=policy.max_logit_abs_error*2;
    const auto pn=strix::assess_mtp_forward_replay(policy,failed_native);
    expect(!pn.passes_declared_policy && !pn.logits_within_policy &&
        pn.native_arithmetic_profile_identity==native_a.native_arithmetic_profile_identity,
        "failed native observations retain their profile for diagnostics rather than becoming portable or passing");
    auto token_mismatch = observation;
    token_mismatch.engine_tokens[1] = 99;
    const auto token_fail =
        strix::assess_mtp_forward_replay(
            policy,
            token_mismatch);
    expect(!token_fail.tokens_match &&
               !token_fail.passes_declared_policy,
           "exact-token replay mismatch blocks semantic evidence");

    auto hidden_mismatch = observation;
    hidden_mismatch.maximum_hidden_abs_error =
        2e-4;
    const auto hidden_fail =
        strix::assess_mtp_forward_replay(
            policy,
            hidden_mismatch);
    expect(!hidden_fail.hidden_within_policy &&
               !hidden_fail.passes_declared_policy,
           "hidden-state drift above predeclared policy blocks replay evidence");

    auto missing_reference = observation;
    missing_reference.independent_reference_identity = 0;
    const auto identity_fail =
        strix::assess_mtp_forward_replay(
            policy,
            missing_reference);
    expect(!identity_fail.identity_valid &&
               !identity_fail.passes_declared_policy,
           "replay evidence requires independently identified reference");

    bool invalid_policy = false;
    try {
        auto bad = policy;
        bad.max_logit_abs_error =
            std::numeric_limits<double>::infinity();
        static_cast<void>(
            strix::assess_mtp_forward_replay(
                bad,
                observation));
    } catch (const std::invalid_argument&) {
        invalid_policy = true;
    }
    expect(invalid_policy,
           "non-finite replay tolerance is rejected");

    if (failures == 0) {
        std::cout << "MTP forward replay tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
