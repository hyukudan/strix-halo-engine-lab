#include "strix/mtp_draft_activation.hpp"

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

strix::CompiledMtpDraftPlan ready_structural_plan() {
    strix::CompiledMtpDraftPlan plan;
    plan.plan_identity = 0x1111ULL;
    plan.proposal_identity = 0x2222ULL;
    plan.contract_identity = 0x3333ULL;
    plan.readiness.tensor_families_complete = true;
    plan.readiness.input_fusion_semantics_resolved = true;
    plan.readiness.attention_semantics_resolved = true;
    plan.readiness.draft_state_semantics_resolved = true;
    plan.readiness.output_mixer_semantics_resolved = true;
    plan.readiness.output_head_binding_resolved = true;
    plan.readiness.output_semantics_resolved = true;
    plan.readiness.blockers = {
        "complete MTP forward semantics gate is unresolved",
    };
    return plan;
}

strix::MtpForwardReplayPolicy policy() {
    strix::MtpForwardReplayPolicy result;
    result.policy_identity = 0x4444ULL;
    result.max_hidden_abs_error = 1e-4;
    result.max_logit_abs_error = 1e-4;
    result.minimum_steps = 3;
    return result;
}

strix::MtpForwardReplayObservation observation() {
    strix::MtpForwardReplayObservation result;
    result.proposal_identity = 0x2222ULL;
    result.plan_identity = 0x1111ULL;
    result.independent_reference_identity =
        0x5555ULL;
    result.reference_tokens = {10, 20, 30};
    result.engine_tokens = {10, 20, 30};
    result.hidden_values_compared = 30'720;
    result.maximum_hidden_abs_error = 5e-5;
    result.logit_values_compared = 744'960;
    result.maximum_logit_abs_error = 7e-5;
    return result;
}

// Compatibility oracle for the pre-profile portable activation wire identity.
std::uint64_t legacy_activation_identity(const strix::CompiledMtpDraftPlan& plan,
    const strix::MtpForwardReplayPolicy& policy,const strix::MtpForwardReplayObservation& replay) {
    std::uint64_t hash=14695981039346656037ULL;
    for(const auto value:{plan.plan_identity,plan.proposal_identity,policy.policy_identity,replay.independent_reference_identity})
        for(unsigned shift=0;shift<64;shift+=8)hash=(hash^((value>>shift)&255ULL))*1099511628211ULL;
    return hash;
}
}  // namespace

int main() {
    const auto plan =
        ready_structural_plan();
    const auto replay_policy =
        policy();
    const auto replay =
        observation();

    const auto activated =
        strix::activate_mtp_draft_plan(
            plan,
            replay_policy,
            replay);

    expect(activated.semantics_qualified &&
               !activated.hardware_qualified &&
               activated.activation_identity != 0,
           "replay evidence activates semantics without claiming hardware qualification");
    expect(activated.structural_plan_identity ==
               plan.plan_identity &&
               activated.replay_policy_identity ==
                   replay_policy.policy_identity &&
               activated.independent_reference_identity ==
                   replay.independent_reference_identity,
           "activation identity is anchored to plan, policy and independent reference");

    expect(!activated.native_arithmetic_profile_identity &&
        activated.activation_identity==legacy_activation_identity(plan,replay_policy,replay),
        "portable profile zero preserves the historical activation identity");
    auto native_a=replay;native_a.native_arithmetic_profile_identity=0x4d5450410001ULL;
    auto native_b=native_a;native_b.native_arithmetic_profile_identity=0x4d5450410002ULL;
    const auto aa=strix::activate_mtp_draft_plan(plan,replay_policy,native_a);
    const auto ab=strix::activate_mtp_draft_plan(plan,replay_policy,native_b);
    const auto assessed=strix::assess_mtp_forward_replay(replay_policy,native_a);
    expect(assessed.native_arithmetic_profile_identity==native_a.native_arithmetic_profile_identity &&
        aa.native_arithmetic_profile_identity==assessed.native_arithmetic_profile_identity &&
        ab.native_arithmetic_profile_identity==native_b.native_arithmetic_profile_identity &&
        aa.activation_identity!=ab.activation_identity && aa.activation_identity!=activated.activation_identity,
        "assessment-to-activation preserves the actual native arithmetic profile and separates otherwise identical evidence");
    expect(aa.semantics_qualified && ab.semantics_qualified && !aa.hardware_qualified && !ab.hardware_qualified,
        "profile-bound semantic activation still does not assert native hardware qualification");
    auto native_bad=native_a;native_bad.engine_tokens[0]=999;
    bool native_bad_rejected=false;
    try{(void)strix::activate_mtp_draft_plan(plan,replay_policy,native_bad);}catch(const std::logic_error&){native_bad_rejected=true;}
    expect(native_bad_rejected,"nonzero native arithmetic identity cannot bypass failing replay tokens");
    auto wrong_plan_evidence = replay;
    ++wrong_plan_evidence.plan_identity;
    bool wrong_identity_rejected = false;
    try {
        static_cast<void>(
            strix::activate_mtp_draft_plan(
                plan,
                replay_policy,
                wrong_plan_evidence));
    } catch (const std::invalid_argument&) {
        wrong_identity_rejected = true;
    }
    expect(wrong_identity_rejected,
           "semantic activation rejects replay evidence from another plan");

    auto unresolved = plan;
    unresolved.readiness.blockers.insert(
        unresolved.readiness.blockers.begin(),
        "MTP LM-head checkpoint binding is unresolved");
    bool unresolved_rejected = false;
    try {
        static_cast<void>(
            strix::activate_mtp_draft_plan(
                unresolved,
                replay_policy,
                replay));
    } catch (const std::logic_error&) {
        unresolved_rejected = true;
    }
    expect(unresolved_rejected,
           "semantic activation refuses plans with unresolved subcontracts");

    auto failing_replay = replay;
    failing_replay.engine_tokens[1] = 99;
    bool replay_rejected = false;
    try {
        static_cast<void>(
            strix::activate_mtp_draft_plan(
                plan,
                replay_policy,
                failing_replay));
    } catch (const std::logic_error&) {
        replay_rejected = true;
    }
    expect(replay_rejected,
           "semantic activation refuses replay that fails declared policy");

    if (failures == 0) {
        std::cout << "MTP draft activation tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
