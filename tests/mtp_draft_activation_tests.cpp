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
