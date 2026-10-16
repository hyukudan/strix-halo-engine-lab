#include "strix/mtp_draft_qualification.hpp"

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

strix::CompiledMtpDraftPlan fixture_plan() {
    strix::CompiledMtpDraftPlan plan;
    plan.proposal_identity = 0x1111ULL;
    plan.contract_identity = 0x2222ULL;
    plan.readiness.tensor_families_complete = true;
    plan.readiness.input_fusion_semantics_resolved = true;
    plan.readiness.attention_semantics_resolved = true;
    plan.readiness.draft_state_semantics_resolved = true;
    plan.readiness.output_mixer_semantics_resolved = true;
    plan.readiness.output_head_binding_resolved = true;
    plan.readiness.output_semantics_resolved = true;
    plan.readiness.executable = false;
    plan.readiness.blockers = {
        "complete MTP forward semantics gate is unresolved",
    };
    plan.executable = false;
    plan.plan_identity =
        strix::recompute_compiled_mtp_draft_plan_identity(
            plan);
    return plan;
}

strix::MtpForwardReplayPolicy policy() {
    strix::MtpForwardReplayPolicy value;
    value.policy_identity = 0x4444ULL;
    value.max_hidden_abs_error = 1e-3;
    value.max_logit_abs_error = 1e-3;
    value.minimum_steps = 2;
    value.require_exact_tokens = true;
    return value;
}

strix::MtpForwardReplayObservation passing_observation(
    const strix::CompiledMtpDraftPlan& plan) {
    strix::MtpForwardReplayObservation value;
    value.proposal_identity =
        plan.proposal_identity;
    value.plan_identity =
        plan.plan_identity;
    value.independent_reference_identity = 0x5555ULL;
    value.reference_tokens = {10, 20, 30};
    value.engine_tokens = value.reference_tokens;
    value.hidden_values_compared = 128;
    value.maximum_hidden_abs_error = 5e-4;
    value.logit_values_compared = 256;
    value.maximum_logit_abs_error = 5e-4;
    return value;
}

// Historical portable qualification framing, deliberately without a native
// profile word. Zero must preserve this identity rather than append a zero.
std::uint64_t legacy_qualification_identity(const strix::CompiledMtpDraftPlan& plan,
    const strix::MtpForwardReplayPolicy& policy,const strix::MtpForwardReplayObservation& observed) {
    std::uint64_t hash=14695981039346656037ULL;
    const auto word=[&](std::uint64_t value){for(unsigned shift=0;shift<64;shift+=8)
        hash=(hash^((value>>shift)&255ULL))*1099511628211ULL;};
    word(plan.plan_identity);word(plan.proposal_identity);word(policy.policy_identity);
    word(observed.independent_reference_identity);word(observed.plan_identity);
    word(observed.reference_tokens.size());
    for(auto token:observed.reference_tokens)word(token);
    for(auto token:observed.engine_tokens)word(token);
    word(observed.hidden_values_compared);word(observed.logit_values_compared);
    return hash;
}
}  // namespace

int main() {
    const auto plan = fixture_plan();
    const auto replay_policy = policy();
    const auto observation =
        passing_observation(plan);

    const auto qualified =
        strix::qualify_mtp_draft_plan(
            plan,
            replay_policy,
            observation);

    expect(qualified.executable &&
               qualified.replay_assessment.passes_declared_policy &&
               qualified.qualification_identity != 0 &&
               qualified.qualified_plan_identity ==
                   plan.plan_identity &&
               qualified.independent_reference_identity ==
                   observation.independent_reference_identity,
           "independent passing replay qualifies otherwise-complete MTP plan");

    strix::require_qualified_mtp_draft_plan(
        qualified);

    expect(!qualified.native_arithmetic_profile_identity && !qualified.replay_assessment.native_arithmetic_profile_identity &&
        qualified.qualification_identity==legacy_qualification_identity(plan,replay_policy,observation),
        "portable profile zero retains historical qualification identity framing");
    auto native_a=observation;native_a.native_arithmetic_profile_identity=0x4d5450410001ULL;
    auto native_b=native_a;native_b.native_arithmetic_profile_identity=0x4d5450410002ULL;
    const auto qa=strix::qualify_mtp_draft_plan(plan,replay_policy,native_a);
    const auto qb=strix::qualify_mtp_draft_plan(plan,replay_policy,native_b);
    expect(qa.executable && qb.executable &&
        qa.native_arithmetic_profile_identity==native_a.native_arithmetic_profile_identity &&
        qa.replay_assessment.native_arithmetic_profile_identity==native_a.native_arithmetic_profile_identity &&
        qb.native_arithmetic_profile_identity==native_b.native_arithmetic_profile_identity &&
        qb.replay_assessment.native_arithmetic_profile_identity==native_b.native_arithmetic_profile_identity &&
        qa.qualification_identity!=qb.qualification_identity && qa.qualification_identity!=qualified.qualification_identity,
        "otherwise identical passing replay qualifications remain bound to distinct native arithmetic profiles");
    strix::require_qualified_mtp_draft_plan(qa);strix::require_qualified_mtp_draft_plan(qb);
    auto changed_profile=qa;changed_profile.native_arithmetic_profile_identity=native_b.native_arithmetic_profile_identity;
    bool changed_profile_rejected=false;
    try{strix::require_qualified_mtp_draft_plan(changed_profile);}catch(const std::logic_error&){changed_profile_rejected=true;}
    expect(changed_profile_rejected,"qualified native profile cannot diverge from its replay assessment");
    auto changed_assessment=qa;changed_assessment.replay_assessment.native_arithmetic_profile_identity=0;
    bool changed_assessment_rejected=false;
    try{strix::require_qualified_mtp_draft_plan(changed_assessment);}catch(const std::logic_error&){changed_assessment_rejected=true;}
    expect(changed_assessment_rejected,"native qualification cannot be relabelled as portable replay by mutating its assessment");
    auto native_failed=native_a;native_failed.engine_tokens[1]=999;
    const auto native_rejected=strix::qualify_mtp_draft_plan(plan,replay_policy,native_failed);
    expect(!native_rejected.executable && native_rejected.native_arithmetic_profile_identity==native_a.native_arithmetic_profile_identity,
        "a native profile identifies failed evidence without promoting it to executable qualification");
    auto bad_tokens = observation;
    bad_tokens.engine_tokens[1] = 21;
    const auto rejected =
        strix::qualify_mtp_draft_plan(
            plan,
            replay_policy,
            bad_tokens);
    expect(!rejected.executable &&
               !rejected.replay_assessment.passes_declared_policy,
           "token mismatch keeps MTP plan non-executable");

    bool require_rejected = false;
    try {
        strix::require_qualified_mtp_draft_plan(
            rejected);
    } catch (const std::logic_error&) {
        require_rejected = true;
    }
    expect(require_rejected,
           "runtime gate rejects failed MTP replay qualification");

    auto wrong_plan_observation = observation;
    ++wrong_plan_observation.plan_identity;
    bool identity_rejected = false;
    try {
        static_cast<void>(
            strix::qualify_mtp_draft_plan(
                plan,
                replay_policy,
                wrong_plan_observation));
    } catch (const std::invalid_argument&) {
        identity_rejected = true;
    }
    expect(identity_rejected,
           "replay evidence cannot qualify a different compiled plan");

    auto mutated_after_qualification =
        qualified;
    ++mutated_after_qualification.
        base.forward.contract_identity;
    bool mutated_rejected = false;
    try {
        strix::require_qualified_mtp_draft_plan(
            mutated_after_qualification);
    } catch (const std::logic_error&) {
        mutated_rejected = true;
    }
    expect(
        mutated_rejected,
        "qualified MTP evidence cannot survive forward-contract mutation");

    auto reidentified_after_qualification =
        qualified;
    ++reidentified_after_qualification.
        base.output.vocab_size;
    reidentified_after_qualification.
        base.plan_identity =
        strix::recompute_compiled_mtp_draft_plan_identity(
            reidentified_after_qualification.base);
    bool reidentified_rejected = false;
    try {
        strix::require_qualified_mtp_draft_plan(
            reidentified_after_qualification);
    } catch (const std::logic_error&) {
        reidentified_rejected = true;
    }
    expect(
        reidentified_rejected,
        "recomputing a mutated base plan cannot reuse old replay qualification");

    auto structurally_blocked = plan;
    structurally_blocked.readiness.blockers.insert(
        structurally_blocked.readiness.blockers.begin(),
        "required MTP tensor families are missing");
    bool blocker_rejected = false;
    try {
        static_cast<void>(
            strix::qualify_mtp_draft_plan(
                structurally_blocked,
                replay_policy,
                observation));
    } catch (const std::logic_error&) {
        blocker_rejected = true;
    }
    expect(blocker_rejected,
           "forward replay cannot bypass independent structural blockers");

    if (failures == 0) {
        std::cout << "MTP draft qualification tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
