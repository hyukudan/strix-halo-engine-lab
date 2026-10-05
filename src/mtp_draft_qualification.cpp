#include "strix/mtp_draft_qualification.hpp"

#include <stdexcept>
#include <string_view>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

bool only_forward_gate_remains(
    const MtpDraftReadiness& readiness) {
    if (readiness.executable) {
        return true;
    }
    if (readiness.blockers.size() != 1) {
        return false;
    }
    return readiness.blockers.front() ==
        "complete MTP forward semantics gate is unresolved";
}

std::uint64_t build_qualification_identity(
    const QualifiedMtpDraftPlan& qualified,
    const MtpForwardReplayObservation& observation) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, qualified.base.plan_identity);
    hash_u64(hash, qualified.base.proposal_identity);
    hash_u64(hash, qualified.replay_policy.policy_identity);
    hash_u64(
        hash,
        observation.independent_reference_identity);
    hash_u64(hash, observation.plan_identity);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            observation.reference_tokens.size()));
    for (const auto token : observation.reference_tokens) {
        hash_u64(hash, token);
    }
    for (const auto token : observation.engine_tokens) {
        hash_u64(hash, token);
    }
    hash_u64(
        hash,
        observation.hidden_values_compared);
    hash_u64(
        hash,
        observation.logit_values_compared);
    return hash;
}

}  // namespace

QualifiedMtpDraftPlan
qualify_mtp_draft_plan(
    const CompiledMtpDraftPlan& plan,
    const MtpForwardReplayPolicy& policy,
    const MtpForwardReplayObservation& observation) {
    if (plan.plan_identity == 0 ||
        plan.proposal_identity == 0 ||
        plan.contract_identity == 0 ||
        plan.plan_identity !=
            recompute_compiled_mtp_draft_plan_identity(
                plan)) {
        throw std::invalid_argument(
            "MTP qualification requires an initialized compiled plan");
    }
    if (observation.proposal_identity !=
            plan.proposal_identity ||
        observation.plan_identity !=
            plan.plan_identity) {
        throw std::invalid_argument(
            "MTP replay observation does not belong to compiled plan");
    }
    if (!only_forward_gate_remains(plan.readiness)) {
        throw std::logic_error(
            "MTP qualification cannot bypass unresolved structural blockers");
    }

    QualifiedMtpDraftPlan qualified;
    qualified.base = plan;
    qualified.replay_policy = policy;
    qualified.replay_assessment =
        assess_mtp_forward_replay(
            policy,
            observation);
    qualified.independent_reference_identity =
        observation.independent_reference_identity;
    qualified.qualified_plan_identity =
        plan.plan_identity;

    qualified.executable =
        qualified.replay_assessment.passes_declared_policy;

    qualified.qualification_identity =
        build_qualification_identity(
            qualified,
            observation);

    if (qualified.qualification_identity == 0) {
        throw std::logic_error(
            "MTP qualification identity must be non-zero");
    }

    return qualified;
}

void require_qualified_mtp_draft_plan(
    const QualifiedMtpDraftPlan& plan) {
    if (plan.qualification_identity == 0 ||
        plan.independent_reference_identity == 0 ||
        plan.qualified_plan_identity == 0) {
        throw std::invalid_argument(
            "MTP qualified plan is not initialized");
    }
    if (plan.base.plan_identity !=
            plan.qualified_plan_identity ||
        plan.base.plan_identity !=
            recompute_compiled_mtp_draft_plan_identity(
                plan.base)) {
        throw std::logic_error(
            "MTP qualified plan changed after replay qualification");
    }
    if (!plan.executable ||
        !plan.replay_assessment.passes_declared_policy) {
        throw std::logic_error(
            "MTP plan has not passed independent forward replay qualification");
    }
}

}  // namespace strix
