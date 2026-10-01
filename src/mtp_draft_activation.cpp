#include "strix/mtp_draft_activation.hpp"

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
    if (readiness.blockers.size() != 1) {
        return false;
    }
    return readiness.blockers.front() ==
        "complete MTP forward semantics gate is unresolved";
}

}  // namespace

ActivatedMtpDraftPlan
activate_mtp_draft_plan(
    const CompiledMtpDraftPlan& plan,
    const MtpForwardReplayPolicy& policy,
    const MtpForwardReplayObservation& observation) {
    if (plan.plan_identity == 0 ||
        plan.proposal_identity == 0 ||
        plan.contract_identity == 0) {
        throw std::invalid_argument(
            "MTP semantic activation requires initialized structural plan");
    }
    if (!only_forward_gate_remains(
            plan.readiness)) {
        throw std::logic_error(
            "MTP semantic activation requires all subcontracts except end-to-end replay to be resolved");
    }
    if (observation.plan_identity !=
            plan.plan_identity ||
        observation.proposal_identity !=
            plan.proposal_identity) {
        throw std::invalid_argument(
            "MTP replay evidence does not match structural plan identity");
    }

    const auto assessment =
        assess_mtp_forward_replay(
            policy,
            observation);
    if (!assessment.passes_declared_policy) {
        throw std::logic_error(
            "MTP replay evidence does not satisfy declared semantic policy");
    }

    ActivatedMtpDraftPlan activated;
    activated.structural_plan_identity =
        plan.plan_identity;
    activated.proposal_identity =
        plan.proposal_identity;
    activated.replay_policy_identity =
        policy.policy_identity;
    activated.independent_reference_identity =
        observation.independent_reference_identity;

    std::uint64_t hash = fnv_offset;
    hash_u64(
        hash,
        activated.structural_plan_identity);
    hash_u64(
        hash,
        activated.proposal_identity);
    hash_u64(
        hash,
        activated.replay_policy_identity);
    hash_u64(
        hash,
        activated.independent_reference_identity);
    activated.activation_identity = hash;
    if (activated.activation_identity == 0) {
        throw std::logic_error(
            "MTP activation identity must be non-zero");
    }

    activated.semantics_qualified = true;
    activated.hardware_qualified = false;
    return activated;
}

}  // namespace strix
