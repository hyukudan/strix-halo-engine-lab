#pragma once

#include "strix/compiled_mtp_draft_plan.hpp"
#include "strix/mtp_forward_replay.hpp"

#include <cstdint>

namespace strix {

struct ActivatedMtpDraftPlan {
    std::uint64_t structural_plan_identity = 0;
    std::uint64_t proposal_identity = 0;
    std::uint64_t replay_policy_identity = 0;
    std::uint64_t independent_reference_identity = 0;
    std::uint64_t activation_identity = 0;

    // Semantic activation is deliberately distinct from native Strix Halo
    // performance/hardware qualification.
    bool semantics_qualified = false;
    bool hardware_qualified = false;
};

[[nodiscard]] ActivatedMtpDraftPlan
activate_mtp_draft_plan(
    const CompiledMtpDraftPlan& plan,
    const MtpForwardReplayPolicy& policy,
    const MtpForwardReplayObservation& observation);

}  // namespace strix
