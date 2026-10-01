#pragma once

#include "strix/compiled_mtp_draft_plan.hpp"
#include "strix/mtp_forward_replay.hpp"

#include <cstdint>
#include <vector>

namespace strix {

struct QualifiedMtpDraftPlan {
    CompiledMtpDraftPlan base;
    MtpForwardReplayPolicy replay_policy;
    MtpForwardReplayAssessment replay_assessment;

    std::uint64_t independent_reference_identity = 0;
    std::uint64_t qualification_identity = 0;
    bool executable = false;
};

[[nodiscard]] QualifiedMtpDraftPlan
qualify_mtp_draft_plan(
    const CompiledMtpDraftPlan& plan,
    const MtpForwardReplayPolicy& policy,
    const MtpForwardReplayObservation& observation);

void require_qualified_mtp_draft_plan(
    const QualifiedMtpDraftPlan& plan);

}  // namespace strix
