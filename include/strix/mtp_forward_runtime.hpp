#pragma once

#include "strix/draft_proposal_state.hpp"
#include "strix/mtp_draft_binding.hpp"
#include "strix/mtp_draft_cycle_input.hpp"
#include "strix/mtp_draft_qualification.hpp"
#include "strix/mtp_draft_state_memory.hpp"
#include "strix/serial_session.hpp"

#include <cstdint>

namespace strix {

struct MtpForwardPhysicalResources {
    const ModelManifest* manifest = nullptr;
    const ModelRuntimeBinding* runtime = nullptr;
};

struct MtpForwardStepContext {
    MtpDraftPlanStep step =
        MtpDraftPlanStep::input_fusion;
    std::uint32_t step_index = 0;
    std::uint32_t input_token = 0;
    std::uint64_t target_position = 0;
    // Attention/cache position belongs to hidden[i], paired with token[i+1].
    std::uint64_t predictor_position = 0;
    std::uint64_t proposal_identity = 0;
    std::uint64_t plan_identity = 0;
    const MtpDraftRuntimeBinding* binding = nullptr;
    const MtpDraftStateMemoryBinding* state_memory = nullptr;
    const ModelManifest* manifest = nullptr;
    const ModelRuntimeBinding* runtime = nullptr;
    const MtpDraftCycleInput* cycle_input = nullptr;
};

class MtpForwardExecutionBackend {
public:
    virtual ~MtpForwardExecutionBackend() = default;

    virtual void dispatch_mtp_step(
        const MtpForwardStepContext& context) = 0;
};

struct MtpForwardExecutionResult {
    std::uint32_t completed_steps = 0;
    bool next_state_published = false;
    bool logits_published = false;
};

[[nodiscard]] MtpForwardExecutionResult
execute_qualified_mtp_forward(
    const QualifiedMtpDraftPlan& plan,
    const MtpDraftRuntimeBinding& binding,
    const MtpDraftStateMemoryBinding& state_memory,
    const MtpForwardPhysicalResources& resources,
    const SerialSessionContext& target_session,
    const DraftProposalStateContext& proposal_state,
    const MtpDraftCycleInput& cycle_input,
    MtpForwardExecutionBackend& backend);

}  // namespace strix
