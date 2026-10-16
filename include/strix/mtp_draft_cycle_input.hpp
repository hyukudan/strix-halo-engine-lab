#pragma once

#include "strix/model_contract.hpp"
#include "strix/mtp_draft_activation.hpp"
#include "strix/mtp_draft_state_contract.hpp"
#include "strix/serial_session.hpp"

#include <cstdint>

namespace strix {

struct MtpDraftCycleInput {
    std::uint64_t proposal_identity = 0;
    std::uint64_t activation_identity = 0;
    std::uint64_t target_runtime_compatibility_identity = 0;

    std::uint32_t seed_token_id = 0;
    std::uint64_t seed_position = 0;

    // The source hidden is the target-produced wide HC state supplied to the
    // first MTP step. Its exact lifecycle is explicit rather than inferred from
    // a pointer or from target session counters.
    // Exact shifted pairing: seed_position == hidden_source_position + 1.
    // This source position is also the MTP rotary/cache position.
    std::uint64_t hidden_source_position = 0;
    std::uintptr_t hidden_address = 0;
    std::uint64_t hidden_elements = 0;
    std::uint64_t hidden_element_bytes = 0;
    std::uint64_t hidden_allocation_generation = 0;
};

void validate_mtp_draft_cycle_input(
    const ModelContract& model,
    const ActivatedMtpDraftPlan& activated,
    const MtpDraftStateContract& state_contract,
    const SerialSessionContext& target_session,
    const MtpDraftCycleInput& input);

}  // namespace strix
