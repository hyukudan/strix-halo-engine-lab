#include "strix/mtp_draft_cycle_input.hpp"

#include <limits>
#include <stdexcept>

namespace strix {

void validate_mtp_draft_cycle_input(
    const ModelContract& model,
    const ActivatedMtpDraftPlan& activated,
    const MtpDraftStateContract& state_contract,
    const SerialSessionContext& target_session,
    const MtpDraftCycleInput& input) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (!activated.semantics_qualified ||
        activated.activation_identity == 0 ||
        activated.proposal_identity == 0) {
        throw std::logic_error(
            "MTP cycle input requires semantically activated draft plan");
    }
    if (activated.hardware_qualified) {
        // Hardware qualification is not needed for semantic input validation
        // and must not alter this contract.
    }

    if (input.proposal_identity !=
            activated.proposal_identity ||
        input.activation_identity !=
            activated.activation_identity) {
        throw std::invalid_argument(
            "MTP cycle input does not match activated proposal identity");
    }
    if (target_session.health !=
            SerialSessionHealth::ready ||
        target_session.runtime_compatibility_identity == 0 ||
        input.target_runtime_compatibility_identity !=
            target_session.runtime_compatibility_identity) {
        throw std::invalid_argument(
            "MTP cycle input target session binding is invalid");
    }
    if (static_cast<std::uint64_t>(
            input.seed_token_id) >=
        model.vocab_size) {
        throw std::invalid_argument(
            "MTP seed token is outside model vocabulary");
    }
    if (input.seed_position >=
            model.max_position_embeddings ||
        input.seed_position <
            target_session.next_position) {
        throw std::out_of_range(
            "MTP seed position cannot precede target base position or exceed model range");
    }
    if (input.hidden_source_position >
            input.seed_position) {
        throw std::invalid_argument(
            "MTP hidden source position cannot be after seed position");
    }

    validate_mtp_draft_state_contract(
        model,
        state_contract);
    if (input.hidden_elements !=
        state_contract.state_width) {
        throw std::invalid_argument(
            "MTP hidden binding width does not match wide draft-state contract");
    }
    if (input.hidden_address == 0 ||
        input.hidden_element_bytes == 0 ||
        input.hidden_allocation_generation == 0) {
        throw std::invalid_argument(
            "MTP hidden binding requires initialized physical storage");
    }

    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (input.hidden_elements >
        std::numeric_limits<std::uint64_t>::max() /
            input.hidden_element_bytes) {
        throw std::overflow_error(
            "MTP hidden byte range overflows uint64");
    }
    const auto bytes =
        input.hidden_elements *
        input.hidden_element_bytes;
    if (bytes >
        static_cast<std::uint64_t>(
            maximum - input.hidden_address)) {
        throw std::overflow_error(
            "MTP hidden address range overflows uintptr_t");
    }
}

}  // namespace strix
