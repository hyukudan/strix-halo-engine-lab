#include "strix/mtp_draft_cycle_input.hpp"

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

strix::ActivatedMtpDraftPlan activated() {
    strix::ActivatedMtpDraftPlan result;
    result.proposal_identity = 0x1111ULL;
    result.activation_identity = 0x2222ULL;
    result.semantics_qualified = true;
    result.hardware_qualified = false;
    return result;
}

strix::SerialSessionContext target() {
    strix::SerialSessionContext session;
    session.execution_identity = 0x10ULL;
    session.state_layout_identity = 0x20ULL;
    session.runtime_compatibility_identity =
        0x30ULL;
    session.next_position = 128;
    session.committed_tokens = 128;
    session.health =
        strix::SerialSessionHealth::ready;
    return session;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto state =
        strix::build_mtp_draft_state_contract(
            model);
    const auto activation =
        activated();
    const auto session =
        target();

    strix::MtpDraftCycleInput input;
    input.proposal_identity =
        activation.proposal_identity;
    input.activation_identity =
        activation.activation_identity;
    input.target_runtime_compatibility_identity =
        session.runtime_compatibility_identity;
    input.seed_token_id = 42;
    input.seed_position = 128;
    input.hidden_source_position = 127;
    input.hidden_address =
        static_cast<std::uintptr_t>(
            0x100000000ULL);
    input.hidden_elements =
        state.state_width;
    input.hidden_element_bytes = 2;
    input.hidden_allocation_generation = 7;

    strix::validate_mtp_draft_cycle_input(
        model,
        activation,
        state,
        session,
        input);

    auto collapsed = input;
    collapsed.hidden_elements =
        model.hidden_size;
    bool collapsed_rejected = false;
    try {
        strix::validate_mtp_draft_cycle_input(
            model,
            activation,
            state,
            session,
            collapsed);
    } catch (const std::invalid_argument&) {
        collapsed_rejected = true;
    }
    expect(collapsed_rejected,
           "MTP cycle rejects collapsed target hidden instead of wide HC state");

    auto foreign = input;
    ++foreign.activation_identity;
    bool foreign_rejected = false;
    try {
        strix::validate_mtp_draft_cycle_input(
            model,
            activation,
            state,
            session,
            foreign);
    } catch (const std::invalid_argument&) {
        foreign_rejected = true;
    }
    expect(foreign_rejected,
           "MTP cycle input cannot cross semantic activation identity");

    auto later_draft = input;
    later_draft.seed_position =
        session.next_position + 3;
    later_draft.hidden_source_position =
        later_draft.seed_position - 1;
    strix::validate_mtp_draft_cycle_input(
        model,
        activation,
        state,
        session,
        later_draft);
    expect(
        later_draft.seed_position >
            session.next_position,
        "MTP cycle allows later speculative positions without advancing target session");

    auto past = input;
    past.seed_position =
        session.next_position - 1;
    past.hidden_source_position =
        past.seed_position;
    bool past_rejected = false;
    try {
        strix::validate_mtp_draft_cycle_input(
            model,
            activation,
            state,
            session,
            past);
    } catch (const std::out_of_range&) {
        past_rejected = true;
    }
    expect(
        past_rejected,
        "MTP cycle rejects seed positions before target-session base");

    auto future_hidden = input;
    future_hidden.hidden_source_position =
        input.seed_position + 1;
    bool future_rejected = false;
    try {
        strix::validate_mtp_draft_cycle_input(
            model,
            activation,
            state,
            session,
            future_hidden);
    } catch (const std::invalid_argument&) {
        future_rejected = true;
    }
    expect(future_rejected,
           "MTP hidden source cannot come from a future position");

    if (failures == 0) {
        std::cout << "MTP draft cycle input tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
