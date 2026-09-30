#include "strix/speculation_cycle.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    const char* message) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(message);
    }
    return left + right;
}

}  // namespace

GreedySpeculationCyclePlan
build_greedy_speculation_cycle_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const GreedySpeculationCycleRequest& request) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.compiled_execution_identity == 0 ||
        request.runtime_compatibility_identity == 0) {
        throw std::invalid_argument(
            "speculation cycle requires compiled and runtime identities");
    }
    if (request.base_position >
        state_layout.context_tokens) {
        throw std::out_of_range(
            "speculation base position exceeds allocated state capacity");
    }
    if (request.base_position >
        model.max_position_embeddings) {
        throw std::out_of_range(
            "speculation base position exceeds model context limit");
    }

    GreedyVerificationRequest verification_request;
    verification_request.candidate_tokens =
        request.candidate_tokens;
    verification_request.target_tokens =
        request.target_tokens;
    verification_request.token_budget =
        request.token_budget;
    verification_request.eos_tokens =
        request.eos_tokens;

    GreedySpeculationCyclePlan plan;
    plan.verification =
        verify_greedy_candidates(
            verification_request);
    plan.base_position =
        request.base_position;
    plan.base_committed_tokens =
        request.committed_tokens;
    plan.emitted_tokens =
        plan.verification.valid_tokens;
    plan.speculation_used =
        plan.verification.candidate_length != 0;
    plan.verify_state_advance =
        plan.verification.candidate_length;
    plan.committed_state_advance =
        plan.verification.accepted_matches;

    const auto verify_end =
        checked_add(
            request.base_position,
            plan.verify_state_advance,
            "speculative verify position overflow");
    if (verify_end >
            state_layout.context_tokens ||
        verify_end >
            model.max_position_embeddings) {
        throw std::out_of_range(
            "speculative verify window exceeds allocated/model context");
    }

    plan.post_cycle_state_position =
        checked_add(
            request.base_position,
            plan.committed_state_advance,
            "speculative committed-state position overflow");

    // Output tokens include the target bonus token even though that token is
    // not yet processed into target state.
    plan.post_cycle_output_position =
        checked_add(
            request.base_position,
            static_cast<std::uint64_t>(
                plan.emitted_tokens.size()),
            "speculative output position overflow");

    if (plan.post_cycle_output_position >
        model.max_position_embeddings) {
        throw std::out_of_range(
            "speculative output exceeds model context");
    }

    plan.post_cycle_committed_tokens =
        checked_add(
            request.committed_tokens,
            static_cast<std::uint64_t>(
                plan.emitted_tokens.size()),
            "speculative committed-token counter overflow");

    if (!plan.speculation_used) {
        // Depth zero is a serial semantic baseline. No target state has been
        // advanced by candidates, so no snapshot/restore/replay is required.
        return plan;
    }

    plan.snapshot_required = true;
    plan.restore_required =
        plan.verification.accepted_matches !=
        plan.verification.candidate_length;

    if (plan.restore_required) {
        plan.replay_tokens.assign(
            request.candidate_tokens.begin(),
            request.candidate_tokens.begin() +
                static_cast<std::ptrdiff_t>(
                    plan.verification.accepted_matches));
    }

    SerialSessionSnapshotRequest snapshot_request;
    snapshot_request.compiled_execution_identity =
        request.compiled_execution_identity;
    snapshot_request.runtime_compatibility_identity =
        request.runtime_compatibility_identity;
    snapshot_request.next_position =
        request.base_position;
    snapshot_request.committed_tokens =
        request.committed_tokens;
    snapshot_request.state =
        request.state;
    snapshot_request.alignment =
        request.snapshot_alignment;

    plan.pre_window_snapshot =
        build_serial_session_snapshot_plan(
            model,
            state_layout,
            snapshot_request);

    return plan;
}

}  // namespace strix
