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

GreedySpeculationPreparation
prepare_greedy_speculation_cycle(
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
        state_layout.context_tokens ||
        request.base_position >
            model.max_position_embeddings) {
        throw std::out_of_range(
            "speculation base position exceeds context capacity");
    }

    if (request.candidate_tokens.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
            "speculation candidate depth exceeds uint32");
    }
    for (const auto token : request.candidate_tokens) {
        if (static_cast<std::uint64_t>(token) >=
            model.vocab_size) {
            throw std::invalid_argument(
                "speculation candidate token is outside model vocabulary");
        }
    }

    const auto verify_end =
        checked_add(
            request.base_position,
            static_cast<std::uint64_t>(
                request.candidate_tokens.size()),
            "speculative verify position overflow");
    if (verify_end >
            state_layout.context_tokens ||
        verify_end >
            model.max_position_embeddings) {
        throw std::out_of_range(
            "speculative verify window exceeds allocated/model context");
    }

    GreedySpeculationPreparation preparation;
    preparation.compiled_execution_identity =
        request.compiled_execution_identity;
    preparation.runtime_compatibility_identity =
        request.runtime_compatibility_identity;
    preparation.base_position =
        request.base_position;
    preparation.committed_tokens =
        request.committed_tokens;
    preparation.state =
        request.state;
    preparation.candidate_tokens.assign(
        request.candidate_tokens.begin(),
        request.candidate_tokens.end());

    if (!preparation.candidate_tokens.empty()) {
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

        preparation.pre_window_snapshot =
            build_serial_session_snapshot_plan(
                model,
                state_layout,
                snapshot_request);
    }

    return preparation;
}

GreedySpeculationCyclePlan
finalize_greedy_speculation_cycle(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const GreedySpeculationPreparation& preparation,
    const GreedySpeculationFinalizeRequest& request) {
    if (preparation.compiled_execution_identity == 0 ||
        preparation.runtime_compatibility_identity == 0) {
        throw std::invalid_argument(
            "speculation preparation is not initialized");
    }
    if (preparation.base_position >
            state_layout.context_tokens ||
        preparation.base_position >
            model.max_position_embeddings) {
        throw std::out_of_range(
            "speculation preparation position exceeds context capacity");
    }

    GreedyVerificationRequest verification_request;
    verification_request.candidate_tokens =
        preparation.candidate_tokens;
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
        preparation.base_position;
    plan.base_committed_tokens =
        preparation.committed_tokens;
    plan.emitted_tokens =
        plan.verification.valid_tokens;
    plan.speculation_used =
        plan.verification.candidate_length != 0;
    plan.verify_state_advance =
        plan.verification.candidate_length;
    plan.committed_state_advance =
        plan.verification.accepted_matches;

    plan.post_cycle_state_position =
        checked_add(
            preparation.base_position,
            plan.committed_state_advance,
            "speculative committed-state position overflow");
    plan.post_cycle_output_position =
        checked_add(
            preparation.base_position,
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
            preparation.committed_tokens,
            static_cast<std::uint64_t>(
                plan.committed_state_advance),
            "speculative committed-token counter overflow");

    if (!plan.speculation_used) {
        return plan;
    }

    if (preparation.pre_window_snapshot.plan_identity == 0 ||
        preparation.pre_window_snapshot.next_position !=
            preparation.base_position ||
        preparation.pre_window_snapshot.compiled_execution_identity !=
            preparation.compiled_execution_identity ||
        preparation.pre_window_snapshot.runtime_compatibility_identity !=
            preparation.runtime_compatibility_identity) {
        throw std::invalid_argument(
            "positive-depth speculation requires matching pre-window snapshot plan");
    }

    plan.snapshot_required = true;
    plan.restore_required =
        plan.verification.accepted_matches !=
        plan.verification.candidate_length;

    if (plan.restore_required) {
        plan.replay_tokens.assign(
            preparation.candidate_tokens.begin(),
            preparation.candidate_tokens.begin() +
                static_cast<std::ptrdiff_t>(
                    plan.verification.accepted_matches));
    }

    plan.pre_window_snapshot =
        preparation.pre_window_snapshot;
    return plan;
}

GreedySpeculationCyclePlan
build_greedy_speculation_cycle_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const GreedySpeculationCycleRequest& request) {
    const auto preparation =
        prepare_greedy_speculation_cycle(
            model,
            state_layout,
            request);

    GreedySpeculationFinalizeRequest finalize;
    finalize.target_tokens =
        request.target_tokens;
    finalize.token_budget =
        request.token_budget;
    finalize.eos_tokens =
        request.eos_tokens;

    return finalize_greedy_speculation_cycle(
        model,
        state_layout,
        preparation,
        finalize);
}

}  // namespace strix
