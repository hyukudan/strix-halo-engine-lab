#include "strix/speculation_sampling_cycle.hpp"

#include "strix/speculation_sampling.hpp"

#include <algorithm>
#include <cstddef>
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

bool is_eos(
    std::uint32_t token,
    std::span<const std::uint32_t> eos_tokens) noexcept {
    return std::find(
               eos_tokens.begin(),
               eos_tokens.end(),
               token) !=
           eos_tokens.end();
}

}  // namespace

SampledSpeculationPreparation
prepare_sampled_speculation_cycle(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const SampledSpeculationCycleRequest& request) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.compiled_execution_identity == 0 ||
        request.runtime_compatibility_identity == 0 ||
        request.token_budget == 0) {
        throw std::invalid_argument(
            "sampled speculation requires identities and non-zero token budget");
    }
    if (request.base_position >
            state_layout.context_tokens ||
        request.base_position >
            model.max_position_embeddings) {
        throw std::out_of_range(
            "sampled speculation base position exceeds context capacity");
    }
    if (request.candidate_tokens.size() >
        static_cast<std::size_t>(
            request.token_budget - 1U)) {
        throw std::invalid_argument(
            "sampled speculation candidate depth must leave room for residual/bonus token");
    }
    for (const auto token :
         request.candidate_tokens) {
        if (static_cast<std::uint64_t>(token) >=
            model.vocab_size) {
            throw std::invalid_argument(
                "sampled speculation candidate token is outside model vocabulary");
        }
    }
    for (const auto token :
         request.eos_tokens) {
        if (static_cast<std::uint64_t>(token) >=
            model.vocab_size) {
            throw std::invalid_argument(
                "sampled speculation EOS token is outside model vocabulary");
        }
    }

    const auto verify_end =
        checked_add(
            request.base_position,
            static_cast<std::uint64_t>(
                request.candidate_tokens.size()),
            "sampled speculation verify position overflow");
    if (verify_end >
            state_layout.context_tokens ||
        verify_end >
            model.max_position_embeddings) {
        throw std::out_of_range(
            "sampled speculation verify window exceeds context capacity");
    }

    SampledSpeculationPreparation preparation;
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

SampledSpeculationCyclePlan
finalize_sampled_speculation_cycle(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const SampledSpeculationPreparation& preparation,
    const SampledSpeculationFinalizeRequest& request) {
    if (preparation.compiled_execution_identity == 0 ||
        preparation.runtime_compatibility_identity == 0 ||
        request.token_budget == 0) {
        throw std::invalid_argument(
            "sampled speculation finalization is not initialized");
    }
    if (preparation.base_position >
            state_layout.context_tokens ||
        preparation.base_position >
            model.max_position_embeddings) {
        throw std::out_of_range(
            "sampled speculation preparation exceeds context capacity");
    }
    const auto depth =
        preparation.candidate_tokens.size();
    if (request.draft_probabilities.size() !=
            depth ||
        request.target_probabilities.size() !=
            depth + 1U) {
        throw std::invalid_argument(
            "sampled speculation requires one draft distribution per candidate and one extra target distribution");
    }
    if (depth >
        static_cast<std::size_t>(
            request.token_budget - 1U)) {
        throw std::invalid_argument(
            "sampled speculation depth exceeds token budget");
    }
    for (const auto token :
         request.eos_tokens) {
        if (static_cast<std::uint64_t>(token) >=
            model.vocab_size) {
            throw std::invalid_argument(
                "sampled speculation EOS token is outside model vocabulary");
        }
    }

    SampledSpeculationCyclePlan plan;
    plan.candidate_length =
        static_cast<std::uint32_t>(
            depth);
    plan.verify_state_advance =
        plan.candidate_length;
    plan.speculation_used =
        depth != 0;
    plan.snapshot_required =
        plan.speculation_used;
    plan.base_position =
        preparation.base_position;
    plan.base_committed_tokens =
        preparation.committed_tokens;

    for (std::size_t index = 0;
         index < depth;
         ++index) {
        SpeculativeSamplingStepRequest step;
        step.target_probabilities =
            request.target_probabilities[index];
        step.draft_probabilities =
            request.draft_probabilities[index];
        step.draft_token =
            preparation.candidate_tokens[index];
        step.rng =
            request.rng;
        step.absolute_position =
            checked_add(
                preparation.base_position,
                static_cast<std::uint64_t>(
                    index),
                "sampled speculation RNG position overflow");

        const auto verified =
            verify_speculative_sample(
                step);
        if (!verified.accepted) {
            plan.rejected = true;
            plan.used_residual =
                verified.used_residual;
            plan.emitted_tokens.push_back(
                verified.token);
            break;
        }

        plan.emitted_tokens.push_back(
            preparation.candidate_tokens[index]);
        ++plan.accepted_candidates;

        if (is_eos(
                preparation.candidate_tokens[index],
                request.eos_tokens)) {
            break;
        }

        ++plan.committed_state_advance;
    }

    if (!plan.rejected &&
        plan.accepted_candidates ==
            plan.candidate_length &&
        plan.emitted_tokens.size() <
            request.token_budget) {
        const auto bonus =
            sample_speculation_target_token(
                request.target_probabilities[depth],
                request.rng,
                checked_add(
                    preparation.base_position,
                    static_cast<std::uint64_t>(
                        depth),
                    "sampled speculation bonus position overflow"));
        plan.emitted_tokens.push_back(
            bonus);
    }

    if (plan.emitted_tokens.empty()) {
        const auto serial =
            sample_speculation_target_token(
                request.target_probabilities[0],
                request.rng,
                preparation.base_position);
        plan.emitted_tokens.push_back(
            serial);
    }

    if (plan.speculation_used) {
        if (preparation.pre_window_snapshot.plan_identity == 0 ||
            preparation.pre_window_snapshot.next_position !=
                preparation.base_position ||
            preparation.pre_window_snapshot.compiled_execution_identity !=
                preparation.compiled_execution_identity ||
            preparation.pre_window_snapshot.runtime_compatibility_identity !=
                preparation.runtime_compatibility_identity) {
            throw std::invalid_argument(
                "positive-depth sampled speculation requires matching pre-window snapshot");
        }

        plan.restore_required =
            plan.committed_state_advance !=
                plan.candidate_length;
        if (plan.restore_required) {
            plan.replay_tokens.assign(
                preparation.candidate_tokens.begin(),
                preparation.candidate_tokens.begin() +
                    static_cast<std::ptrdiff_t>(
                        plan.committed_state_advance));
        }
        plan.pre_window_snapshot =
            preparation.pre_window_snapshot;
    }

    plan.post_cycle_state_position =
        checked_add(
            preparation.base_position,
            plan.committed_state_advance,
            "sampled speculation state position overflow");
    plan.post_cycle_output_position =
        checked_add(
            preparation.base_position,
            static_cast<std::uint64_t>(
                plan.emitted_tokens.size()),
            "sampled speculation output position overflow");
    if (plan.post_cycle_output_position >
        model.max_position_embeddings) {
        throw std::out_of_range(
            "sampled speculation output exceeds model context");
    }
    plan.post_cycle_committed_tokens =
        checked_add(
            preparation.committed_tokens,
            plan.committed_state_advance,
            "sampled speculation committed-token overflow");

    return plan;
}

}  // namespace strix
