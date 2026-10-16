#pragma once

#include "strix/model_contract.hpp"
#include "strix/session_snapshot.hpp"
#include "strix/speculation.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct GreedySpeculationCycleRequest {
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;

    std::uint64_t base_position = 0;
    std::uint64_t committed_tokens = 0;
    PersistentStateGenerations state;

    std::span<const std::uint32_t> candidate_tokens;
    std::span<const std::uint32_t> target_tokens;
    std::uint32_t token_budget = 1;
    std::span<const std::uint32_t> eos_tokens;

    std::uint64_t snapshot_alignment = 64;
    SessionSnapshotPolicy snapshot_policy = SessionSnapshotPolicy::full_components;
    std::uint64_t max_snapshot_payload_bytes = 0;
};

struct GreedySpeculationPreparation {
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t base_position = 0;
    std::uint64_t committed_tokens = 0;
    PersistentStateGenerations state;
    std::vector<std::uint32_t> candidate_tokens;
    SerialSessionSnapshotPlan pre_window_snapshot;
};

struct GreedySpeculationFinalizeRequest {
    std::span<const std::uint32_t> target_tokens;
    std::uint32_t token_budget = 1;
    std::span<const std::uint32_t> eos_tokens;
};

struct GreedySpeculationCyclePlan {
    GreedyVerificationPlan verification;

    bool speculation_used = false;
    bool snapshot_required = false;
    bool restore_required = false;

    // Number of candidate input tokens evaluated by the target verification
    // path. The bonus target token is selected from logits and is NOT processed
    // into target state by this cycle.
    std::uint32_t verify_state_advance = 0;

    // Number of accepted candidate input tokens that must remain represented in
    // committed target state after the cycle.
    std::uint32_t committed_state_advance = 0;

    // Candidate prefix to replay after restoring the pre-window snapshot.
    std::vector<std::uint32_t> replay_tokens;

    // Continuation exposed by the cycle: accepted candidate prefix plus one
    // target-produced bonus token, subject to EOS/length cropping.
    std::vector<std::uint32_t> emitted_tokens;

    std::uint64_t base_position = 0;
    std::uint64_t post_cycle_state_position = 0;
    std::uint64_t post_cycle_output_position = 0;

    // Serial-session target-state token count. This advances by accepted
    // candidate inputs only; the emitted bonus token is not yet in target state.
    // Serial-session target-state token count. This advances by accepted
    // candidate inputs only; the emitted bonus token is not yet in target state.
    std::uint64_t base_committed_tokens = 0;
    std::uint64_t post_cycle_committed_tokens = 0;

    // Present only for positive candidate depth.
    SerialSessionSnapshotPlan pre_window_snapshot;
};

[[nodiscard]] GreedySpeculationPreparation
prepare_greedy_speculation_cycle(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const GreedySpeculationCycleRequest& request);

[[nodiscard]] GreedySpeculationCyclePlan
finalize_greedy_speculation_cycle(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const GreedySpeculationPreparation& preparation,
    const GreedySpeculationFinalizeRequest& request);

[[nodiscard]] GreedySpeculationCyclePlan
build_greedy_speculation_cycle_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const GreedySpeculationCycleRequest& request);

}  // namespace strix
