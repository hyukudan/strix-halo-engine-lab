#pragma once

#include "strix/model_contract.hpp"
#include "strix/session_snapshot.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/speculation_rng.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct SampledSpeculationCycleRequest {
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t base_position = 0;
    std::uint64_t committed_tokens = 0;
    PersistentStateGenerations state;
    std::span<const std::uint32_t> candidate_tokens;
    std::uint32_t token_budget = 1;
    std::span<const std::uint32_t> eos_tokens;
    std::uint64_t snapshot_alignment = 64;
};

struct SampledSpeculationPreparation {
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t base_position = 0;
    std::uint64_t committed_tokens = 0;
    PersistentStateGenerations state;
    std::vector<std::uint32_t> candidate_tokens;
    SerialSessionSnapshotPlan pre_window_snapshot;
};

struct SampledSpeculationFinalizeRequest {
    std::span<const std::vector<double>> target_probabilities;
    std::span<const std::vector<double>> draft_probabilities;
    SpeculationRngIdentity rng;
    std::uint32_t token_budget = 1;
    std::span<const std::uint32_t> eos_tokens;
};

struct SampledSpeculationCyclePlan {
    std::uint32_t candidate_length = 0;
    std::uint32_t accepted_candidates = 0;
    std::uint32_t committed_state_advance = 0;
    std::uint32_t verify_state_advance = 0;
    bool speculation_used = false;
    bool rejected = false;
    bool used_residual = false;
    bool snapshot_required = false;
    bool restore_required = false;
    std::vector<std::uint32_t> replay_tokens;
    std::vector<std::uint32_t> emitted_tokens;
    std::uint64_t base_position = 0;
    std::uint64_t post_cycle_state_position = 0;
    std::uint64_t post_cycle_output_position = 0;
    std::uint64_t base_committed_tokens = 0;
    std::uint64_t post_cycle_committed_tokens = 0;
    SerialSessionSnapshotPlan pre_window_snapshot;
};

[[nodiscard]] SampledSpeculationPreparation
prepare_sampled_speculation_cycle(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const SampledSpeculationCycleRequest& request);

[[nodiscard]] SampledSpeculationCyclePlan
finalize_sampled_speculation_cycle(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const SampledSpeculationPreparation& preparation,
    const SampledSpeculationFinalizeRequest& request);

}  // namespace strix
