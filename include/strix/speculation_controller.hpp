#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace strix {

struct SpeculationDepthSample {
    std::uint32_t depth = 0;
    bool qualified = false;

    double proposal_seconds = 0.0;
    double verification_seconds = 0.0;
    double commit_replay_seconds = 0.0;
    double uncovered_communication_seconds = 0.0;
    double control_seconds = 0.0;

    // Expected number of tokens committed by one completed cycle.
    // Depth zero should use exactly 1.
    double expected_committed_tokens = 0.0;
};

struct SpeculationControllerPolicy {
    std::optional<std::uint32_t> current_depth;

    // A new depth must improve cost/token by at least this fraction before
    // replacing the current qualified choice.
    double switch_hysteresis_fraction = 0.03;
};

struct SpeculationDepthCost {
    std::uint32_t depth = 0;
    double cycle_seconds = 0.0;
    double expected_committed_tokens = 0.0;
    double seconds_per_committed_token = 0.0;
};

struct SpeculationDecision {
    std::uint32_t selected_depth = 0;
    bool speculation_enabled = false;
    bool switched = false;
    std::vector<SpeculationDepthCost> qualified_costs;
};

[[nodiscard]] SpeculationDecision choose_speculation_depth(
    std::span<const SpeculationDepthSample> samples,
    const SpeculationControllerPolicy& policy = {});

}  // namespace strix
