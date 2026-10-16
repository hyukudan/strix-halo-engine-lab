#pragma once

#include <cstdint>
#include <array>
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

    // Zero denotes the serial/no-drafter baseline or a legacy generic
    // proposal. Distinct positive identities allow multiple drafters at the
    // same candidate depth to compete on measured complete-cycle cost.
    std::uint64_t proposal_identity = 0;
};

struct SpeculationControllerPolicy {
    std::optional<std::uint32_t> current_depth;
    std::optional<std::uint64_t> current_proposal_identity;

    // A new depth must improve cost/token by at least this fraction before
    // replacing the current qualified choice.
    double switch_hysteresis_fraction = 0.03;
};

struct SpeculationDepthCost {
    std::uint32_t depth = 0;
    double cycle_seconds = 0.0;
    double expected_committed_tokens = 0.0;
    double seconds_per_committed_token = 0.0;
    std::uint64_t proposal_identity = 0;
};

struct SpeculationDecision {
    std::uint32_t selected_depth = 0;
    std::uint64_t selected_proposal_identity = 0;
    bool speculation_enabled = false;
    bool switched = false;
    std::vector<SpeculationDepthCost> qualified_costs;
};

[[nodiscard]] SpeculationDecision choose_speculation_depth(
    std::span<const SpeculationDepthSample> samples,
    const SpeculationControllerPolicy& policy = {});

// Online cost eligibility is independent of numerical/native qualification.
// Only complete, successfully consumed cycles enter these fixed-size totals.
struct ObservedSpeculationPolicy {
    std::uint32_t bootstrap_samples_per_depth = 2;
    std::uint32_t reexplore_interval = 32;
    double switch_hysteresis_fraction = 0.03;
};

struct ObservedSpeculationDepthCost {
    std::uint64_t samples = 0;
    std::uint64_t committed_tokens = 0;
    double complete_cycle_seconds = 0.0;
};

// A caller resets this bounded controller when cost conditions change. The
// native lookup owner supplies the real completed-cycle wall measurement; this
// class neither starts work nor grants correctness qualification to a depth.
class ObservedSpeculationController {
public:
    explicit ObservedSpeculationController(ObservedSpeculationPolicy policy = {});
    [[nodiscard]] std::uint32_t select(std::uint32_t maximum_depth);
    // No allocation or exception after a target commit. Invalid/censored
    // measurements do not alter cost totals or advance exploration.
    [[nodiscard]] bool observe(std::uint32_t depth, double complete_cycle_seconds,
                               std::uint64_t committed_tokens) noexcept;
    void cancel_observation() noexcept;
    void reset() noexcept;
    [[nodiscard]] const std::array<ObservedSpeculationDepthCost, 3>& costs() const noexcept { return costs_; }
private:
    ObservedSpeculationPolicy policy_;
    std::array<ObservedSpeculationDepthCost, 3> costs_{};
    std::uint64_t since_exploration_ = 0;
    std::uint32_t preferred_depth_ = 0, exploration_cursor_ = 0, selected_depth_ = 0;
    bool pending_ = false, exploring_ = false;
};

}  // namespace strix
