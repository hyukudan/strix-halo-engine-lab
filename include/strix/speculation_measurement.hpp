#pragma once

#include "strix/speculation_controller.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct SpeculationCycleMeasurement {
    std::uint64_t campaign_identity = 0;
    std::uint64_t proposal_identity = 0;
    std::uint32_t depth = 0;
    bool correctness_passed = false;

    double proposal_seconds = 0.0;
    double verification_seconds = 0.0;
    double commit_replay_seconds = 0.0;
    double uncovered_communication_seconds = 0.0;
    double control_seconds = 0.0;
    double committed_tokens = 0.0;
};

struct SpeculationMeasurementPolicy {
    std::uint64_t campaign_identity = 0;
    std::uint32_t minimum_repetitions = 3;
};

[[nodiscard]] std::vector<SpeculationDepthSample>
build_speculation_samples_from_measurements(
    std::span<const SpeculationCycleMeasurement> measurements,
    const SpeculationMeasurementPolicy& policy);

}  // namespace strix
