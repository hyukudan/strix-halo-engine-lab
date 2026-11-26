#pragma once

#include "strix/speculation_rng.hpp"

#include <cstdint>
#include <span>

namespace strix {

// The legacy identity uses the platform's long-double format. The explicitly
// selected binary64 identity defines the arithmetic for supplied probability rows.
enum class SpeculationProbabilityIdentity : std::uint8_t {
    legacy,
    binary64_v1,
};

struct SpeculativeSamplingStepRequest {
    std::span<const double> target_probabilities;
    std::span<const double> draft_probabilities;
    std::uint32_t draft_token = 0;
    SpeculationRngIdentity rng;
    std::uint64_t absolute_position = 0;
    std::uint32_t acceptance_draw_index = 0;
    std::uint32_t residual_draw_index = 0;
    SpeculationProbabilityIdentity probability_identity = SpeculationProbabilityIdentity::legacy;
};

struct SpeculativeSamplingStepResult {
    bool accepted = false;
    bool used_residual = false;
    std::uint32_t token = 0;
    double acceptance_probability = 0.0;
    double acceptance_uniform = 0.0;
    double residual_mass = 0.0;
    double residual_uniform = 0.0;
};

[[nodiscard]] std::uint32_t sample_speculation_target_token(
    std::span<const double> target_probabilities,
    const SpeculationRngIdentity& rng,
    std::uint64_t absolute_position,
    std::uint32_t draw_index = 0,
    SpeculationProbabilityIdentity probability_identity = SpeculationProbabilityIdentity::legacy);

[[nodiscard]] SpeculativeSamplingStepResult
verify_speculative_sample(
    const SpeculativeSamplingStepRequest& request);

}  // namespace strix
