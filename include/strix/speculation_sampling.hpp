#pragma once

#include "strix/speculation_rng.hpp"

#include <cstdint>
#include <span>

namespace strix {

struct SpeculativeSamplingStepRequest {
    std::span<const double> target_probabilities;
    std::span<const double> draft_probabilities;
    std::uint32_t draft_token = 0;
    SpeculationRngIdentity rng;
    std::uint64_t absolute_position = 0;
    std::uint32_t acceptance_draw_index = 0;
    std::uint32_t residual_draw_index = 0;
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
    std::uint32_t draw_index = 0);

[[nodiscard]] SpeculativeSamplingStepResult
verify_speculative_sample(
    const SpeculativeSamplingStepRequest& request);

}  // namespace strix
