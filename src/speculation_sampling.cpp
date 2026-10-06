#include "strix/speculation_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix {
namespace {

struct NormalizedDistribution {
    std::vector<double> probabilities;
};

NormalizedDistribution normalize(
    std::span<const double> probabilities,
    const char* name) {
    if (probabilities.empty()) {
        throw std::invalid_argument(
            std::string(name) +
            " distribution must be non-empty");
    }

    long double mass = 0.0L;
    for (const auto value : probabilities) {
        if (!std::isfinite(value) ||
            value < 0.0) {
            throw std::invalid_argument(
                std::string(name) +
                " distribution must contain finite non-negative probabilities");
        }
        mass +=
            static_cast<long double>(
                value);
    }

    if (!(mass > 0.0L) ||
        !std::isfinite(
            static_cast<double>(mass))) {
        throw std::invalid_argument(
            std::string(name) +
            " distribution must have finite positive mass");
    }

    NormalizedDistribution result;
    result.probabilities.reserve(
        probabilities.size());
    for (const auto value : probabilities) {
        result.probabilities.push_back(
            static_cast<double>(
                static_cast<long double>(
                    value) /
                mass));
    }
    return result;
}

std::uint32_t sample_normalized(
    std::span<const double> probabilities,
    double uniform) {
    if (!std::isfinite(uniform) ||
        uniform < 0.0 ||
        uniform >= 1.0) {
        throw std::invalid_argument(
            "categorical uniform must lie in [0,1)");
    }

    long double cumulative = 0.0L;
    std::uint32_t last_positive = 0;
    bool have_positive = false;
    for (std::size_t index = 0;
         index < probabilities.size();
         ++index) {
        const auto probability =
            probabilities[index];
        if (probability > 0.0) {
            last_positive =
                static_cast<std::uint32_t>(
                    index);
            have_positive = true;
        }
        cumulative +=
            static_cast<long double>(
                probability);
        if (static_cast<long double>(
                uniform) < cumulative) {
            return static_cast<std::uint32_t>(
                index);
        }
    }

    if (!have_positive) {
        throw std::logic_error(
            "normalized categorical distribution has no positive entry");
    }
    return last_positive;
}

}  // namespace

std::uint32_t sample_speculation_target_token(
    std::span<const double> target_probabilities,
    const SpeculationRngIdentity& rng,
    std::uint64_t absolute_position,
    std::uint32_t draw_index) {
    const auto target =
        normalize(
            target_probabilities,
            "target");
    const auto uniform =
        speculation_uniform01(
            rng,
            absolute_position,
            SpeculationRandomDrawClass::
                serial_target,
            draw_index);
    return sample_normalized(
        target.probabilities,
        uniform);
}

SpeculativeSamplingStepResult
verify_speculative_sample(
    const SpeculativeSamplingStepRequest& request) {
    if (request.target_probabilities.size() !=
        request.draft_probabilities.size()) {
        throw std::invalid_argument(
            "target and draft distributions must have identical vocabulary size");
    }
    if (request.target_probabilities.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()) +
            static_cast<std::size_t>(1U)) {
        throw std::overflow_error(
            "speculative sampling vocabulary exceeds uint32 token ids");
    }
    if (static_cast<std::size_t>(
            request.draft_token) >=
        request.target_probabilities.size()) {
        throw std::out_of_range(
            "draft token exceeds speculative sampling vocabulary");
    }

    const auto target =
        normalize(
            request.target_probabilities,
            "target");
    const auto draft =
        normalize(
            request.draft_probabilities,
            "draft");

    const auto q =
        draft.probabilities[
            request.draft_token];
    if (!(q > 0.0)) {
        throw std::invalid_argument(
            "draft token must have positive probability under draft distribution");
    }
    const auto p =
        target.probabilities[
            request.draft_token];

    SpeculativeSamplingStepResult result;
    result.acceptance_probability =
        std::min(
            1.0,
            p / q);
    result.acceptance_uniform =
        speculation_uniform01(
            request.rng,
            request.absolute_position,
            SpeculationRandomDrawClass::
                draft_acceptance,
            request.acceptance_draw_index);

    if (result.acceptance_uniform <
        result.acceptance_probability) {
        result.accepted = true;
        result.token =
            request.draft_token;
        return result;
    }

    std::vector<double> residual(
        target.probabilities.size(),
        0.0);
    long double residual_mass = 0.0L;
    for (std::size_t index = 0;
         index < residual.size();
         ++index) {
        residual[index] =
            std::max(
                0.0,
                target.probabilities[index] -
                    draft.probabilities[index]);
        residual_mass +=
            static_cast<long double>(
                residual[index]);
    }

    if (!(residual_mass > 0.0L)) {
        throw std::logic_error(
            "rejected speculative sample has zero residual probability mass");
    }

    result.used_residual = true;
    result.residual_mass =
        static_cast<double>(
            residual_mass);
    for (auto& value : residual) {
        value =
            static_cast<double>(
                static_cast<long double>(
                    value) /
                residual_mass);
    }

    result.residual_uniform =
        speculation_uniform01(
            request.rng,
            request.absolute_position,
            SpeculationRandomDrawClass::
                residual_fallback,
            request.residual_draw_index);
    result.token =
        sample_normalized(
            residual,
            result.residual_uniform);
    return result;
}

}  // namespace strix
