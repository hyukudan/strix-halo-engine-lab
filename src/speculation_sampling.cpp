#include "strix/speculation_sampling.hpp"

#include "strix/speculation_multi_draft.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace strix {
namespace {

bool use_binary64(SpeculationProbabilityIdentity identity) {
    switch (identity) {
    case SpeculationProbabilityIdentity::legacy:
        return false;
    case SpeculationProbabilityIdentity::binary64_v1: {
        if (!std::numeric_limits<double>::is_iec559 || FLT_EVAL_METHOD != 0 || std::fegetround() != FE_TONEAREST) {
            throw std::invalid_argument("binary64-v1 speculation requires binary64 round-to-nearest-even");
        }
        const volatile double tiny = std::numeric_limits<double>::denorm_min();
        const volatile double two = 2.0;
        if (tiny * two == 0.0) {
            throw std::invalid_argument("binary64-v1 speculation requires gradual subnormals");
        }
        return true;
    }
    }
    throw std::invalid_argument("unknown speculation probability identity");
}

template <typename Float>
std::vector<double> normalize(
    std::span<const double> probabilities,
    const char* name) {
    if (probabilities.empty()) {
        throw std::invalid_argument(
            std::string(name) +
            " distribution must be non-empty");
    }

    Float mass = 0;
    [[maybe_unused]] double lost = 0.0;
    for (const auto value : probabilities) {
        if (!std::isfinite(value) ||
            value < 0.0) {
            throw std::invalid_argument(
                std::string(name) +
                " distribution must contain finite non-negative probabilities");
        }
        if constexpr (std::is_same_v<Float, double>) {
            // Kahan summation in token order, using binary64 for every operation.
            const double adjusted = value - lost;
            const double updated = mass + adjusted;
            lost = (updated - mass) - adjusted;
            mass = updated;
        } else {
            mass += static_cast<Float>(value);
        }
    }

    if (!(mass > 0.0L) ||
        !std::isfinite(
            static_cast<double>(mass))) {
        throw std::invalid_argument(
            std::string(name) +
            " distribution must have finite positive mass");
    }

    std::vector<double> result;
    result.reserve(
        probabilities.size());
    for (const auto value : probabilities) {
        result.push_back(
            static_cast<double>(
                static_cast<Float>(value) / mass));
    }
    return result;
}

std::vector<double> normalize(std::span<const double> probabilities, const char* name, bool binary64) {
    return binary64 ? normalize<double>(probabilities, name) : normalize<long double>(probabilities, name);
}

std::vector<std::uint32_t> dense_token_ids(std::size_t size) {
    if (size != 0 && size - 1U > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("speculative sampling vocabulary exceeds uint32 token ids");
    }
    std::vector<std::uint32_t> ids(size);
    std::iota(ids.begin(), ids.end(), std::uint32_t{0});
    return ids;
}

}  // namespace

std::uint32_t sample_speculation_target_token(
    std::span<const double> target_probabilities,
    const SpeculationRngIdentity& rng,
    std::uint64_t absolute_position,
    std::uint32_t draw_index,
    SpeculationProbabilityIdentity probability_identity) {
    const auto ids = dense_token_ids(target_probabilities.size());
    const auto probabilities = normalize(target_probabilities, "target", use_binary64(probability_identity));
    const auto target = make_token_distribution_from_probabilities(ids, probabilities);
    const auto uniform = speculation_uniform01(rng, absolute_position,
        SpeculationRandomDrawClass::serial_target, draw_index);
    return decide_token(target, nullptr, uniform, 0.0, 0.0).token;
}

SpeculativeSamplingStepResult
verify_speculative_sample(
    const SpeculativeSamplingStepRequest& request) {
    if (request.target_probabilities.size() !=
        request.draft_probabilities.size()) {
        throw std::invalid_argument(
            "target and draft distributions must have identical vocabulary size");
    }
    if (static_cast<std::size_t>(
            request.draft_token) >=
        request.target_probabilities.size()) {
        throw std::out_of_range(
            "draft token exceeds speculative sampling vocabulary");
    }

    const auto ids = dense_token_ids(request.target_probabilities.size());
    const bool binary64 = use_binary64(request.probability_identity);
    const auto target = normalize(request.target_probabilities, "target", binary64);
    const auto draft = normalize(request.draft_probabilities, "draft", binary64);
    const auto target_distribution = make_token_distribution_from_probabilities(ids, target);
    const TokenDraft proposal{request.absolute_position, request.draft_token,
        make_token_distribution_from_probabilities(ids, draft), {}};
    const double q = proposal.distribution.probability(proposal.token);
    if (!(q > 0.0)) {
        throw std::invalid_argument("draft token must have positive probability under draft distribution");
    }

    SpeculativeSamplingStepResult result;
    result.acceptance_probability = std::min(1.0, target_distribution.probability(proposal.token) / q);
    result.acceptance_uniform = speculation_uniform01(request.rng, request.absolute_position,
        SpeculationRandomDrawClass::draft_acceptance, request.acceptance_draw_index);
    const double residual_uniform = speculation_uniform01(request.rng, request.absolute_position,
        SpeculationRandomDrawClass::residual_fallback, request.residual_draw_index);
    const auto decision = decide_token(target_distribution, &proposal, 0.0,
        result.acceptance_uniform, residual_uniform);
    result.accepted = decision.accepted;
    result.used_residual = decision.residual;
    result.token = decision.token;
    if (decision.residual) {
        result.residual_mass = binary64 ? detail::binary64_rejected_target_mass(target_distribution, proposal.distribution)
                                        : detail::rejected_target_mass(target_distribution, proposal.distribution);
        result.residual_uniform = residual_uniform;
    }
    return result;
}

}  // namespace strix
