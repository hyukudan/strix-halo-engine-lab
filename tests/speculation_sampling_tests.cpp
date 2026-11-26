#include "strix/speculation_sampling.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::SpeculationRngIdentity rng() {
    return {1234, 0x1111ULL, 0x2222ULL};
}

void check_binary64_golden() {
    constexpr auto identity = strix::SpeculationProbabilityIdentity::binary64_v1;
    const auto bits = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    // At position 1 the draw is 0x1.e36df62d0e73ep-2. Dropping the two
    // half-ulp addends at mass 1 would leave token 0 above that boundary.
    constexpr std::array<double, 4> cdf_row{0x1.e36df62d0e73fp-2, 0x1.0e4904e978c60p-1, 0x1p-53, 0x1p-53};
    expect(strix::sample_speculation_target_token(cdf_row, rng(), 1, 0, identity) == 1,
           "binary64 bonus crosses the compensated mass boundary");

    constexpr std::array<double, 2> halfway{1.0, 0x1p-53};
    constexpr std::array<double, 2> point{1.0, 0.0};
    const auto accepted = strix::verify_speculative_sample({halfway, point, 0, rng(), 0, 0, 0, identity});
    // An extended-precision denominator would produce the next double below 1.
    expect(accepted.token == 0 && accepted.accepted && !accepted.used_residual &&
               bits(accepted.acceptance_probability) == 0x3ff0000000000000ULL &&
               bits(accepted.acceptance_uniform) == 0x3fc19df882fbb40cULL,
           "binary64 acceptance uses the rounded halfway denominator and the declared draw");

    constexpr std::array<double, 5> row{0.25, 0.5, 0.25, 0x1p-53, 0x1p-53};
    constexpr std::array<double, 5> draft{1.0, 0.0, 0.0, 0.0, 0.0};
    const auto rejected = strix::verify_speculative_sample({row, draft, 0, rng(), 7, 0, 0, identity});
    expect(rejected.token == 1 && !rejected.accepted && rejected.used_residual &&
               bits(rejected.acceptance_probability) == 0x3fcffffffffffffeULL &&
               bits(rejected.acceptance_uniform) == 0x3fd30c990185bf5aULL &&
               bits(rejected.residual_mass) == 0x3fe8000000000000ULL &&
               bits(rejected.residual_uniform) == 0x3fc92a1f9d7d7440ULL,
           "binary64 rejection keeps the residual decision and all floating fields bitwise");

    const int original = std::fegetround();
    if (std::fesetround(FE_DOWNWARD) == 0) {
        bool refused = false;
        try {
            static_cast<void>(strix::sample_speculation_target_token(cdf_row, rng(), 1, 0, identity));
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        static_cast<void>(std::fesetround(original));
        expect(refused, "binary64 speculation refuses a non-nearest rounding environment");
    }
}

strix::SpeculativeSamplingStepResult legacy_verify(const strix::SpeculativeSamplingStepRequest& request) {
    const auto normalize = [](std::span<const double> input) {
        long double total = 0.0L;
        for (const auto weight : input) {
            total += weight;
        }
        std::vector<double> output;
        for (const auto weight : input) {
            output.push_back(static_cast<double>(weight / total));
        }
        return output;
    };
    const auto p = normalize(request.target_probabilities);
    const auto q = normalize(request.draft_probabilities);
    using Draw = strix::SpeculationRandomDrawClass;
    strix::SpeculativeSamplingStepResult result;
    result.acceptance_probability = std::min(1.0, p[request.draft_token] / q[request.draft_token]);
    result.acceptance_uniform = strix::speculation_uniform01(request.rng, request.absolute_position,
        Draw::draft_acceptance, request.acceptance_draw_index);
    result.accepted = result.acceptance_uniform < result.acceptance_probability;
    result.token = request.draft_token;
    if (result.accepted) {
        return result;
    }

    result.used_residual = true;
    std::vector<double> residual(p.size());
    long double mass = 0.0L;
    for (std::size_t i = 0; i < p.size(); ++i) {
        residual[i] = std::max(0.0, p[i] - q[i]);
        mass += residual[i];
    }
    result.residual_mass = static_cast<double>(mass);
    result.residual_uniform = strix::speculation_uniform01(request.rng, request.absolute_position,
        Draw::residual_fallback, request.residual_draw_index);
    long double cumulative = 0.0L;
    for (std::size_t i = 0; i < residual.size(); ++i) {
        if (residual[i] > 0.0) {
            result.token = static_cast<std::uint32_t>(i);
        }
        cumulative += static_cast<double>(residual[i] / mass);
        if (result.residual_uniform < cumulative) {
            break;
        }
    }
    return result;
}

void check_legacy_parity() {
    using Draw = strix::SpeculationRandomDrawClass;
    std::size_t residuals = 0;
    double max_mass_delta = 0.0;
    bool match = true;
    for (std::uint32_t trial = 0; trial < 10'000; ++trial) {
        const strix::SpeculationRngIdentity identity{2718, trial + 1U, 90210};
        const auto vocab = 3U + trial % 34U;
        std::vector<double> p(vocab);
        std::vector<double> q(vocab);
        for (std::uint32_t token = 0; token < vocab; ++token) {
            p[token] = static_cast<double>(
                strix::speculation_random_u64(identity, token, Draw::draft_proposal, 1) % 19U);
            q[token] = static_cast<double>(
                strix::speculation_random_u64(identity, token, Draw::draft_proposal, 2) % 19U);
        }
        p[0] += 1.0;
        const auto proposed = trial % vocab;
        q[proposed] += 1.0;
        const strix::SpeculativeSamplingStepRequest request{p, q, proposed, identity, 100U + trial,
            trial % 13U, trial % 17U};
        const auto old = legacy_verify(request);
        const auto next = strix::verify_speculative_sample(request);
        match = match && old.token == next.token && old.accepted == next.accepted &&
            old.used_residual == next.used_residual &&
            old.acceptance_probability == next.acceptance_probability &&
            old.acceptance_uniform == next.acceptance_uniform &&
            old.residual_uniform == next.residual_uniform;
        max_mass_delta = std::max(max_mass_delta, std::abs(old.residual_mass - next.residual_mass));
        if (next.used_residual) {
            ++residuals;
        }
    }
    expect(match && residuals > 0 && max_mass_delta == 0.0,
           "shared decision matches all legacy result fields over 10,000 random cases");
    std::cout << "Legacy parity: 10000 cases, " << residuals << " residuals, max mass delta "
              << max_mass_delta << '\n';
}

}  // namespace

int main() {
    check_binary64_golden();
    check_legacy_parity();
    {
        const std::vector<double> target{
            0.1, 0.6, 0.3,
        };
        const auto token =
            strix::sample_speculation_target_token(
                target,
                rng(),
                7);
        expect(
            token < target.size(),
            "serial target categorical sample stays inside vocabulary");
        expect(
            token ==
                strix::sample_speculation_target_token(
                    target,
                    rng(),
                    7),
            "serial target categorical sample is reproducible by position");
    }

    {
        const std::vector<double> target{
            0.2, 0.5, 0.3,
        };
        const std::vector<double> draft{
            0.1, 0.5, 0.4,
        };
        strix::SpeculativeSamplingStepRequest request;
        request.target_probabilities = target;
        request.draft_probabilities = draft;
        request.draft_token = 0;
        request.rng = rng();
        request.absolute_position = 8;

        const auto result =
            strix::verify_speculative_sample(
                request);
        expect(
            result.accepted &&
                !result.used_residual &&
                result.token == 0 &&
                result.acceptance_probability == 1.0,
            "p>=q accepts draft token with probability one");
    }

    {
        const std::vector<double> target{
            0.0, 1.0, 0.0,
        };
        const std::vector<double> draft{
            1.0, 0.0, 0.0,
        };
        strix::SpeculativeSamplingStepRequest request;
        request.target_probabilities = target;
        request.draft_probabilities = draft;
        request.draft_token = 0;
        request.rng = rng();
        request.absolute_position = 9;

        const auto result =
            strix::verify_speculative_sample(
                request);
        expect(
            !result.accepted &&
                result.used_residual &&
                result.token == 1 &&
                result.acceptance_probability == 0.0 &&
                std::abs(
                    result.residual_mass -
                    1.0) < 1e-12,
            "certain rejection samples the exact residual support");
    }

    {
        const std::vector<double> target{
            1.0, 3.0,
        };
        const std::vector<double> draft{
            2.0, 2.0,
        };
        strix::SpeculativeSamplingStepRequest request;
        request.target_probabilities = target;
        request.draft_probabilities = draft;
        request.draft_token = 0;
        request.rng = rng();
        request.absolute_position = 10;

        const auto result =
            strix::verify_speculative_sample(
                request);
        expect(
            std::abs(
                result.acceptance_probability -
                0.5) < 1e-12,
            "acceptance probability uses normalized p/q for unnormalized inputs");
    }

    {
        const std::vector<double> target{
            0.5, 0.5,
        };
        const std::vector<double> draft{
            0.0, 1.0,
        };
        strix::SpeculativeSamplingStepRequest request;
        request.target_probabilities = target;
        request.draft_probabilities = draft;
        request.draft_token = 0;
        request.rng = rng();

        bool rejected = false;
        try {
            static_cast<void>(
                strix::verify_speculative_sample(
                    request));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected,
            "draft token with zero draft probability is rejected");
    }

    {
        const std::vector<double> bad{
            0.5,
            std::numeric_limits<double>::
                quiet_NaN(),
        };
        bool rejected = false;
        try {
            static_cast<void>(
                strix::sample_speculation_target_token(
                    bad,
                    rng(),
                    0));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected,
            "categorical sampler rejects non-finite probabilities");
    }

    if (failures == 0) {
        std::cout
            << "speculation sampling tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
