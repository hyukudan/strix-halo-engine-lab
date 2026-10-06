#include "strix/speculation_sampling.hpp"

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

}  // namespace

int main() {
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
