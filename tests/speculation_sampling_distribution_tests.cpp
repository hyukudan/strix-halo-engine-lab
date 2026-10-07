#include "strix/speculation_rng.hpp"
#include "strix/speculation_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::uint32_t sample_categorical(
    std::span<const double> probabilities,
    double uniform) {
    double cumulative = 0.0;
    for (std::size_t index = 0;
         index < probabilities.size();
         ++index) {
        cumulative += probabilities[index];
        if (uniform < cumulative) {
            return static_cast<std::uint32_t>(index);
        }
    }
    return static_cast<std::uint32_t>(
        probabilities.size() - 1U);
}

}  // namespace

int main() {
    constexpr std::uint64_t samples = 100000;
    constexpr double frequency_tolerance = 0.01;
    constexpr double acceptance_tolerance = 0.01;

    const std::array<double, 4> target{
        0.10, 0.20, 0.30, 0.40,
    };
    const std::array<double, 4> draft{
        0.40, 0.10, 0.40, 0.10,
    };

    std::array<std::uint64_t, 4> output_counts{};
    std::uint64_t accepted = 0;

    for (std::uint64_t sample = 0;
         sample < samples;
         ++sample) {
        strix::SpeculationRngIdentity target_rng;
        target_rng.seed = 0x123456789abcdef0ULL;
        target_rng.request_identity = sample + 1ULL;
        target_rng.sampling_policy_identity =
            0x1111222233334444ULL;

        auto draft_rng = target_rng;
        draft_rng.sampling_policy_identity =
            0x5555666677778888ULL;

        const auto draft_uniform =
            strix::speculation_uniform01(
                draft_rng,
                17,
                strix::SpeculationRandomDrawClass::
                    serial_target);
        const auto draft_token =
            sample_categorical(
                draft,
                draft_uniform);

        strix::SpeculativeSamplingStepRequest request;
        request.target_probabilities = target;
        request.draft_probabilities = draft;
        request.draft_token = draft_token;
        request.rng = target_rng;
        request.absolute_position = 17;

        const auto result =
            strix::verify_speculative_sample(
                request);
        ++output_counts[result.token];
        if (result.accepted) {
            ++accepted;
        }
    }

    for (std::size_t index = 0;
         index < target.size();
         ++index) {
        const auto observed =
            static_cast<double>(
                output_counts[index]) /
            static_cast<double>(
                samples);
        expect(
            std::abs(
                observed -
                target[index]) <=
                frequency_tolerance,
            "speculative sampled output marginal matches target distribution");
    }

    double expected_acceptance = 0.0;
    for (std::size_t index = 0;
         index < target.size();
         ++index) {
        expected_acceptance +=
            std::min(
                target[index],
                draft[index]);
    }
    const auto observed_acceptance =
        static_cast<double>(accepted) /
        static_cast<double>(samples);
    expect(
        std::abs(
            observed_acceptance -
            expected_acceptance) <=
            acceptance_tolerance,
        "speculative draft acceptance rate matches sum(min(p,q))");

    const std::array<double, 3> equal{
        0.2, 0.3, 0.5,
    };
    for (std::uint64_t sample = 0;
         sample < 1024;
         ++sample) {
        strix::SpeculationRngIdentity identity;
        identity.seed = 99;
        identity.request_identity =
            0x10000ULL + sample;
        identity.sampling_policy_identity =
            0x20000ULL;

        const auto uniform =
            strix::speculation_uniform01(
                identity,
                23,
                strix::SpeculationRandomDrawClass::
                    serial_target);
        const auto token =
            sample_categorical(
                equal,
                uniform);

        strix::SpeculativeSamplingStepRequest request;
        request.target_probabilities = equal;
        request.draft_probabilities = equal;
        request.draft_token = token;
        request.rng = identity;
        request.absolute_position = 23;

        const auto result =
            strix::verify_speculative_sample(
                request);
        expect(
            result.accepted &&
                !result.used_residual &&
                result.token == token &&
                std::abs(
                    result.acceptance_probability -
                    1.0) < 1e-15,
            "identical target/draft distributions accept every draft sample");
    }

    if (failures == 0) {
        std::cout
            << "speculation sampling distribution tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
