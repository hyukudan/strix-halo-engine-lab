#include "strix/speculation_rng.hpp"
#include "strix/speculation.hpp"
#include "strix/speculation_sampling.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

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

struct Interval {
    double begin = 0.0;
    double end = 0.0;
};

std::vector<Interval> cdf_intervals(std::span<const double> weights, bool descending) {
    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (descending) {
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return weights[a] != weights[b] ? weights[a] > weights[b] : a < b;
        });
    }
    std::vector<Interval> intervals(weights.size());
    double cumulative = 0.0;
    for (const auto token : order) {
        intervals[token].begin = cumulative;
        cumulative += weights[token];
        intervals[token].end = cumulative;
    }
    return intervals;
}

double cdf_agreement(std::span<const double> p, std::span<const double> q, bool descending) {
    const auto a = cdf_intervals(p, descending);
    const auto b = cdf_intervals(q, descending);
    double agreement = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        agreement += std::max(0.0, std::min(a[i].end, b[i].end) - std::max(a[i].begin, b[i].begin));
    }
    return agreement;
}

double gumbel_agreement(std::span<const double> p, std::span<const double> q) {
    // Shared Gumbels are shared exponential clocks. Token i wins both races
    // with probability 1 / sum_j max(p[j]/p[i], q[j]/q[i]).
    std::vector<std::size_t> common;
    long double p_right = 0.0L;
    long double q_left = 0.0L;
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (p[i] > 0.0) {
            p_right += p[i];
            if (q[i] > 0.0) {
                common.push_back(i);
            }
        } else {
            q_left += q[i];
        }
    }
    // Split the maximum by the likelihood ratio to evaluate every denominator in one sweep.
    std::sort(common.begin(), common.end(), [&](std::size_t a, std::size_t b) {
        return p[a] / q[a] < p[b] / q[b];
    });
    long double agreement = 0.0L;
    for (const auto token : common) {
        agreement += 1.0L / (p_right / p[token] + q_left / q[token]);
        p_right -= p[token];
        q_left += q[token];
    }
    return static_cast<double>(agreement);
}

std::vector<double> dense(const strix::TokenDistribution& distribution, std::size_t vocab) {
    std::vector<double> result(vocab);
    for (std::size_t i = 0; i < distribution.tokens().size(); ++i) {
        result[distribution.tokens()[i]] = distribution.probabilities()[i];
    }
    return result;
}

void print_coupling_table() {
    constexpr strix::SpeculationRngIdentity identity{1729, 21, 1};
    using Draw = strix::SpeculationRandomDrawClass;
    std::cout << "\nV | sigma | T | top_p | argmax | token-ID CDF | ranked CDF | rejection | shared Gumbel\n"
                 "---:|---:|---:|---:|---:|---:|---:|---:|---:\n"
              << std::fixed << std::setprecision(6);
    for (const std::size_t vocab : {32768U, 1024U}) {
        std::vector<float> logits(vocab);
        std::vector<double> noise(vocab);
        for (std::size_t i = 0; i < vocab; ++i) {
            logits[i] = static_cast<float>(-std::log(static_cast<double>(i + 1U)));
        }
        for (std::size_t i = vocab - 1U; i > 0; --i) {
            const auto j = strix::speculation_random_u64(identity, i, Draw::draft_proposal) % (i + 1U);
            std::swap(logits[i], logits[static_cast<std::size_t>(j)]);
        }
        for (std::size_t i = 0; i < vocab; ++i) {
            const double radius = std::sqrt(-2.0 * std::log(
                1.0 - strix::speculation_uniform01(identity, i, Draw::draft_proposal, 1)));
            noise[i] = radius * std::cos(6.2831853071795864769 *
                strix::speculation_uniform01(identity, i, Draw::draft_proposal, 2));
        }
        for (const double sigma : {0.1, 0.3, 1.0}) {
            std::vector<float> perturbed(vocab);
            for (std::size_t i = 0; i < vocab; ++i) {
                perturbed[i] = static_cast<float>(static_cast<double>(logits[i]) + sigma * noise[i]);
            }
            for (const float temperature : {0.7F, 1.0F}) {
                for (const float top_p : {1.0F, 0.9F}) {
                    strix::TokenSamplingPolicy policy;
                    policy.temperature = temperature;
                    policy.top_p = top_p;
                    const auto target = dense(strix::make_token_distribution(logits, {}, policy), vocab);
                    const auto draft = dense(strix::make_token_distribution(perturbed, {}, policy), vocab);
                    const auto draft_argmax = static_cast<std::size_t>(
                        std::max_element(draft.begin(), draft.end()) - draft.begin());
                    double overlap = 0.0;
                    for (std::size_t i = 0; i < vocab; ++i) {
                        overlap += std::min(target[i], draft[i]);
                    }
                    std::cout << vocab << " | " << sigma << " | " << temperature << " | " << top_p
                              << " | " << target[draft_argmax] << " | " << cdf_agreement(target, draft, false)
                              << " | " << cdf_agreement(target, draft, true) << " | " << overlap
                              << " | " << gumbel_agreement(target, draft) << '\n';
                }
            }
        }
    }
}

void check_chain_joint() {
    // Index 3 is the start-of-sequence row. Rows 0..2 condition on the last emitted token.
    constexpr std::size_t vocab = 3;
    constexpr std::uint32_t trials = 200'000;
    constexpr std::array<std::uint32_t, vocab> ids{0, 1, 2};
    constexpr std::array<std::array<double, vocab>, vocab + 1U> target{{
        {0.50, 0.25, 0.25}, {0.25, 0.50, 0.25}, {0.25, 0.25, 0.50}, {0.25, 0.375, 0.375},
    }};
    constexpr std::array<std::array<double, vocab>, vocab + 1U> draft{{
        {0.25, 0.25, 0.50}, {0.375, 0.375, 0.25}, {0.50, 0.25, 0.25}, {0.50, 0.25, 0.25},
    }};
    std::array<strix::TokenDistribution, vocab + 1U> p;
    std::array<strix::TokenDraft, vocab + 1U> q;
    for (std::size_t i = 0; i < p.size(); ++i) {
        p[i] = strix::make_token_distribution_from_probabilities(ids, target[i]);
        q[i].distribution = strix::make_token_distribution_from_probabilities(ids, draft[i]);
    }

    std::array<std::uint32_t, vocab * vocab * vocab> counts{};
    std::uint32_t full_window_bonus = 0;
    std::uint32_t rejected_windows = 0;
    using Draw = strix::SpeculationRandomDrawClass;
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        const strix::SpeculationRngIdentity rng{877, trial + 1U, 12};
        std::array<std::uint32_t, 3> tokens{};
        std::size_t emitted = 0;
        std::size_t prior = vocab;
        while (emitted < tokens.size()) {
            const auto depth = tokens.size() - emitted - 1U;
            std::array<strix::TokenDraft*, 2> proposals{};
            // Use separate slots because a Markov row can repeat within the draft chain.
            std::array<std::uint32_t, 2> proposed_tokens{};
            auto projected = prior;
            const std::uint64_t start = 51U + emitted;
            for (std::size_t i = 0; i < depth; ++i) {
                proposals[i] = &q[projected];
                proposed_tokens[i] = q[projected].distribution.sample(
                    strix::speculation_uniform01(rng, start + i, Draw::draft_proposal));
                projected = proposed_tokens[i];
            }
            bool rejected = false;
            for (std::size_t i = 0; i < depth; ++i) {
                const auto pos = start + i;
                proposals[i]->token = proposed_tokens[i];
                proposals[i]->position = pos;
                const auto decision = strix::decide_token(p[prior], proposals[i], 0.0,
                    strix::speculation_uniform01(rng, pos, Draw::draft_acceptance),
                    strix::speculation_uniform01(rng, pos, Draw::residual_fallback));
                tokens[emitted++] = decision.token;
                prior = decision.token;
                if (!decision.accepted) {
                    ++rejected_windows;
                    rejected = true;
                    break;
                }
            }
            if (!rejected) {
                const auto bonus = strix::decide_token(p[prior], nullptr,
                    strix::speculation_uniform01(rng, start + depth, Draw::serial_target), 0.0, 0.0);
                tokens[emitted++] = bonus.token;
                prior = bonus.token;
                if (depth == 2) {
                    ++full_window_bonus;
                }
            }
        }
        ++counts[(tokens[0] * vocab + tokens[1]) * vocab + tokens[2]];
    }
    double statistic = 0.0;
    for (std::size_t a = 0; a < vocab; ++a) {
        for (std::size_t b = 0; b < vocab; ++b) {
            for (std::size_t c = 0; c < vocab; ++c) {
                const double expected = trials * target[vocab][a] * target[a][b] * target[b][c];
                const double difference = counts[(a * vocab + b) * vocab + c] - expected;
                statistic += difference * difference / expected;
            }
        }
    }
    std::cout << "Three-position chain chi-square=" << statistic << " df=26 cutoff=70"
              << ", full-window bonuses=" << full_window_bonus << ", rejected windows=" << rejected_windows << '\n';
    expect(statistic < 70.0 && full_window_bonus > 0 && rejected_windows > 0,
           "stochastic chain with a bonus matches the target three-position joint");

    const strix::TokenSamplingPolicy zero_temperature;
    const std::array<std::uint32_t, 3> target_greedy{2, 1, 0};
    const std::array<double, 1> mass{1.0};
    for (const auto candidates : {std::array<std::uint32_t, 2>{2, 0}, std::array<std::uint32_t, 2>{2, 1}}) {
        const auto greedy = strix::verify_greedy_candidates({candidates, target_greedy, 3, {}});
        std::vector<std::uint32_t> point_mass_output;
        for (std::size_t i = 0; i < target_greedy.size(); ++i) {
            const auto point_target = strix::make_token_distribution_from_probabilities(
                std::span<const std::uint32_t>(target_greedy).subspan(i, 1), mass);
            if (i == candidates.size()) {
                point_mass_output.push_back(strix::decide_token(point_target, nullptr, 0.7, 0.0, 0.0).token);
                break;
            }
            const strix::TokenDraft point_draft{51U + i, candidates[i],
                strix::make_token_distribution_from_probabilities(
                    std::span<const std::uint32_t>(candidates).subspan(i, 1), mass), {}};
            const auto decision = strix::decide_token(point_target, &point_draft, 0.0, 0.4, 0.2);
            point_mass_output.push_back(decision.token);
            if (!decision.accepted) {
                break;
            }
        }
        expect(!strix::token_sampling_enabled(zero_temperature) && point_mass_output == greedy.valid_tokens,
               "temperature zero bypasses the stochastic sampler and its point-mass limit matches greedy");
    }
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
    check_chain_joint();
    print_coupling_table();
    return failures == 0 ? 0 : 1;
}
