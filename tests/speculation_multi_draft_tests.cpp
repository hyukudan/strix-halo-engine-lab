#include "strix/speculation_multi_draft.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

using Draw = strix::SpeculationRandomDrawClass;
constexpr std::uint32_t samples = 200'000;
constexpr std::uint64_t position = 103;
int failures = 0;

void expect(bool condition, const char* finding) {
    if (!condition) {
        std::cerr << "FAIL: " << finding << '\n';
        ++failures;
    }
}

strix::TokenDistribution distribution(std::vector<double> weights) {
    std::vector<std::uint32_t> ids(weights.size());
    std::iota(ids.begin(), ids.end(), std::uint32_t{0});
    const double mass = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto& weight : weights) {
        weight /= mass;
    }
    return strix::make_token_distribution_from_probabilities(ids, weights);
}

void propose(std::span<strix::SpeculationTreeCandidate> candidates,
    const strix::SpeculationRngIdentity& rng, std::uint64_t pos, std::uint32_t first) {
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        auto& draft = candidates[i].draft;
        draft.position = pos;
        draft.token = draft.distribution.sample(strix::speculation_uniform01(
            rng, pos, Draw::draft_proposal, first + static_cast<std::uint32_t>(i)));
    }
}

void chi_square(const char* name, std::span<const std::uint32_t> counts, std::span<const double> p) {
    double statistic = 0.0;
    unsigned bins = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (p[i] == 0.0) {
            expect(counts[i] == 0, "no samples outside target support");
            continue;
        }
        ++bins;
        const double expected = samples * p[i];
        const double difference = counts[i] - expected;
        statistic += difference * difference / expected;
    }
    const unsigned df = bins - 1U;
    const double limit = df + 6.0 * std::sqrt(2.0 * df) + 12.0;
    expect(statistic < limit, name);
    std::cout << name << " chi-square=" << statistic << " df=" << df << " cutoff=" << limit << '\n';
}

enum class Overlap { mixed, partial, identical, disjoint };

void check_row(const char* name, std::size_t vocab, Overlap overlap) {
    std::vector<double> p_weights(vocab);
    std::array<std::vector<double>, 3> q_weights{
        std::vector<double>(vocab), std::vector<double>(vocab), std::vector<double>(vocab)};
    for (std::size_t i = 0; i < vocab; ++i) {
        p_weights[i] = overlap != Overlap::disjoint || i < vocab / 2 ? static_cast<double>(1U + i % 5U) : 0.0;
        for (std::size_t j = 0; j < q_weights.size(); ++j) {
            if (overlap == Overlap::identical) {
                q_weights[j][i] = p_weights[i];
            } else if (overlap == Overlap::mixed ||
                       (overlap == Overlap::partial && i % 3U != j) ||
                       (overlap == Overlap::disjoint && i == vocab - 1U)) {
                q_weights[j][i] = static_cast<double>(1U + (i + 3U * j) % 7U);
            }
        }
    }
    const auto target = distribution(p_weights);
    std::array<strix::SpeculationTreeCandidate, 3> candidates{};
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        candidates[i].draft.distribution = distribution(q_weights[i]);
    }
    const strix::SpeculationTreeNode row{target, candidates, 11};
    std::vector<std::uint32_t> counts(vocab);
    std::array<std::uint32_t, 3> accepted{};
    std::uint32_t fallback = 0;
    std::uint32_t duplicate = 0;
    for (std::uint32_t trial = 0; trial < samples; ++trial) {
        const strix::SpeculationRngIdentity rng{3079, trial + 1U, vocab};
        propose(candidates, rng, position, row.first_draw_index);
        const auto result = strix::verify_multi_draft_row(row, rng, position);
        ++counts.at(result.token);
        if (result.accepted_candidate) {
            ++accepted[*result.accepted_candidate];
        } else {
            ++fallback;
        }
        if (candidates[0].draft.token == candidates[1].draft.token) {
            ++duplicate;
        }
    }
    std::vector<double> expected(vocab);
    for (std::size_t i = 0; i < vocab; ++i) {
        expected[i] = target.probability(static_cast<std::uint32_t>(i));
    }
    chi_square(name, counts, expected);
    if (overlap == Overlap::identical) {
        expect(accepted[0] == samples, "identical rows always accept first sibling");
    } else if (overlap == Overlap::disjoint) {
        expect(fallback == samples && duplicate == samples, "duplicates on disjoint support always fall back");
    } else {
        expect(accepted[1] > 0 && accepted[2] > 0 && fallback > 0 && duplicate > 0,
               "mixed rows exercise later accepts, final fallback and repeated proposals");
    }
}

void check_tree() {
    constexpr std::size_t vocab = 6;
    const auto root_target = distribution({1, 2, 3, 4, 5, 6});
    const std::array child_target{
        distribution({6, 1, 2, 3, 4, 5}), distribution({5, 6, 1, 2, 3, 4}),
        distribution({4, 5, 6, 1, 2, 3}), distribution({3, 4, 5, 6, 1, 2}),
        distribution({2, 3, 4, 5, 6, 1}), distribution({1, 2, 3, 4, 5, 6}),
    };
    const auto q_a = distribution({2, 5, 0, 2, 1, 1});
    const auto q_b = distribution({4, 2, 0, 1, 1, 2});
    std::array<strix::SpeculationTreeCandidate, 2> root_candidates{{{{0, 0, q_a, {}}, nullptr},
                                                                  {{0, 0, q_b, {}}, nullptr}}};
    std::array<std::array<strix::SpeculationTreeCandidate, 2>, 2> children{{root_candidates, root_candidates}};
    const strix::SpeculationTreeNode root{root_target, root_candidates, 7};
    std::array<std::uint32_t, vocab * vocab> counts{};
    std::array<double, vocab * vocab> expected{};
    for (std::uint32_t a = 0; a < vocab; ++a) {
        for (std::uint32_t b = 0; b < vocab; ++b) {
            expected[a * vocab + b] = root_target.probability(a) * child_target[a].probability(b);
        }
    }
    std::uint32_t both = 0;
    for (std::uint32_t trial = 0; trial < samples; ++trial) {
        const strix::SpeculationRngIdentity rng{6029, trial + 1U, 3};
        propose(root_candidates, rng, position, root.first_draw_index);
        const strix::SpeculationTreeNode left{child_target[root_candidates[0].draft.token], children[0], 7};
        const strix::SpeculationTreeNode right{child_target[root_candidates[1].draft.token], children[1], 9};
        propose(children[0], rng, position + 1U, left.first_draw_index);
        propose(children[1], rng, position + 1U, right.first_draw_index);
        root_candidates[0].child = &left;
        root_candidates[1].child = &right;
        const auto tokens = strix::verify_speculative_tree(root, rng, position, 2);
        const auto a = tokens.at(0);
        // Root rejection ends speculative verification. Ordinary decoding
        // supplies the next position from the row at the emitted prefix.
        const auto b = tokens.size() == 2 ? tokens[1] :
            child_target[a].sample(strix::speculation_uniform01(rng, position + 1U, Draw::serial_target));
        ++counts.at(a * vocab + b);
        if (tokens.size() == 2) {
            ++both;
        }
    }
    expect(both > 0 && both < samples, "tree exercises both accepted paths and serial continuation");
    chi_square("depth-two tree", counts, expected);
}

void check_tree_boundaries() {
    const auto mass = distribution({1, 0, 0});
    const strix::SpeculationTreeNode bonus{mass, {}};
    const std::array candidates{strix::SpeculationTreeCandidate{{position, 0, mass, {}}, &bonus}};
    const strix::SpeculationTreeNode root{mass, candidates, UINT32_MAX};
    const strix::SpeculationRngIdentity rng{97, 1, 3};
    const std::array<std::uint32_t, 1> eos{0};
    expect(strix::verify_speculative_tree(root, rng, position, 0).empty() &&
           strix::verify_speculative_tree(root, rng, position, 1) == std::vector<std::uint32_t>{0} &&
           strix::verify_speculative_tree(root, rng, position, 2, eos) == std::vector<std::uint32_t>{0} &&
           strix::verify_speculative_tree(root, rng, position, 2) == std::vector<std::uint32_t>{0, 0},
           "tree obeys EOS and budget, and an admitted empty child samples its bonus");
    const std::array two{candidates[0], candidates[0]};
    bool index_overflow = false;
    try {
        static_cast<void>(strix::verify_multi_draft_row({mass, two, UINT32_MAX}, rng, position));
    } catch (const std::overflow_error&) {
        index_overflow = true;
    }
    expect(index_overflow, "tree rejects overlapping draw indices caused by integer overflow");

    const auto target = distribution({1, 1, 0});
    const auto disjoint = distribution({0, 0, 1});
    const strix::SpeculationTreeCandidate rejected{{position, 2, disjoint, {}}, nullptr};
    const std::array siblings{rejected, rejected, rejected};
    const strix::SpeculationTreeNode fallback_row{target, siblings, 11};
    bool match = true;
    bool distinguishes_previous = false;
    for (std::uint32_t trial = 1; trial <= 64; ++trial) {
        const strix::SpeculationRngIdentity identity{97, trial, 3};
        const auto token = target.sample(strix::speculation_uniform01(
            identity, position, Draw::residual_fallback, 13));
        const auto previous = target.sample(strix::speculation_uniform01(
            identity, position, Draw::residual_fallback, 12));
        const auto decision = strix::verify_multi_draft_row(fallback_row, identity, position);
        match = match && !decision.accepted_candidate && decision.token == token;
        distinguishes_previous = distinguishes_previous || token != previous;
    }
    expect(match && distinguishes_previous, "final fallback uses the last candidate index in its own draw class");
}

}  // namespace

int main() {
    check_row("mixed six-token row", 6, Overlap::mixed);
    check_row("partial eight-token row", 8, Overlap::partial);
    check_row("identical twelve-token row", 12, Overlap::identical);
    check_row("disjoint sixteen-token row with duplicates", 16, Overlap::disjoint);
    check_tree();
    check_tree_boundaries();
    return failures == 0 ? 0 : 1;
}
