#include "strix/token_sampling.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
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

bool close(double a, double b, double tolerance = 1e-12) { return std::fabs(a - b) <= tolerance; }

// Pearson chi-square of observed counts against expected probabilities; cells
// with expected count below 5 are pooled. Returns a Wilson-Hilferty z score.
double chi_square_z(const std::vector<std::uint64_t>& observed, const std::vector<double>& expected, std::uint64_t n) {
    double chi = 0.0, pooled_observed = 0.0, pooled_expected = 0.0;
    std::size_t cells = 0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        const double e = expected[i] * static_cast<double>(n);
        if (e < 5.0) {
            pooled_observed += static_cast<double>(observed[i]);
            pooled_expected += e;
            continue;
        }
        const double d = static_cast<double>(observed[i]) - e;
        chi += d * d / e;
        ++cells;
    }
    if (pooled_expected > 0.0) {
        if (pooled_expected >= 5.0) {
            const double d = pooled_observed - pooled_expected;
            chi += d * d / pooled_expected;
            ++cells;
        } else if (pooled_observed > 20.0) {
            return 100.0;
        }
    }
    const double df = static_cast<double>(cells > 1 ? cells - 1 : 1);
    const double t = 2.0 / (9.0 * df);
    return (std::cbrt(chi / df) - (1.0 - t)) / std::sqrt(t);
}

constexpr std::uint32_t vocabulary = 8;

// Toy autoregressive target and draft: logits depend on the previous token.
std::vector<float> target_logits(std::uint32_t previous) {
    std::vector<float> out(vocabulary);
    for (std::uint32_t j = 0; j < vocabulary; ++j)
        out[j] = static_cast<float>(2.0 * std::sin(1.7 * previous + 0.9 * j) + 0.3 * j);
    return out;
}
// The draft covers a vocabulary subset (no 4 or 7) in a non-sorted id order.
const std::array<std::uint32_t, 6> draft_ids{6, 0, 3, 1, 5, 2};
std::vector<float> draft_logits(std::uint32_t previous) {
    const auto target = target_logits(previous);
    std::vector<float> out;
    for (const auto id : draft_ids)
        out.push_back(static_cast<float>(target[id] + 0.8 * std::cos(2.3 * previous + 1.1 * id)));
    return out;
}

// One generation in the shape of the engine cycle: pending token, a draft
// chain that stops on a low max probability, one verification of `rows`
// target rows (fewer than drafts + 1 is allowed: the unverified drafts are
// dropped), rows decided in order until the first non-accepted row.
std::vector<std::uint32_t> generate(const strix::TokenSamplingPolicy& policy, std::uint32_t prompt_token,
    std::uint32_t length, std::uint32_t depth, std::uint64_t* accepted) {
    strix::TokenSampler sampler(policy);
    std::uint64_t position = 1; // the prompt token sits at position 0
    std::vector<std::uint32_t> out;
    // The prompt row: a plain sample.
    out.push_back(sampler.decide(target_logits(prompt_token), position, nullptr).token);
    while (out.size() < length) {
        const auto pending = out.back();
        std::vector<strix::TokenDraft> drafts;
        auto previous = pending;
        for (std::uint32_t k = 0; k < depth; ++k) {
            const auto logits = draft_logits(previous);
            drafts.push_back(sampler.draft(logits, draft_ids, position + 1 + k));
            float max = logits[0], sum = 0.0F;
            for (const auto l : logits) max = std::max(max, l);
            for (const auto l : logits) sum += std::exp(l - max);
            previous = drafts.back().token;
            if (1.0F / sum < 0.3F) break;
        }
        // Deterministic owner choice from the draft count: serial, 2 or 4 rows.
        const std::uint32_t want = static_cast<std::uint32_t>(drafts.size()) + 1U;
        const std::uint32_t rows = want == 1 ? 1 : want <= 2 ? 2 : want <= 3 ? 2 : 4;
        // Inputs: pending, then the drafts; row r predicts position + 1 + r.
        std::vector<std::uint32_t> inputs{pending};
        for (std::uint32_t r = 1; r < rows; ++r) inputs.push_back(drafts[std::min<std::size_t>(r - 1, drafts.size() - 1)].token);
        const auto budget = static_cast<std::uint32_t>(length - out.size());
        const auto decisions = sampler.decide_rows(std::min(rows, budget), position + 1, drafts, {},
            [&](std::uint32_t r) {
                static thread_local std::vector<float> row;
                row = target_logits(inputs[r]);
                return std::span<const float>(row);
            });
        for (const auto& d : decisions) {
            out.push_back(d.token);
            if (d.accepted && accepted) ++*accepted;
        }
        position += decisions.size();
    }
    return out;
}

void test_policy_and_distribution() {
    strix::TokenSamplingPolicy greedy;
    expect(!strix::token_sampling_enabled(greedy), "temperature 0 is greedy");
    strix::TokenSamplingPolicy a{0.7F, 0, 1.0F, 0.0F, 1}, b = a;
    b.top_k = 5;
    expect(strix::token_sampling_policy_identity(a) != strix::token_sampling_policy_identity(b), "policy identity covers top-k");
    b = a; b.seed = 99;
    expect(strix::token_sampling_policy_identity(a) == strix::token_sampling_policy_identity(b), "policy identity excludes the seed");
    bool threw = false;
    try { strix::validate_token_sampling_policy({1.0F, 0, 0.0F, 0.0F, 0}); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "top-p zero is rejected");

    const std::vector<float> logits{1.0F, 3.0F, 2.0F, 3.0F, -50.0F};
    {
        const auto p = strix::make_token_distribution(logits, {}, {1.0F, 0, 1.0F, 0.0F, 0});
        expect(p.tokens().size() == 4, "the e^-40 tail is dropped");
        const double z = std::exp(-2.0) + 2.0 + std::exp(-1.0);
        expect(close(p.probability(1), 1.0 / z) && close(p.probability(0), std::exp(-2.0) / z) &&
            p.probability(4) == 0.0, "temperature-1 softmax");
        expect(p.tokens()[0] == 0 && p.tokens()[3] == 3, "without top-p the order is token id ascending");
    }
    {
        const auto p = strix::make_token_distribution(logits, {}, {0.5F, 2, 1.0F, 0.0F, 0});
        expect(p.tokens().size() == 2 && close(p.probability(1), 0.5) && close(p.probability(3), 0.5), "top-k keeps ties by id");
    }
    {
        // weights 1, 1, e^-1, e^-2: 2/(2+e^-1+e^-2) = 0.79 < 0.8 so the third entry is kept.
        const auto p = strix::make_token_distribution(logits, {}, {1.0F, 0, 0.8F, 0.0F, 0});
        expect(p.tokens().size() == 3 && p.tokens()[0] == 1 && p.tokens()[1] == 3 && p.tokens()[2] == 2,
            "top-p keeps the shortest (weight desc, id asc) prefix");
        const double z = 2.0 + std::exp(-1.0);
        expect(close(p.probability(2), std::exp(-1.0) / z), "top-p renormalizes");
    }
    {
        const auto p = strix::make_token_distribution(logits, {}, {1.0F, 0, 1.0F, 0.3F, 0});
        expect(p.tokens().size() == 3 && p.probability(0) == 0.0, "min-p keeps p >= min_p * p_max");
    }
    {
        const std::vector<std::uint32_t> ids{40, 7, 12};
        const std::vector<float> l{0.0F, 1.0F, 0.5F};
        const auto q = strix::make_token_distribution(l, ids, {1.0F, 0, 1.0F, 0.0F, 0});
        expect(q.tokens()[0] == 7 && q.tokens()[2] == 40 && q.probability(5) == 0.0 && q.probability(7) > q.probability(12),
            "subset ids map to real tokens");
    }
    bool rejected = false;
    try { (void)strix::make_token_distribution(std::vector<float>{1.0F, NAN}, {}, {1.0F, 0, 1.0F, 0.0F, 0}); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "non-finite logits are rejected");
}

// One decision: output ~ p for p != q, q on a subset, rejected decisions never return the draft.
void test_single_decision() {
    const strix::TokenSamplingPolicy policy{1.0F, 0, 1.0F, 0.0F, 7};
    const auto rng = strix::token_sampling_rng(policy);
    const auto target = target_logits(3);
    const auto p = strix::make_token_distribution(target, {}, policy);
    constexpr std::uint64_t n = 400000;
    std::vector<std::uint64_t> counts(vocabulary);
    std::uint64_t accepted = 0;
    bool draft_on_reject = false;
    for (std::uint64_t i = 0; i < n; ++i) {
        strix::TokenDraft draft;
        draft.position = i;
        draft.distribution = strix::make_token_distribution(draft_logits(3), draft_ids, policy);
        draft.token = draft.distribution.sample(strix::speculation_uniform01(rng, i, strix::SpeculationRandomDrawClass::draft_proposal));
        const auto d = strix::decide_token(p, &draft,
            strix::speculation_uniform01(rng, i, strix::SpeculationRandomDrawClass::serial_target),
            strix::speculation_uniform01(rng, i, strix::SpeculationRandomDrawClass::draft_acceptance),
            strix::speculation_uniform01(rng, i, strix::SpeculationRandomDrawClass::residual_fallback));
        ++counts[d.token];
        accepted += d.accepted;
        draft_on_reject = draft_on_reject || (!d.accepted && d.token == draft.token);
    }
    std::vector<double> expected(vocabulary);
    for (std::uint32_t j = 0; j < vocabulary; ++j) expected[j] = p.probability(j);
    const auto z = chi_square_z(counts, expected, n);
    std::cout << "single decision: chi-square z " << z << ", acceptance " << static_cast<double>(accepted) / n << '\n';
    expect(z < 4.5, "speculative decision reproduces the target distribution");
    expect(!draft_on_reject, "a rejected decision never returns its draft");
    expect(counts[4] > 0 && counts[7] > 0, "the residual reaches tokens outside the draft vocabulary");
}

// Whole sequences through the engine-shaped cycle against exact target sequence probabilities.
void test_sequences(const strix::TokenSamplingPolicy& base, const char* name) {
    constexpr std::uint32_t length = 4, prompt = 2;
    constexpr std::uint64_t n = 200000;
    // Exact joint probability of the first `length` generated tokens.
    std::vector<double> expected(1U << (3 * length), 0.0);
    std::map<std::uint32_t, strix::TokenDistribution> p;
    for (std::uint32_t t = 0; t < vocabulary; ++t) p.emplace(t, strix::make_token_distribution(target_logits(t), {}, base));
    for (std::uint32_t cell = 0; cell < expected.size(); ++cell) {
        double probability = 1.0;
        std::uint32_t previous = prompt;
        for (std::uint32_t k = 0; k < length; ++k) {
            const auto token = (cell >> (3 * k)) & 7U;
            probability *= p.at(previous).probability(token);
            previous = token;
        }
        expected[cell] = probability;
    }
    for (const std::uint32_t depth : {0U, 3U}) {
        std::vector<std::uint64_t> counts(expected.size());
        std::uint64_t accepted = 0;
        for (std::uint64_t i = 0; i < n; ++i) {
            auto policy = base;
            policy.seed = i + 1000003ULL * depth;
            const auto tokens = generate(policy, prompt, length, depth, &accepted);
            std::uint32_t cell = 0;
            for (std::uint32_t k = 0; k < length; ++k) cell |= tokens[k] << (3 * k);
            ++counts[cell];
        }
        const auto z = chi_square_z(counts, expected, n);
        std::cout << name << " depth " << depth << ": chi-square z " << z << ", accepted drafts per sequence "
                  << static_cast<double>(accepted) / n << '\n';
        expect(z < 4.5, "speculative sequences reproduce the target sequence distribution");
        if (depth) expect(accepted > 0, "the sequence test accepts drafts");
    }
}

void test_reproducible() {
    const strix::TokenSamplingPolicy policy{0.8F, 6, 0.95F, 0.02F, 42};
    expect(generate(policy, 1, 24, 3, nullptr) == generate(policy, 1, 24, 3, nullptr), "same seed, same tokens");
    auto other = policy;
    other.seed = 43;
    expect(generate(policy, 1, 24, 3, nullptr) != generate(other, 1, 24, 3, nullptr), "a different seed changes tokens");
}

}  // namespace

int main() {
    test_policy_and_distribution();
    test_single_decision();
    test_sequences({1.0F, 0, 1.0F, 0.0F, 0}, "T=1");
    test_sequences({0.7F, 5, 1.0F, 0.0F, 0}, "T=0.7 top-k 5");
    test_sequences({1.0F, 0, 0.8F, 0.0F, 0}, "T=1 top-p 0.8");
    test_sequences({0.6F, 0, 1.0F, 0.1F, 0}, "T=0.6 min-p 0.1");
    test_reproducible();
    if (failures) {
        std::cerr << failures << " token sampling check(s) failed\n";
        return 1;
    }
    std::cout << "token sampling tests passed\n";
    return 0;
}
