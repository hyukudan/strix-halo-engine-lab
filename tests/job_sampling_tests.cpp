// The shared serve sampling translation (strix/serve/job_sampling.hpp): temperature 0 builds no sampler,
// deterministic drafts decided with coupled acceptance (tokens equal plain sampling's for any draft, the marginal
// of each row is the target distribution, acceptance of a draft d is p(d)), the receipt, and the batch row mask
// (strix/serve/batch_sampling.hpp).
#include "strix/serve/job_sampling.hpp"
#include "strix/serve/batch_sampling.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

void test_batch_rows() {
    using namespace strix::serve;
    struct BatchRow { std::uint32_t slot; std::uint32_t token; };
    const auto greedy = make_job_sampler(SamplingParameters{});
    SamplingParameters policy;
    policy.temperature = 0.7;
    const auto sampler = make_job_sampler(policy, strix::SampledAcceptance::coupled);
    const std::array<BatchRow, 6> rows{{{0U, 40U}, {0U, 41U}, {0U, 42U}, {1U, 50U}, {2U, 60U}, {2U, 61U}}};
    const std::vector<bool> emits{true, true, true, true, false, true};
    const std::array samplers{sampler.get(), greedy.get(), sampler.get()};
    const auto mask = sampling_row_mask(rows, emits, samplers);
    expect(mask == 0b100111U, "sample complete verify window and final prompt row");
    expect(sampled_position(884U, 0U) == 885U && sampled_position(884U, 2U) == 887U &&
               sampled_position(9U, 1U) == 11U, "positions include consumed prompt rows and use the next token offset");
    std::array<std::uint32_t, 6> best{1U, 2U, 3U, 4U, 5U, 6U};
    const std::array<std::uint32_t, 4> rank0{41U, 42U, 43U, 62U};
    const auto write = [&](std::size_t row, std::uint32_t token) { best[row] = token; };
    apply_sampled_tokens(mask, rank0, write);
    const std::array<std::uint32_t, 6> expected{41U, 42U, 43U, 4U, 5U, 62U};
    expect(best == expected, "rank 0 tokens replace precisely the sampled rows");
    const std::array<strix::TokenSampler*, 3> greedy_samplers{};
    const auto greedy_mask = sampling_row_mask(rows, emits, greedy_samplers);
    apply_sampled_tokens(greedy_mask, {}, write);
    expect(greedy_mask == 0U && best == expected, "temperature zero leaves all decisions untouched");
}

strix::serve::SamplingParameters params(double t, std::uint64_t seed, std::uint32_t top_k = 0, double top_p = 1.0,
                                        double min_p = 0.0) {
    strix::serve::SamplingParameters p;
    p.temperature = t; p.seed = seed; p.seeded = true; p.top_k = top_k; p.top_p = top_p; p.min_p = min_p;
    return p;
}

const std::vector<float> logits{2.0F, 1.0F, 0.5F, 0.0F, -0.5F, -1.0F, 1.5F, 0.25F};

std::vector<double> target(double t) {
    std::vector<double> p(logits.size());
    double m = -1e30, s = 0.0;
    for (const auto v : logits) m = std::fmax(m, static_cast<double>(v));
    for (std::size_t i = 0; i < p.size(); ++i) { p[i] = std::exp((static_cast<double>(logits[i]) - m) / t); s += p[i]; }
    for (auto& v : p) v /= s;
    return p;
}

void test_greedy_and_policy() {
    expect(strix::serve::make_job_sampler(params(0.0, 7)) == nullptr, "temperature 0 builds no sampler");
    const auto policy = strix::serve::job_sampling_policy(params(0.7, 9, 40, 0.9, 0.05));
    expect(policy.temperature == 0.7F && policy.top_k == 40U && policy.top_p == 0.9F && policy.min_p == 0.05F &&
               policy.seed == 9U, "policy conversion");
    bool threw = false;
    try { (void)strix::serve::make_job_sampler(params(0.7, 1, 0, 0.0)); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "top_p 0 refused");
    std::string error;
    expect(strix::serve::make_job_sampler(params(0.7, 1, 0, 0.0), error) == nullptr &&
               error.rfind("sampling policy refused: ", 0) == 0, "a refused policy fails the request alone");
    const auto s = strix::serve::make_job_sampler(params(1.0, 3), strix::SampledAcceptance::coupled);
    expect(s && s->coupled(), "deterministic drafts use coupled acceptance");
    strix::serve::GenerationStats untouched;
    strix::serve::record_job_sampling(untouched, nullptr);
    expect(untouched.sampling == "greedy" && untouched.engine_extra.empty(), "null sampler leaves the receipt");
}

// Coupled: a row with any draft decides the same token as the row without it, and accepts exactly when equal.
void test_draft_invariance_and_marginal() {
    const double t = 1.0;
    const auto p = target(t);
    const std::uint32_t draft_token = 1;  // a deterministic (one-hot) draft
    std::vector<std::uint64_t> counts(p.size(), 0);
    std::uint64_t accepted = 0, n = 0;
    for (std::uint64_t seed = 1; seed <= 400; ++seed) {
        auto plain = strix::serve::make_job_sampler(params(t, seed), strix::SampledAcceptance::coupled);
        auto drafted = strix::serve::make_job_sampler(params(t, seed), strix::SampledAcceptance::coupled);
        for (std::uint64_t pos = 100; pos < 150; ++pos) {
            const auto a = plain->decide(logits, pos, nullptr);
            strix::TokenDraft d; d.position = pos; d.token = draft_token;
            const auto b = drafted->decide(logits, pos, &d);
            expect(a.token == b.token, "draft does not change the token");
            expect(b.drafted && b.accepted == (b.token == draft_token), "accepted iff equal");
            ++counts[a.token]; accepted += b.accepted ? 1U : 0U; ++n;
        }
    }
    double chi = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double e = p[i] * static_cast<double>(n), dlt = static_cast<double>(counts[i]) - e;
        chi += dlt * dlt / e;
    }
    // 7 degrees of freedom: the 0.999 quantile is 24.3.
    std::cout << "marginal chi-square " << chi << " (df 7), acceptance " << static_cast<double>(accepted) / static_cast<double>(n)
              << " vs p(d) " << p[draft_token] << '\n';
    expect(chi < 24.3, "row marginal equals the target distribution");
    const double rate = static_cast<double>(accepted) / static_cast<double>(n);
    const double sd = std::sqrt(p[draft_token] * (1.0 - p[draft_token]) / static_cast<double>(n));
    expect(std::fabs(rate - p[draft_token]) < 4.0 * sd, "acceptance of a one-hot draft is p(d)");
}

void test_reproducible_and_receipt() {
    auto a = strix::serve::make_job_sampler(params(0.8, 42, 4), strix::SampledAcceptance::coupled);
    auto b = strix::serve::make_job_sampler(params(0.8, 42, 4), strix::SampledAcceptance::coupled);
    auto c = strix::serve::make_job_sampler(params(0.8, 43, 4), strix::SampledAcceptance::coupled);
    bool differs = false;
    for (std::uint64_t pos = 0; pos < 64; ++pos) {
        const auto x = a->decide(logits, pos, nullptr).token;
        expect(x == b->decide(logits, pos, nullptr).token, "same (seed, request) gives the same tokens");
        differs = differs || x != c->decide(logits, pos, nullptr).token;
    }
    expect(differs, "another seed gives other tokens");
    strix::serve::GenerationStats stats;
    strix::serve::record_job_sampling(stats, a.get());
    expect(stats.sampling == "sampling-v1", "receipt word");
    expect(stats.engine_extra.size() == 1U && stats.engine_extra[0].first == "sampling_receipt" &&
               stats.engine_extra[0].second.find("\"acceptance\":\"coupled\"") != std::string::npos &&
               stats.engine_extra[0].second.find("\"seed\":42") != std::string::npos, "receipt members");
}

}  // namespace

int main() {
    test_batch_rows();
    test_greedy_and_policy();
    test_draft_invariance_and_marginal();
    test_reproducible_and_receipt();
    if (failures != 0) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
    std::cout << "job sampling tests passed\n";
    return EXIT_SUCCESS;
}
