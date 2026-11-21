#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "strix/spec/engine_target.hpp"
#include "strix/spec/mtp_chain.hpp"

namespace sp = strix::spec;
using strix::TokenSamplingPolicy;

namespace {

void check(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(EXIT_FAILURE);
}

constexpr std::uint32_t vocab = 8;
constexpr std::uint64_t request = 37;
using History = std::array<std::uint32_t, 3>;
using Logits = std::array<float, vocab>;

void append(History& history, std::uint32_t token) { history = {history[1], history[2], token}; }

History history_of(std::span<const std::uint32_t> tokens) {
    History history{};
    for (const auto token : tokens) append(history, token);
    return history;
}

Logits lm(const History& history, std::uint64_t position) {
    auto hash = position * 2654435761U;
    for (const auto token : history) hash = (hash ^ token) * 1099511628211ULL;
    Logits logits{};
    for (auto& logit : logits) {
        hash ^= hash >> 27;
        hash *= 0x2545f4914f6cdd1dULL;
        logit = static_cast<float>(static_cast<int>(hash >> 59) - 15) / 24.0F;
    }
    if (history[0] == history[1] && history[1] == history[2]) logits[history[2]] += 1.5F;
    return logits;
}

strix::TokenSamplerOptions sampler_options() {
    return {strix::SampledAcceptance::coupled, 0.0F, request, strix::TokenSamplerOptions::forced_rejection_off};
}

std::uint32_t argmax(const Logits& logits) {
    return static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

std::vector<std::uint32_t> reference(const TokenSamplingPolicy& policy, std::span<const std::uint32_t> prompt,
                                     std::uint32_t count) {
    std::vector<std::uint32_t> output;
    auto history = history_of(prompt);
    std::optional<strix::TokenSampler> sampler;
    if (policy.temperature > 0.0F) sampler.emplace(policy, sampler_options());
    for (std::uint64_t i = prompt.size(); output.size() < count; ++i) {
        const auto logits = lm(history, i);
        const auto token = sampler ? sampler->decide(logits, i, nullptr).token : argmax(logits);
        output.push_back(token);
        append(history, token);
    }
    return output;
}

struct Costs {
    double row = 0.004;
    double draft = 0.01;
    double verify(std::uint32_t rows) const { return 1.0 + row * static_cast<double>(rows - 1); }
};

struct FakeEngine {
    struct Rows {};
    enum class LogitRows { all };

    FakeEngine(std::span<const std::uint32_t> prefix, double& clock, Costs costs)
        : state(history_of(prefix)), position(prefix.size()), clock_ms(clock), price(costs) {}

    Rows forward_rows(const std::uint32_t* ids, std::uint32_t rows, LogitRows) {
        auto history = state;
        for (std::uint32_t r = 0; r < rows; ++r) {
            append(history, ids[r]);
            histories[r] = history;
            logits[r] = lm(history, position + r + 1);
        }
        clock_ms += price.verify(rows);
        return {};
    }

    std::uint32_t vocab() const { return static_cast<std::uint32_t>(logits[0].size()); }
    void read_logits_rows(Rows, std::uint32_t rows, float* host) {
        ++logit_reads;
        for (std::uint32_t r = 0; r < rows; ++r) host = std::copy(logits[r].begin(), logits[r].end(), host);
    }
    void read_top(Rows, std::uint32_t rows, std::uint32_t* host) {
        ++top_reads;
        for (std::uint32_t r = 0; r < rows; ++r) host[r] = argmax(logits[r]);
    }
    void commit(std::uint32_t k) {
        state = histories[k - 1];
        position += k;
    }
    void sync() {}

    History state;
    std::uint64_t position;
    double& clock_ms;
    Costs price;
    std::array<History, 16> histories{};
    std::array<Logits, 16> logits{};
    std::uint64_t logit_reads = 0, top_reads = 0;
};

struct TreeEngine : FakeEngine {
    using FakeEngine::FakeEngine;
    std::uint64_t committed() const { return position; }
    std::uint32_t tree_rows() const { return static_cast<std::uint32_t>(logits.size()); }
    Rows forward_tree_rows(const sp::TreeRows& tree, LogitRows) {
        for (std::size_t r = 0; r < tree.tokens.size(); ++r) {
            auto history = r == 0 ? state : histories[static_cast<std::size_t>(tree.parent[r])];
            append(history, tree.tokens[r]);
            histories[r] = history;
            logits[r] = lm(history, position + tree.depth[r] + 1);
        }
        clock_ms += price.verify(static_cast<std::uint32_t>(tree.tokens.size()));
        return {};
    }
    void commit_path(std::span<const std::uint32_t> path) {
        state = histories[path.back()];
        position += path.size();
    }
};

enum class Draft { copy, coupled, argmax, tree, garbage, exact };

class FakeChain final : public sp::ChainDrafter {
public:
    FakeChain(std::span<const std::uint32_t> prompt, Draft mode, double& clock, Costs costs)
        : history_(history_of(prompt)), position_(prompt.size()), mode_(mode), clock_ms_(clock), price_(costs) {}

    void set_row_rule(const sp::RowRule& rule) override { rule_ = rule; }
    void set_first_position(std::uint64_t position) override {
        check(position == position_, "drafter first position");
    }
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        auto history = history_;
        for (const auto token : prefix) append(history, token);
        out.clear();
        alternatives_.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const auto position = position_ + prefix.size() + j;
            auto logits = lm(history, position);
            if (mode_ != Draft::exact) logits[0] += 0.2F;
            auto token = rule_.draft_token(logits, {}, position);
            if (mode_ == Draft::argmax) token = argmax(logits);
            if (mode_ == Draft::garbage) token = static_cast<std::uint32_t>(position % vocab);
            if (mode_ == Draft::tree) {
                auto other = (token + 1) % vocab;
                if (position % 7 == 0) std::swap(other, token);
                alternatives_.push_back(other);
            }
            out.push_back(token);
            append(history, token);
        }
        clock_ms_ += price_.draft * k;
    }
    void accept(std::span<const std::uint32_t> emitted) override {
        for (const auto token : emitted) append(history_, token);
        position_ += emitted.size();
    }
    std::span<const std::uint32_t> runner_ups() const override { return alternatives_; }

private:
    History history_;
    std::uint64_t position_;
    Draft mode_;
    double& clock_ms_;
    Costs price_;
    sp::RowRule rule_;
    std::vector<std::uint32_t> alternatives_;
};

struct Run {
    std::vector<std::uint32_t> output;
    std::uint64_t proposed = 0, accepted = 0, tree_windows = 0, sibling_taken = 0;
    std::uint32_t late_short = 0, late_long = 0;
};

sp::BlockConfig config(Draft mode, const double& clock, Costs price) {
    sp::BlockConfig cfg;
    cfg.proposer.index.capacity = 2048U;
    cfg.copy = mode == Draft::copy;
    cfg.hybrid = false;
    cfg.chain_seed_flush_ms = cfg.chain_seed_step_ms = price.draft;
    cfg.arbiter.min_lookup_evidence = 0.0;
    cfg.arbiter.probe_every = 32;
    cfg.clock_ms = [&clock] { return clock; };
    return cfg;
}

void seed_costs(sp::BlockDrafter& drafter, Costs price) {
    for (std::uint32_t r = 1; r <= 16; ++r) drafter.seed_verify_ms(r, price.verify(r));
}

template<class Engine>
Run run(const TokenSamplingPolicy& policy, std::span<const std::uint32_t> prompt, std::uint32_t count,
        Draft mode, Costs price = {}) {
    Run result;
    double clock = 0.0;
    Engine engine(prompt.first(prompt.size() - 1), clock, price);
    sp::EngineVerifyTarget<Engine> target(engine);
    FakeChain chain(prompt, mode, clock, price);
    sp::BlockDrafter drafter(config(mode, clock, price), mode == Draft::copy ? nullptr : &chain);
    sp::BlockSpeculator spec(drafter, target, sp::RowRule(policy, request));
    seed_costs(drafter, price);
    drafter.commit(prompt);
    spec.start(prompt.back());
    while (result.output.size() < count) {
        const auto before = static_cast<std::uint32_t>(result.output.size());
        const auto step = spec.step(result.output, {count - before, 16, prompt.size() + before - 1});
        check(step.emitted > 0 && step.emitted <= count - before, "step token budget");
        const std::uint32_t sibling = step.sibling_taken ? 1U : 0U;
        check(step.emitted == step.accepted + sibling + 1, "accepted prefix plus correction or bonus");
        check(engine.position == prompt.size() + result.output.size() - 1, "committed position excludes anchor");
        result.proposed += step.proposed;
        result.accepted += step.accepted;
        result.sibling_taken += sibling;
        if (before >= count / 2 && count - before >= 16) {
            if (step.rows <= 2) ++result.late_short;
            if (step.rows >= 5) ++result.late_long;
        }
    }
    const auto& stats = drafter.stats();
    check(stats.emitted == count && stats.sibling_taken == result.sibling_taken, "emission and sibling stats");
    check(std::accumulate(stats.source_proposed.begin(), stats.source_proposed.end(), std::uint64_t{0}) == result.proposed,
          "proposed stats");
    check(std::accumulate(stats.source_accepted.begin(), stats.source_accepted.end(), std::uint64_t{0}) == result.accepted &&
          std::accumulate(stats.depth_accepted.begin(), stats.depth_accepted.end(), std::uint64_t{0}) == result.accepted,
          "accepted stats");
    check(policy.temperature > 0.0F ? engine.top_reads == 0 && engine.logit_reads == stats.passes
                                    : engine.logit_reads == 0 && engine.top_reads == stats.passes, "selected row read");
    result.tree_windows = stats.tree_windows;
    return result;
}

void identity_and_accounting() {
    std::array<std::uint32_t, 32> prompt{};
    prompt.fill(2);
    std::uint64_t copy_rows = 0, chain_rows = 0, garbage_rows = 0, trees = 0, siblings = 0;
    constexpr std::array<TokenSamplingPolicy, 4> filters{{
        {}, {.top_k = 5}, {.top_p = 0.9F}, {.min_p = 0.05F}
    }};
    for (const float temperature : {0.7F, 1.0F}) {
        for (auto policy : filters) {
            policy.temperature = temperature;
            for (std::uint64_t seed = 0; seed < 300; ++seed) {
                policy.seed = seed;
                const auto expected = reference(policy, prompt, 64);
                const auto copy = run<FakeEngine>(policy, prompt, 64, Draft::copy);
                const auto chain = run<FakeEngine>(policy, prompt, 64, Draft::coupled);
                const auto garbage = run<FakeEngine>(policy, prompt, 64, Draft::garbage);
                const auto tree = run<TreeEngine>(policy, prompt, 64, Draft::tree);
                check(copy.output == expected && chain.output == expected && tree.output == expected &&
                      garbage.output == expected, "T1 sampled ids match serial");
                copy_rows += copy.proposed;
                chain_rows += chain.proposed;
                garbage_rows += garbage.proposed;
                trees += tree.tree_windows;
                siblings += tree.sibling_taken;
            }
        }
    }
    check(copy_rows > 0 && chain_rows > 0 && garbage_rows > 0 && trees > 0 && siblings > 0, "T1 all verify paths ran");
    const TokenSamplingPolicy greedy{};
    check(run<TreeEngine>(greedy, prompt, 64, Draft::tree).output == reference(greedy, prompt, 64), "T2 greedy ids");
    check(sp::RowRule{}.row_token(Logits{}, 0) == 0, "T2 lowest greedy id");
    const std::array<float, 2> tied{};
    const std::array<std::uint32_t, 2> ids{5, 3};
    check(sp::RowRule{}.draft_token(tied, ids, 0) == 3, "T2 lowest greedy subset id");

    std::uint64_t coupled_proposed = 0, coupled_accepted = 0, argmax_proposed = 0, argmax_accepted = 0;
    for (std::uint64_t seed = 0; seed < 100; ++seed) {
        const TokenSamplingPolicy policy{.temperature = 1.0F, .seed = seed};
        const auto coupled = run<FakeEngine>(policy, prompt, 64, Draft::coupled);
        const auto maximum = run<FakeEngine>(policy, prompt, 64, Draft::argmax);
        coupled_proposed += coupled.proposed;
        coupled_accepted += coupled.accepted;
        argmax_proposed += maximum.proposed;
        argmax_accepted += maximum.accepted;
    }
    check(coupled_proposed > 0 && argmax_proposed > 0, "T4 both drafters proposed");
    const double coupled_rate = static_cast<double>(coupled_accepted) / static_cast<double>(coupled_proposed);
    const double argmax_rate = static_cast<double>(argmax_accepted) / static_cast<double>(argmax_proposed);
    std::printf("T4 acceptance: coupled %.4f, argmax %.4f\n", coupled_rate, argmax_rate);
    check(coupled_rate > argmax_rate, "T4 coupled draft acceptance improves");
}

template<std::size_t N>
double chi_square_p(const std::array<std::uint32_t, N>& actual, const std::array<double, N>& probability) {
    static_assert(N % 2 == 0);
    const auto n = std::accumulate(actual.begin(), actual.end(), std::uint32_t{0});
    double chi_square = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double expected = n * probability[i];
        check(expected >= 5.0, "T3 chi-square bin count");
        const double delta = actual[i] - expected;
        chi_square += delta * delta / expected;
    }
    // Upper tail for odd degrees of freedom, using the half-integer incomplete gamma recurrence.
    const double x = chi_square / 2.0;
    double p = std::erfc(std::sqrt(x));
    double term = std::exp(-x) * std::sqrt(x) / std::tgamma(1.5);
    for (std::size_t j = 1; j < N / 2; ++j) {
        p += term;
        term *= x / (static_cast<double>(j) + 0.5);
    }
    return p;
}

void distribution() {
    constexpr History prompt{1, 2, 3};
    TokenSamplingPolicy policy{.temperature = 1.0F};
    std::array<std::uint32_t, vocab> marginal_count{};
    std::array<std::uint32_t, vocab * vocab> pair_count{};
    for (std::uint64_t seed = 0; seed < 4000; ++seed) {
        policy.seed = seed;
        const auto result = run<TreeEngine>(policy, prompt, 2, Draft::tree);
        ++marginal_count[result.output[0]];
        ++pair_count[result.output[0] * vocab + result.output[1]];
    }
    std::array<double, vocab> marginal{};
    std::array<double, vocab * vocab> pair{};
    const auto first = strix::make_token_distribution(lm(prompt, 3), {}, policy);
    for (std::uint32_t a = 0; a < vocab; ++a) {
        auto history = prompt;
        append(history, a);
        marginal[a] = first.probability(a);
        const auto second = strix::make_token_distribution(lm(history, 4), {}, policy);
        for (std::uint32_t b = 0; b < vocab; ++b) pair[a * vocab + b] = marginal[a] * second.probability(b);
    }
    const double marginal_p = chi_square_p(marginal_count, marginal), pair_p = chi_square_p(pair_count, pair);
    std::printf("T3 chi-square p-values: position 3 %.6f, pair (3, 4) %.6f\n", marginal_p, pair_p);
    check(marginal_p > 1e-4 && pair_p > 1e-4, "T3 serial marginal and joint distribution");
}

void arbiter() {
    constexpr History prompt{1, 2, 3};
    const TokenSamplingPolicy policy{.temperature = 1.0F, .seed = 25};
    const Costs price{0.34, 0.10};
    const auto bad = run<FakeEngine>(policy, prompt, 1024, Draft::garbage, price);
    const auto good = run<FakeEngine>(policy, prompt, 1024, Draft::exact, price);
    std::printf("T5 late windows (short / long): garbage %u / %u, coupled %u / %u\n",
                bad.late_short, bad.late_long, good.late_short, good.late_long);
    check(bad.proposed > 0 && bad.late_short > 5 * bad.late_long && good.late_long > 5 * good.late_short,
          "T5 pricing selects serial/short for garbage and long for a good draft");

    double clock = 0.0;
    sp::BlockDrafter drafter(config(Draft::coupled, clock, price));
    const auto prior = drafter.arbiter().survival(sp::Source::model, 0, 1);
    drafter.seed_chain_survival(std::array{0.98}, 100.0);
    check(drafter.arbiter().survival(sp::Source::model, 0, 1) > prior, "T5 learned greedy history");
    seed_costs(drafter, price);
    const auto seeded_cost = drafter.arbiter().verify_ms(4, 0);
    const sp::RowRule sampled(policy, request);
    drafter.set_row_rule(sampled);
    check(drafter.arbiter().survival(sp::Source::model, 0, 1) == prior &&
          drafter.arbiter().verify_ms(4, 0) == seeded_cost, "T5 sampled transition clears evidence and keeps cost seeds");
    drafter.seed_chain_survival(std::array{0.02}, 100.0);
    const auto sampled_history = drafter.arbiter().survival(sp::Source::model, 0, 1);
    drafter.set_row_rule(sp::RowRule(policy, request + 1));
    check(sampled_history < prior && drafter.arbiter().survival(sp::Source::model, 0, 1) == sampled_history,
          "T5 same regime retains history across requests");
    drafter.set_row_rule({});
    check(drafter.arbiter().survival(sp::Source::model, 0, 1) == prior, "T5 greedy transition clears sampled evidence");
}

class SerialTarget : public sp::VerifyTarget {
public:
    explicit SerialTarget(FakeEngine& engine) : engine_(engine) {}
    void verify(std::span<const std::uint32_t>, std::vector<std::uint32_t>&) override {
        check(false, "sampled serial fallback must use logits");
    }
    std::span<const float> verify_serial_logits(std::uint32_t anchor) override {
        engine_.forward_rows(&anchor, 1, FakeEngine::LogitRows::all);
        return engine_.logits[0];
    }
    void commit(std::uint32_t rows) override {
        check(rows == 1, "sampled fallback commits one row");
        engine_.commit(rows);
    }

private:
    FakeEngine& engine_;
};

struct IdOnlyEngine : FakeEngine {
    using FakeEngine::FakeEngine;
    void read_logits_rows(Rows, std::uint32_t, float*) = delete;
};

void serial_fallback() {
    constexpr History prompt{1, 2, 3};
    const TokenSamplingPolicy policy{.temperature = 1.0F, .seed = 153};
    double clock = 0.0;
    FakeEngine engine(std::span(prompt).first<2>(), clock, {});
    SerialTarget target(engine);
    FakeChain chain(prompt, Draft::coupled, clock, {});
    sp::BlockDrafter drafter(config(Draft::coupled, clock, {}), &chain);
    sp::BlockSpeculator spec(drafter, target, sp::RowRule(policy, request));
    check(!target.supports_row_rule(true), "serial target declines sampled windows");
    spec.set_prelaunch(true);
    spec.start(prompt.back());
    std::vector<std::uint32_t> output;
    for (std::uint32_t i = 0; i < 16; ++i) {
        const auto step = spec.step(output, {16 - i, 16, engine.position});
        check(step.source == sp::Source::serial && step.rows == 1 && step.proposed == 0, "serial-only sampling result");
    }
    check(output == reference(policy, prompt, 16), "serial fallback sampled ids");
    check(drafter.stats().source_passes[0] == 16 && spec.prelaunched_steps() == 0, "serial-only stats");

    IdOnlyEngine id_engine(std::span(prompt).first<2>(), clock, {});
    sp::EngineVerifyTarget<IdOnlyEngine> id_target(id_engine);
    sp::BlockDrafter id_drafter(config(Draft::copy, clock, {}));
    sp::BlockSpeculator id_spec(id_drafter, id_target, sp::RowRule(policy, request));
    id_spec.start(prompt.back());
    check(!id_target.supports_row_rule(true), "argmax engine declines sampled windows");
    check(id_spec.step(output, {0, 16, id_engine.position}).emitted == 0, "empty budget needs no logits");
    bool threw = false;
    try {
        id_spec.step(output, {1, 16, id_engine.position});
    } catch (const std::logic_error&) {
        threw = true;
    }
    check(threw && output.size() == 16 && id_engine.position == 2 && id_engine.top_reads == 0,
          "sampling without any logits fails before emitting");
}

struct HostMtpEngine {
    struct Rows {
        const std::uint32_t* top = nullptr;
        const float* hidden_pre = nullptr;
        const float* hidden_post = nullptr;
    };
    using MtpOut = Rows;
    struct Config { std::uint32_t hidden = 1; };
    Config config() const { return {}; }
    static constexpr std::array<std::uint32_t, 3> ids{5, 1, 3};
    static std::array<float, 3> logits(std::uint32_t input, std::uint64_t position) {
        return {static_cast<float>(input) / 9.0F, static_cast<float>(position % 4) / 5.0F, 0.25F};
    }
    MtpOut mtp_forward(const std::uint32_t* input, const float*, std::uint32_t rows, std::uint32_t pos, bool want_top) {
        if (want_top) {
            for (std::uint32_t r = 0; r < rows; ++r) {
                output[r] = logits(input[r], std::uint64_t(pos) + r + 2);
                const auto top = std::max_element(output[r].begin(), output[r].end());
                top_ids[r] = ids[static_cast<std::size_t>(top - output[r].begin())];
            }
        }
        return {top_ids.data(), hidden.data(), hidden.data()};
    }
    void read_top(const Rows& rows, std::uint32_t n, std::uint32_t* out) {
        ++top_reads;
        std::copy_n(rows.top, n, out);
    }
    void read_mtp_logits(const MtpOut&, std::uint32_t row, std::vector<float>& values, std::vector<std::uint32_t>& tokens) {
        ++logit_reads;
        values.assign(output[row].begin(), output[row].end());
        tokens.assign(ids.begin(), ids.end());
    }
    std::uint32_t top_reads = 0, logit_reads = 0;
    std::array<std::uint32_t, 3> top_ids{};
    std::array<float, 3> hidden{};
    std::array<std::array<float, 3>, 3> output{};
};

struct HostMemory {
    static float* alloc(HostMtpEngine&, std::size_t n) { return new float[n]{}; }
    static void release(HostMtpEngine&, float* p) { delete[] p; }
    static void copy(HostMtpEngine&, float* dst, const float* src, std::size_t n) { std::copy_n(src, n, dst); }
};

void mtp_host_logits() {
    constexpr History prompt{1, 2, 3};
    constexpr std::uint32_t anchor = 4;
    const TokenSamplingPolicy policy{.temperature = 1.0F, .seed = 11};
    const sp::RowRule rule(policy, request);
    check(rule.identity() == strix::speculation_rng_identity(strix::token_sampling_rng(policy, request)),
          "sampled row identity");
    HostMtpEngine engine;
    sp::MtpChain<HostMtpEngine, HostMemory> chain(engine, false, 3);
    chain.set_row_rule(rule);
    strix::TokenSampler sampler(policy, sampler_options());
    const auto predict = [&](std::uint32_t input, std::uint64_t position) {
        return sampler.draft(HostMtpEngine::logits(input, position), HostMtpEngine::ids, position).token;
    };
    const std::array<float, 3> hidden{};
    const HostMtpEngine::Rows prefill{nullptr, hidden.data(), hidden.data()};
    std::vector<std::uint32_t> output;
    chain.observe_prefill(prefill, 3, prompt.data(), anchor);
    chain.draft({}, 3, output);
    const auto first = predict(anchor, 4);
    const auto second = predict(first, 5);
    check(output == (std::vector<std::uint32_t>{first, second, predict(second, 6)}), "MTP flush and step positions");

    constexpr std::array<std::uint32_t, 2> prefix{2, 6};
    chain.reset();
    chain.observe_prefill(prefill, 3, prompt.data(), anchor);
    chain.draft(prefix, 2, output);
    const auto after_prefix = predict(prefix.back(), 6);
    check(output == (std::vector<std::uint32_t>{after_prefix, predict(after_prefix, 7)}), "MTP copy prefix positions");
    check(engine.logit_reads == 6 && engine.top_reads == 0, "MTP uses host logits under sampling");
}

}  // namespace

int main() {
    identity_and_accounting();
    distribution();
    arbiter();
    serial_fallback();
    mtp_host_logits();
}
