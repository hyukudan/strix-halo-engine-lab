// ArmSelector (strix/spec/arm_selector.hpp): per-regime choice between two drafters of one decode.
// Policy tests drive end_pass with synthetic per-regime rates (the measured Hy4 one-node shapes: bundled head vs block
// drafter on prose and code); the integration test runs BlockSpeculator over two chain drafters and checks the ids
// equal the target's greedy stream and that the idle arm's chain followed every pass.
#include "strix/spec/arm_selector.hpp"
#include "strix/spec/block_drafter.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sp = strix::spec;

namespace {
#define AS_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

double g_clock = 0.0;

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33U;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33U;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33U;
    return x;
}

// One arm's pass in a regime: tokens alternate around the mean (lo / hi), wall = fixed + per token.
struct Shape {
    double tokens = 1.0;  // mean emitted tokens per pass
    double ms = 50.0;     // mean pass wall
};

struct PolicyRun {
    double tokens = 0.0, ms = 0.0;
    std::uint64_t switches = 0, probes1 = 0, passes1 = 0;
    std::size_t final_active = 0;
};

// regime(pass) -> {arm0 shape, arm1 shape}
template <class Regime>
PolicyRun run_policy(std::uint32_t passes, double min_tokens1, Regime regime, std::uint64_t seed = 1, bool tie1 = true) {
    sp::BlockConfig bc;
    bc.clock_ms = [] { return g_clock; };
    sp::BlockDrafter a0(bc), a1(bc);
    sp::ArmSelectorConfig cfg;
    cfg.clock_ms = [] { return g_clock; };
    sp::ArmSelector sel(cfg);
    sel.add(a0, "bundled");
    sel.add(a1, "block", min_tokens1, {}, tie1);
    PolicyRun r;
    std::vector<std::uint32_t> emitted;
    for (std::uint32_t i = 0; i < passes; ++i) {
        const auto arm = sel.begin_pass();
        const auto shapes = regime(i);
        const Shape s = arm == 0 ? shapes.first : shapes.second;
        // integer tokens with the shape's mean: floor or ceil by a deterministic draw; wall +-10 % noise
        const double u = double(mix(seed * 1000003ULL + i) % 10000U) / 10000.0;
        const double fl = double(std::uint32_t(s.tokens));
        const auto n = std::uint32_t(fl + (u < s.tokens - fl ? 1.0 : 0.0));
        const double ms = s.ms * (0.9 + 0.2 * double(mix(seed * 7919ULL + i) % 1000U) / 1000.0);
        emitted.assign(n, 5U);
        sel.end_pass(arm, emitted, ms);
        r.tokens += n;
        r.ms += ms;
    }
    r.switches = sel.switches();
    r.probes1 = sel.stats(1).probes;
    r.passes1 = sel.stats(1).passes;
    r.final_active = sel.active();
    return r;
}

// Hy4 one node, measured shapes: prose bundled 21.70 tok/s at ~2.1 tokens/pass; block 16.98 with its
// arbiter mostly serial (~1.1 tokens/pass at 65 ms); code bundled 32.32 (3.05 tokens/pass), block 33.26 (3.88).
const Shape kProse0{2.1, 2.1 / 21.70 * 1000.0}, kProse1{1.1, 1.1 / 16.98 * 1000.0};
const Shape kCode0{3.05, 3.05 / 32.32 * 1000.0}, kCode1{3.88, 3.88 / 33.26 * 1000.0};

void test_prose_gate() {
    // the block arm is gated by 2.5 tokens per pass: never probed while the bundled arm emits ~2.1
    const auto r = run_policy(120, 2.5, [](std::uint32_t) { return std::pair{kProse0, kProse1}; });
    AS_CHECK(r.probes1 == 0 && r.passes1 == 0, "gated block arm probed %llu times", (unsigned long long)r.probes1);
    std::printf("prose gated: %.2f tok/s (bundled alone 21.70), probes %llu\n", r.tokens / r.ms * 1000.0,
                (unsigned long long)r.probes1);
}

void test_prose_ungated() {
    // without the gate the block arm is probed (first probe free), loses, and the loss stays within ~2 %
    const auto r = run_policy(120, 0.0, [](std::uint32_t) { return std::pair{kProse0, kProse1}; });
    const double rate = r.tokens / r.ms * 1000.0;
    AS_CHECK(r.probes1 >= 1, "the unmeasured block arm was never probed");
    AS_CHECK(r.final_active == 0 && r.switches == 0, "switched to the losing arm (switches %llu)", (unsigned long long)r.switches);
    AS_CHECK(rate >= 21.70 * 0.975, "prose rate %.2f more than 2.5 %% under the bundled arm", rate);
    std::printf("prose ungated: %.2f tok/s, probes %llu, probe passes %llu\n", rate, (unsigned long long)r.probes1,
                (unsigned long long)r.passes1);
}

void test_code_switch() {
    // the block arm is better by 2.9 %: the first probe moves the decode to it; the run ends within 1 % of it
    const auto r = run_policy(240, 2.5, [](std::uint32_t) { return std::pair{kCode0, kCode1}; });
    const double rate = r.tokens / r.ms * 1000.0;
    AS_CHECK(r.final_active == 1, "stayed on the bundled arm in code");
    AS_CHECK(rate >= 33.26 * 0.99, "code rate %.2f under 0.99 x the block arm 33.26", rate);
    std::printf("code: %.2f tok/s (block alone 33.26, bundled 32.32), switches %llu\n", rate, (unsigned long long)r.switches);
}

void test_regime_change() {
    // code then prose: the block arm wins code; in prose its rate falls below drop x the bundled arm's last rate and the
    // bundled arm is probed at once and takes over
    const auto regime = [](std::uint32_t i) { return i < 200 ? std::pair{kCode0, kCode1} : std::pair{kProse0, kProse1}; };
    const auto r = run_policy(400, 2.5, regime);
    AS_CHECK(r.final_active == 0, "did not return to the bundled arm in prose");
    AS_CHECK(r.switches >= 2, "switches %llu (expected code -> block, prose -> bundled)", (unsigned long long)r.switches);
    const auto whole = r.tokens / r.ms * 1000.0;
    const double ideal_ms = 200.0 * kCode1.ms + 200.0 * kProse0.ms;
    const double ideal = (200.0 * kCode1.tokens + 200.0 * kProse0.tokens) / ideal_ms * 1000.0;
    AS_CHECK(whole >= ideal * 0.95, "code -> prose run %.2f under 0.95 x the per-regime best %.2f", whole, ideal);
    std::printf("code->prose: %.2f tok/s (per-regime best %.2f), switches %llu\n", whole, ideal, (unsigned long long)r.switches);
}

void test_equal_arms_bounded() {
    // equal arms: hysteresis bounds the switches (at most one per probe interval)
    const auto r = run_policy(960, 0.0, [](std::uint32_t) { return std::pair{kCode0, kCode0}; }, 3, false);
    AS_CHECK(r.switches <= 960 / 48, "equal arms switched %llu times", (unsigned long long)r.switches);
    const double rate = r.tokens / r.ms * 1000.0;
    AS_CHECK(rate >= 32.32 * 0.97, "equal arms rate %.2f", rate);
    std::printf("equal arms: %.2f tok/s, switches %llu\n", rate, (unsigned long long)r.switches);
}

void test_context_limit() {
    // the block arm (better on code) has room only below 400 committed positions (a drafter with a context limit): it
    // takes over, then the bundled arm falls back in at the limit and the block arm never runs past it
    sp::BlockConfig bc;
    bc.clock_ms = [] { return g_clock; };
    sp::BlockDrafter a0(bc), a1(bc);
    sp::ArmSelectorConfig cfg;
    cfg.clock_ms = [] { return g_clock; };
    sp::ArmSelector sel(cfg);
    sel.add(a0, "bundled");
    constexpr std::uint64_t kLimit = 400;
    sel.add(a1, "block", 2.5, [](std::uint64_t context, std::uint32_t rows) { return context + rows <= kLimit; });
    std::uint64_t context = 0, late1 = 0;
    std::vector<std::uint32_t> emitted;
    bool block_active = false;
    for (std::uint32_t i = 0; i < 300; ++i) {
        const auto arm = sel.begin_pass(context, 8);
        if (sel.active() == 1) block_active = true;
        if (arm == 1 && context + 8 > kLimit) ++late1;
        const Shape s = arm == 0 ? kCode0 : kCode1;
        const std::uint32_t n = arm == 0 ? 3U : 4U;  // 31.8 vs 34.3 tok/s at the code walls
        emitted.assign(n, 5U);
        sel.end_pass(arm, emitted, s.ms);
        context += n;
    }
    AS_CHECK(block_active, "the block arm never took over below the limit");
    AS_CHECK(late1 == 0, "the block arm ran %llu passes past its limit", (unsigned long long)late1);
    AS_CHECK(sel.active() == 0 && sel.fallbacks() >= 1, "no fallback at the limit (active %zu, fallbacks %llu)", sel.active(),
             (unsigned long long)sel.fallbacks());
    std::printf("context limit: fallbacks %llu, block passes %llu\n", (unsigned long long)sel.fallbacks(),
                (unsigned long long)sel.stats(1).passes);
}

// ---- integration: BlockSpeculator over two chain drafters on one target ----
struct World {
    std::vector<std::uint32_t> prompt, response;
};

struct Target final : sp::VerifyTarget {
    const World& w;
    std::uint32_t committed = 0;
    std::size_t last_rows = 0;
    explicit Target(const World& world) : w(world) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        greedy.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const std::size_t pos = committed + i;
            greedy.push_back(pos + 1 < w.response.size() ? w.response[pos + 1] : 7U);
        }
        last_rows = tokens.size();
        g_clock += 50.0 + 10.0 * double(tokens.size());
    }
    void commit(std::uint32_t k) override {
        AS_CHECK(k >= 1 && k <= last_rows, "commit %u of %zu", k, last_rows);
        committed += k;
    }
};

// accuracy by response region: [0, split) and [split, end); cost: flat block or per step
struct Chain final : sp::ChainDrafter {
    const World& w;
    double acc_a, acc_b, flat_ms, step_ms;
    std::size_t split;
    std::uint64_t stream = 1, drafts = 0, accepts = 0;  // response tokens emitted (the anchor is response[0])
    Chain(const World& world, double a, double b, std::size_t s, double flat, double step)
        : w(world), acc_a(a), acc_b(b), flat_ms(flat), step_ms(step), split(s) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        ++drafts;
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t pos = stream + prefix.size() + j;
            const double acc = pos < split ? acc_a : acc_b;
            const bool right = double(mix(pos * 977ULL + 11ULL * std::uint64_t(flat_ms)) % 1000U) / 1000.0 < acc;
            out.push_back(pos < w.response.size() && right ? w.response[pos] : 3U);
        }
        g_clock += flat_ms + step_ms * double(prefix.size() + k);
    }
    void accept(std::span<const std::uint32_t> emitted) override {
        ++accepts;
        stream += emitted.size();
    }
};

void test_speculator_two_arms() {
    World w;
    for (std::uint32_t i = 0; i < 32; ++i) w.prompt.push_back(1000U + i);
    for (std::uint32_t i = 0; i < 900; ++i) w.response.push_back(100U + std::uint32_t(mix(i + 7U) % 50000U));
    const std::size_t split = 450;
    // arm 0: per-step chain, steady 0.6 acceptance; arm 1: flat block, 0.95 in the first half (deep drafts pay), 0.2 after
    Chain c0(w, 0.6, 0.6, split, 0.0, 3.0), c1(w, 0.95, 0.2, split, 9.0, 0.3);
    sp::BlockConfig bc;
    bc.clock_ms = [] { return g_clock; };
    bc.max_block = 8;
    bc.chain_max_k = 7;
    bc.arbiter.windows.clear();
    for (std::uint32_t r = 1; r <= 8; ++r) bc.arbiter.windows.push_back(r);
    sp::BlockDrafter d0(bc, &c0), d1(bc, &c1);
    for (std::uint32_t r = 1; r <= 8; ++r) {
        d0.seed_verify_ms(r, 50.0 + 10.0 * r);
        d1.seed_verify_ms(r, 50.0 + 10.0 * r);
    }
    sp::ArmSelectorConfig cfg;
    cfg.clock_ms = [] { return g_clock; };
    sp::ArmSelector sel(cfg);
    sel.add(d0, "bundled");
    sel.add(d1, "block");
    sel.commit(w.prompt);
    std::vector<std::uint32_t> out{w.response[0]};
    sel.commit(std::span<const std::uint32_t>(out.data(), 1));
    Target t(w);
    sp::BlockSpeculator spec(sel, t);
    spec.start(out[0]);
    std::uint64_t steps = 0;
    const std::uint32_t n = 800;
    while (out.size() < n) {
        sp::StepParams p;
        p.remaining = n - std::uint32_t(out.size());
        p.max_rows = 8;
        p.context = w.prompt.size() + t.committed;
        const auto r = spec.step(out, p);
        AS_CHECK(r.emitted > 0, "step emitted nothing");
        ++steps;
    }
    out.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) AS_CHECK(out[i] == w.response[i], "id %u differs from the greedy stream", i);
    // every pass reached both chains (drafted by one, followed by the other): equal streams, one accept per pass
    AS_CHECK(c0.accepts == steps && c1.accepts == steps, "accepts %llu / %llu of %llu passes",
             (unsigned long long)c0.accepts, (unsigned long long)c1.accepts, (unsigned long long)steps);
    AS_CHECK(c0.stream == c1.stream && c0.stream >= n, "chain streams %llu / %llu", (unsigned long long)c0.stream,
             (unsigned long long)c1.stream);
    AS_CHECK(sel.stats(0).passes + sel.stats(1).passes == steps, "pass count");
    AS_CHECK(sel.stats(1).passes > 0 && sel.switches() >= 1, "the block arm never ran (switches %llu)",
             (unsigned long long)sel.switches());
    std::printf("speculator: %llu passes, arm passes %llu / %llu, switches %llu, ids equal\n", (unsigned long long)steps,
                (unsigned long long)sel.stats(0).passes, (unsigned long long)sel.stats(1).passes,
                (unsigned long long)sel.switches());
}

void test_config_refused() {
    bool threw = false;
    try {
        sp::ArmSelectorConfig c;
        c.switch_in = 0.9;
        sp::ArmSelector s(c);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    AS_CHECK(threw, "switch_in below 1 accepted");
}
}  // namespace

int main() {
    test_config_refused();
    test_prose_gate();
    test_prose_ungated();
    test_code_switch();
    test_regime_change();
    test_equal_arms_bounded();
    test_context_limit();
    test_speculator_two_arms();
    std::printf("strix-spec-arm-selector-tests PASS\n");
    return 0;
}
