// Frozen speculation prices (strix/spec/verify_cost_table.hpp, BlockDrafter::seed_chain_ms).
//   1. Seeded chain walls price the chain under fixed_costs (nearest seeded depth + learned slope).
//   2. A run priced from a table makes the same row choices under any host-clock noise (the invariant the A/B of two
//      builds needs), while it still measures warm per-depth chain walls for the next run.
//   3. Cost-table round trip: verify + draft entries price a run (table_prices_run) only when every window is priced.
#include "strix/serial_request.hpp"
#include "strix/spec/greedy_continuation.hpp"
#include "strix/spec/row_rule.hpp"
#include "strix/spec/verify_cost_table.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace sp = strix::spec;

#define FC_CHECK(c, ...)                                                                                    \
    do {                                                                                                    \
        if (!(c)) {                                                                                         \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                       \
            std::fprintf(stderr, __VA_ARGS__);                                                              \
            std::fprintf(stderr, "\n");                                                                     \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

namespace {

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}
std::uint32_t truth(std::uint64_t position) { return 1000U + std::uint32_t(position % 50000U); }
double g_clock = 0.0;

// Deterministic drafts (a hit at ~70 % of positions); every call advances the virtual clock by 1 + 0.4 k ms plus
// jitter in [0, amp) from its own seed, and the first call of a run by `spike` more (one-time work).
struct NoisyChain final : sp::ChainDrafter {
    std::uint64_t stream = 1;  // response tokens emitted (the anchor included)
    std::vector<std::uint32_t> alt;  // runner-ups of the last draft (set when `tree`): the target's token where the draft misses
    bool tree = false;
    std::uint64_t calls = 0, seed = 0;
    double amp = 0.0, spike = 0.0;
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        out.clear();
        alt.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t p = stream + prefix.size() + j;
            const bool hit = mix(p * 31U + 7U) % 100U < 70U;
            out.push_back(hit ? truth(p) : 7U);
            alt.push_back(hit ? 9U : truth(p));
        }
        g_clock += 1.0 + 0.4 * double(k) + amp * double(mix(seed * 1000003ULL + calls) % 1000U) / 1000.0 +
                   (calls == 0 ? spike : 0.0);
        ++calls;
    }
    void accept(std::span<const std::uint32_t> emitted) override { stream += emitted.size(); }
    [[nodiscard]] std::span<const std::uint32_t> runner_ups() const override {
        return tree ? std::span<const std::uint32_t>(alt) : std::span<const std::uint32_t>();
    }
};

constexpr std::uint32_t kWindows[] = {1U, 2U, 4U, 8U};
double verify_wall(std::uint32_t rows) { return 9.0 + 1.6 * double(rows); }

sp::BlockConfig frozen_config(bool frozen, bool prices = true) {
    sp::BlockConfig cfg;
    cfg.arbiter.windows.assign(std::begin(kWindows), std::end(kWindows));
    cfg.max_block = 8;
    cfg.chain_max_k = 7;
    cfg.copy = false;
    cfg.hybrid = false;
    cfg.fixed_costs = frozen;
    cfg.frozen_prices = frozen && prices;
    if (frozen && prices) sp::set_frozen_price_rules(cfg.arbiter);  // the serial CLI's frozen rules (priced exploration, stop-loss)
    if (!frozen) cfg.arbiter.cold_walls = 1;
    if (frozen) sp::set_context_cost(cfg.arbiter, "off");
    cfg.clock_ms = [] { return g_clock; };
    return cfg;
}

// A tree-capable verify target for the production continuation: model_next is the serial model, so the ids the
// continuation emits are right only when the verifier returns the right correction for every row and the commit keeps
// the right rows. A row whose input token is not the model's own token at its position yields noise.
std::uint32_t model_next(std::uint32_t prev, std::uint64_t position) {
    if (prev == truth(position)) return truth(position + 1U);
    return 1000U + std::uint32_t(mix(std::uint64_t(prev) * 131ULL + position) % 50000U);
}

struct ModelTarget final : sp::VerifyTarget {
    std::uint64_t committed;  // positions in the target; the anchor sits at `committed`
    std::uint64_t seed;
    double amp, spike;
    bool tree;
    std::vector<std::uint32_t> widths;  // rows of every verify (the purchase sequence)
    std::vector<bool> width_seen = std::vector<bool>(16, false);
    std::uint64_t pass = 0;
    ModelTarget(std::uint64_t start, std::uint64_t seed_, double amp_, double spike_, bool tree_)
        : committed(start), seed(seed_), amp(amp_), spike(spike_), tree(tree_) {}
    void wall(std::uint32_t rows) {
        double w = verify_wall(rows) + amp * double(mix(seed * 7919ULL + pass++) % 1000U) / 1000.0;
        if (rows < width_seen.size() && !width_seen[rows]) { width_seen[rows] = true; w += spike; }
        g_clock += w;
        widths.push_back(rows);
    }
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        wall(std::uint32_t(tokens.size()));
        greedy.resize(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) greedy[i] = model_next(tokens[i], committed + i);
    }
    void commit(std::uint32_t k) override { committed += k; }
    [[nodiscard]] std::uint32_t tree_rows() const override { return tree ? 8U : 0U; }
    void verify_tree(const sp::TreeRows& rows, std::vector<std::uint32_t>& greedy) override {
        wall(std::uint32_t(rows.tokens.size()));
        greedy.resize(rows.tokens.size());
        for (std::size_t r = 0; r < rows.tokens.size(); ++r) greedy[r] = model_next(rows.tokens[r], committed + rows.depth[r]);
    }
    void commit_path(std::span<const std::uint32_t> path) override { committed += path.size(); }
};

// The drafter of the production run: absolute-position truth, runner-ups where the draft misses.
struct PositionChain final : sp::ChainDrafter {
    std::vector<std::uint32_t> alt;
    bool tree;
    std::uint64_t calls = 0, seed;
    std::uint64_t stream = 4;  // absolute position after the anchor (prompt 3 + anchor); accept() advances it
    double amp, spike;
    PositionChain(bool tree_, std::uint64_t seed_, double amp_, double spike_)
        : tree(tree_), seed(seed_), amp(amp_), spike(spike_) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        out.clear();
        alt.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t p = stream + prefix.size() + j;
            const bool hit = mix(p * 31U + 7U) % 100U < 70U;
            out.push_back(hit ? truth(p) : 7U);
            alt.push_back(hit ? 9U : truth(p));
        }
        g_clock += 1.0 + 0.4 * double(k) + amp * double(mix(seed * 1000003ULL + calls) % 1000U) / 1000.0 +
                   (calls == 0 ? spike : 0.0);
        ++calls;
    }
    void accept(std::span<const std::uint32_t> emitted) override { stream += emitted.size(); }
    [[nodiscard]] std::span<const std::uint32_t> runner_ups() const override {
        return tree ? std::span<const std::uint32_t>(alt) : std::span<const std::uint32_t>();
    }
};

struct Production {
    std::vector<std::uint32_t> ids, serial;
    std::vector<std::uint32_t> widths;
    std::uint64_t siblings = 0, accepted = 0;
};

// The shared continuation (execute_block_token_continuation) over the model target, with its serial reference.
Production produce(bool prices, std::uint64_t seed, double amp, double spike, bool tree, std::uint32_t n) {
    g_clock = 0.0;
    const std::vector<std::uint32_t> prompt{1U, 2U, 3U};
    const std::uint64_t start = prompt.size();
    const strix::SerialTokenContinuationRequest request{truth(start), start, 100000U, start + n + 64U, n, 99999U};
    Production out;
    const auto serial = strix::execute_serial_token_continuation(
        request, [](std::uint32_t input, std::uint64_t position) { return model_next(input, position); });
    out.serial = serial.generated_tokens;
    PositionChain chain(tree, seed, amp, spike);
    ModelTarget target(start, seed, amp, spike, tree);
    sp::BlockDrafter drafter(frozen_config(true, prices), &chain);
    for (const auto r : kWindows) drafter.seed_verify_ms(r, verify_wall(r));
    drafter.seed_chain_ms(1, 1.4);
    drafter.seed_chain_ms(4, 2.6);
    const auto result = sp::execute_block_token_continuation(request, prompt, drafter, target, 8U);
    out.ids = result.generated_tokens;
    out.widths = target.widths;
    out.siblings = drafter.stats().sibling_rows;
    for (const auto a : drafter.stats().source_accepted) out.accepted += a;
    return out;
}

struct Trace {
    std::vector<std::uint32_t> picks;  // siblings * 4096 + source * 64 + drafts per pass
    std::vector<std::uint32_t> ids;    // every emitted id, in order
    std::uint64_t siblings = 0;        // sibling rows bought
    sp::BlockDrafter::ChainWall k3{};
};

// One greedy run of n tokens; verify walls carry jitter in [0, amp) and the first pass of a width +30 ms.
Trace run(bool frozen, std::uint64_t seed, double amp, double spike, std::uint32_t n, bool tree = false,
          bool prices = true) {
    g_clock = 0.0;
    NoisyChain chain;
    chain.seed = seed;
    chain.amp = amp;
    chain.spike = spike;
    chain.tree = tree;
    sp::BlockDrafter drafter(frozen_config(frozen, prices), &chain);
    for (const auto r : kWindows) drafter.seed_verify_ms(r, verify_wall(r));
    if (frozen) {
        drafter.seed_chain_ms(1, 1.4);
        drafter.seed_chain_ms(4, 2.6);
    }
    const std::vector<std::uint32_t> prompt{1U, 2U, 3U};
    drafter.commit(prompt);
    std::uint64_t emitted_total = 1;  // the anchor (response position 0)
    const std::uint32_t anchor = truth(0);
    drafter.commit(std::span<const std::uint32_t>(&anchor, 1));
    Trace t;
    std::vector<bool> width_seen(9, false);
    std::uint64_t pass = 0;
    while (emitted_total < n) {
        sp::StepParams p;
        p.remaining = n - std::uint32_t(emitted_total);
        p.max_rows = 8;
        p.tree_rows = tree ? 8U : 0U;
        p.context = prompt.size() + emitted_total;
        const auto b = drafter.next(p);
        std::uint32_t accepted = 0;
        while (accepted < b.drafts.size() && b.drafts[accepted] == truth(emitted_total + accepted)) ++accepted;
        std::vector<std::uint32_t> emitted(b.drafts.begin(), b.drafts.begin() + accepted);
        emitted.push_back(truth(emitted_total + accepted));
        const auto rows = b.rows();
        double wall = verify_wall(rows) + amp * double(mix(seed * 7919ULL + pass++) % 1000U) / 1000.0;
        if (rows < width_seen.size() && !width_seen[rows]) { width_seen[rows] = true; wall += 30.0; }
        g_clock += wall;
        drafter.observe(b, accepted, wall, emitted, p.context);
        emitted_total += emitted.size();
        t.picks.push_back(b.siblings * 4096U + std::uint32_t(b.source) * 64U + std::uint32_t(b.drafts.size()));
        t.siblings += b.siblings;
        t.ids.insert(t.ids.end(), emitted.begin(), emitted.end());
    }
    t.k3 = drafter.measured_chain_ms(3);
    return t;
}

void chain_seed_prices() {
    sp::BlockDrafter d(frozen_config(true), nullptr);
    FC_CHECK(std::fabs(d.chain_cost_ms(3) - (1.5 + 1.5 * 2.0)) < 1e-12, "unseeded fixed chain = flush + step (k - 1)");
    d.seed_chain_ms(1, 2.0);
    d.seed_chain_ms(4, 3.5);
    FC_CHECK(std::fabs(d.chain_cost_ms(1) - 2.0) < 1e-12 && std::fabs(d.chain_cost_ms(4) - 3.5) < 1e-12,
             "seeded depths price themselves: %g %g", d.chain_cost_ms(1), d.chain_cost_ms(4));
    FC_CHECK(std::fabs(d.chain_cost_ms(2) - 2.5) < 1e-12 && std::fabs(d.chain_cost_ms(8) - 5.5) < 1e-12,
             "unseeded depths extend the nearest seeded one at the seeded slope: %g %g", d.chain_cost_ms(2),
             d.chain_cost_ms(8));
    FC_CHECK(d.measured_chain_ms(1).samples == 0U, "a seed is not a measurement of this run");
    d.seed_chain_ms(2, -1.0);
    d.seed_chain_ms(2, std::nan(""));
    FC_CHECK(std::fabs(d.chain_cost_ms(2) - 2.5) < 1e-12, "invalid seeds are ignored");
}

void frozen_choices_ignore_clock() {
    const auto a = run(true, 1, 0.0, 0.0, 400);
    const auto b = run(true, 2, 6.0, 25.0, 400);
    const auto c = run(true, 3, 3.0, 0.0, 400);
    FC_CHECK(a.picks == b.picks && a.picks == c.picks, "frozen row choices differ under clock noise (%zu %zu %zu passes)",
             a.picks.size(), b.picks.size(), c.picks.size());
    // The harness is sensitive: the learned form (in-run walls price) changes its row choices under the same noise.
    const auto la = run(false, 1, 0.0, 0.0, 400);
    const auto lb = run(false, 2, 6.0, 25.0, 400);
    FC_CHECK(la.picks != lb.picks, "learned control: clock noise did not move any choice (harness not sensitive)");
    std::uint32_t model = 0;
    for (const auto x : a.picks) model += x / 64U == std::uint32_t(sp::Source::model) ? 1U : 0U;
    FC_CHECK(model > 10U, "the chain drafter runs (%u model passes)", model);
    // Warm walls are still measured: chain k = 3 costs 1 + 1.2 + jitter in [0, 6) (no spike: the first wall is cold).
    if (b.k3.samples > 0U)
        FC_CHECK(b.k3.ms >= 2.2 && b.k3.ms < 8.2, "measured warm chain wall k=3 %g", b.k3.ms);
    std::printf("frozen: %zu passes, %u model, identical under clock noise; warm chain k3 %.3f ms x%u\n", a.picks.size(),
                model, b.k3.ms, b.k3.samples);
}

// A fixed-price block with a tree target and runner-ups never buys a sibling from this run's clock: the stacked purchase
// rule priced rows from the measured linear / tree walls, the surcharge and the recent rate. Frozen prices buy none, so the
// row choices and the emitted ids are the same under any noise.
void frozen_tree_purchases_ignore_clock() {
    const auto a = run(true, 1, 0.0, 0.0, 400, true);
    const auto b = run(true, 2, 6.0, 25.0, 400, true);
    const auto c = run(true, 3, 3.0, 0.0, 400, true);
    FC_CHECK(a.picks == b.picks && a.picks == c.picks, "frozen tree row choices differ under clock noise");
    FC_CHECK(a.siblings == 0 && b.siblings == 0 && c.siblings == 0, "frozen prices bought siblings (%llu)",
             (unsigned long long)a.siblings);
    // The production continuation over a tree-capable target: its emitted ids are the serial reference's under every
    // clock, and the row purchases do not move with the clock.
    const auto pa = produce(true, 1, 0.0, 0.0, true, 300);
    const auto pb = produce(true, 2, 6.0, 25.0, true, 300);
    const auto pc = produce(true, 3, 3.0, 0.0, true, 300);
    FC_CHECK(pa.serial.size() == 300U, "serial reference length %zu", pa.serial.size());
    FC_CHECK(pa.ids == pa.serial && pb.ids == pa.serial && pc.ids == pa.serial,
             "production continuation ids differ from serial decode (frozen, tree target)");
    FC_CHECK(pa.widths == pb.widths && pa.widths == pc.widths, "production row purchases differ under clock noise");
    FC_CHECK(pa.siblings == 0 && pb.siblings == 0 && pc.siblings == 0, "frozen production bought siblings");
    FC_CHECK(pa.accepted > 0U, "harness drafted nothing the target accepted");
    const auto qa = produce(false, 1, 0.0, 0.0, true, 300);
    const auto qb = produce(false, 2, 6.0, 25.0, true, 300);
    FC_CHECK(qa.ids == qa.serial && qb.ids == qa.serial, "production ids differ from serial decode (tree rows bought)");
    FC_CHECK(qa.siblings > 0U && qb.siblings > 0U, "production control bought no siblings (%llu %llu)",
             (unsigned long long)qa.siblings, (unsigned long long)qb.siblings);
    // Sensitivity: the same fixed-cost run without frozen prices buys siblings and decides from the clock.
    const auto la = run(true, 1, 0.0, 0.0, 400, true, false);
    const auto lb = run(true, 2, 6.0, 25.0, 400, true, false);
    FC_CHECK(la.siblings > 0 && lb.siblings > 0, "control bought no siblings (%llu %llu): harness cannot show the defect",
             (unsigned long long)la.siblings, (unsigned long long)lb.siblings);
    FC_CHECK(la.picks != lb.picks, "control: clock noise did not move any sibling choice (harness not sensitive)");
    std::printf("frozen tree: 0 siblings, identical choices and ids under noise; control %llu / %llu sibling rows\n",
                (unsigned long long)la.siblings, (unsigned long long)lb.siblings);
}

// Switching the row rule restarts the observations and keeps the seeds: the seeded depth-1 wall prices the sampled rule
// as it priced the greedy one, and warm chain samples of the greedy rule do not carry over.
void row_rule_switch_keeps_seeds() {
    const sp::RowRule sampled(strix::TokenSamplingPolicy{.temperature = 0.5F, .seed = 7}, 37U);
    sp::BlockDrafter d(frozen_config(true), nullptr);
    d.seed_chain_ms(1, 9.0);
    d.seed_chain_ms(4, 12.0);
    const double d1 = d.chain_cost_ms(1), d3 = d.chain_cost_ms(3), d8 = d.chain_cost_ms(8);
    FC_CHECK(std::fabs(d1 - 9.0) < 1e-12, "seeded depth 1 = 9 ms (%g)", d1);
    d.set_row_rule(sampled);
    FC_CHECK(std::fabs(d.chain_cost_ms(1) - d1) < 1e-12 && std::fabs(d.chain_cost_ms(3) - d3) < 1e-12 &&
                 std::fabs(d.chain_cost_ms(8) - d8) < 1e-12,
             "sampled rule loses the seeded chain prices: %g %g %g vs %g %g %g", d.chain_cost_ms(1), d.chain_cost_ms(3),
             d.chain_cost_ms(8), d1, d3, d8);
    d.set_row_rule(sp::RowRule{});
    FC_CHECK(std::fabs(d.chain_cost_ms(1) - d1) < 1e-12 && std::fabs(d.chain_cost_ms(3) - d3) < 1e-12,
             "switching back loses the seeds");
    // Observations: a warm greedy run measures depth 3; the sampled rule starts without those samples.
    const auto g = run(true, 2, 0.0, 0.0, 200);  // control for the harness (measured_chain_ms needs a chain)
    FC_CHECK(g.k3.samples > 0U, "harness measured a warm chain wall");
    g_clock = 0.0;
    NoisyChain chain;
    sp::BlockDrafter w(frozen_config(true), &chain);
    for (const auto r : kWindows) w.seed_verify_ms(r, verify_wall(r));
    w.seed_chain_ms(1, 1.4);
    w.seed_chain_ms(4, 2.6);
    const std::vector<std::uint32_t> prompt{1U, 2U, 3U};
    w.commit(prompt);
    std::uint64_t emitted_total = 1;
    for (int i = 0; i < 40 && emitted_total < 300; ++i) {
        sp::StepParams p;
        p.remaining = 300U - std::uint32_t(emitted_total);
        p.max_rows = 8;
        p.context = prompt.size() + emitted_total;
        const auto b = w.next(p);
        std::uint32_t accepted = 0;
        while (accepted < b.drafts.size() && b.drafts[accepted] == truth(emitted_total + accepted)) ++accepted;
        std::vector<std::uint32_t> emitted(b.drafts.begin(), b.drafts.begin() + accepted);
        emitted.push_back(truth(emitted_total + accepted));
        w.observe(b, accepted, verify_wall(b.rows()), emitted, p.context);
        emitted_total += emitted.size();
    }
    std::uint32_t warm = 0;
    for (std::uint32_t k = 1; k <= 8; ++k) warm += w.measured_chain_ms(k).samples;
    FC_CHECK(warm > 0U, "greedy run measured warm chain walls");
    w.set_row_rule(sampled);
    for (std::uint32_t k = 1; k <= 8; ++k)
        FC_CHECK(w.measured_chain_ms(k).samples == 0U, "warm greedy chain samples survived the rule switch (k=%u)", k);
    FC_CHECK(std::fabs(w.chain_cost_ms(1) - 1.4) < 1e-12 && std::fabs(w.chain_cost_ms(4) - 2.6) < 1e-12,
             "seeds after the switch: %g %g", w.chain_cost_ms(1), w.chain_cost_ms(4));
    std::printf("row rule switch: seeded chain prices kept, warm observations restarted\n");
}

// The persisted verify walls belong to one row rule: after a switch, the warm price of a width is the mean of the new
// rule's passes without its first, not a mean pooled with the old rule's passes.
void row_rule_switch_restarts_verify_walls() {
    const sp::RowRule sampled(strix::TokenSamplingPolicy{.temperature = 0.5F, .seed = 7}, 37U);
    sp::BlockDrafter d(frozen_config(true), nullptr);
    const sp::Block serial{};  // one physical row
    const std::uint32_t id = 7U;
    const auto feed = [&](double ms) { d.observe(serial, 0U, ms, std::span<const std::uint32_t>(&id, 1U), 8U); };
    feed(10.0);
    feed(20.0);
    FC_CHECK(std::fabs(d.stats().warm_ms(1) - 20.0) < 1e-12, "greedy warm wall %g", d.stats().warm_ms(1));
    d.set_row_rule(sampled);
    FC_CHECK(d.stats().warm_ms(1) < 0.0, "the old rule's walls survived the switch (%g)", d.stats().warm_ms(1));
    feed(40.0);
    feed(60.0);
    FC_CHECK(std::fabs(d.stats().warm_ms(1) - 60.0) < 1e-12, "sampled warm wall %g, expected 60", d.stats().warm_ms(1));
    const auto dir = std::filesystem::temp_directory_path() / ("strix-frozen-costs-sw-" + std::to_string(mix(777)));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const strix::CostTableKey key{"strix-frozen-costs-test", "bin", "dev", "model", "policy", 0U};
    const strix::CostTableOptions options{0.0, 0.15};
    strix::CostTableCache t(dir, key, options);
    (void)t.load();
    FC_CHECK(sp::record_verify_costs(t, d.stats()) == 1U, "one width recorded");
    FC_CHECK(t.store(), "store");
    strix::CostTableCache u(dir, key, options);
    (void)u.load();
    const auto e = u.warm(sp::verify_cost_entry(1));
    FC_CHECK(e && std::fabs(e->ms - 60.0) < 1e-9, "saved entry after the switch %g, expected 60", e ? e->ms : -1.0);
    std::filesystem::remove_all(dir);
    std::printf("row rule switch: persisted verify wall is the new rule's warm mean (60 ms)\n");
}

void table_round_trip() {
    const auto dir = std::filesystem::temp_directory_path() / ("strix-frozen-costs-" + std::to_string(mix(12345)));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const strix::CostTableKey key{"strix-frozen-costs-test", "bin", "dev", "model", "policy", 0U};
    const strix::CostTableOptions options{0.0, 0.15};
    const std::vector<std::uint32_t> windows(std::begin(kWindows), std::end(kWindows));
    {
        strix::CostTableCache t(dir, key, options);
        (void)t.load();
        FC_CHECK(!sp::table_prices_run(t, windows, 8, 7), "an empty table prices nothing");
        for (const auto r : {1U, 2U, 4U}) t.record(sp::verify_cost_entry(r), {verify_wall(r), 4});
        t.record(sp::draft_cost_entry(3), {2.2, 9});
        FC_CHECK(t.store(), "store");
    }
    {
        strix::CostTableCache t(dir, key, options);
        (void)t.load();
        FC_CHECK(!sp::table_prices_run(t, windows, 8, 7), "R8 unpriced: learned form");
        FC_CHECK(sp::table_prices_run(t, windows, 4, 3), "every window up to 4 rows and a chain depth priced");
        FC_CHECK(sp::table_prices_run(t, windows, 4, 0), "no chain: verify windows suffice");
        sp::BlockDrafter d(frozen_config(true), nullptr);
        FC_CHECK(sp::seed_draft_costs(t, d, 7) == 1U && std::fabs(d.chain_cost_ms(3) - 2.2) < 1e-12, "draft seed");
        t.record(sp::verify_cost_entry(8), {verify_wall(8), 2});
        FC_CHECK(t.store(), "store 2");
    }
    {
        strix::CostTableCache t(dir, key, options);
        (void)t.load();
        FC_CHECK(sp::table_prices_run(t, windows, 8, 7), "complete table prices the run");
        strix::CostTableCache u(dir, key, options);
        (void)u.load();
        sp::BlockDrafter d(frozen_config(true), nullptr);
        FC_CHECK(sp::record_draft_costs(u, d, 7) == 0U, "no measured chain wall: nothing recorded");
    }
    std::filesystem::remove_all(dir);
    std::printf("table: verify + draft entries round trip; prices_run only when complete\n");
}

}  // namespace

int main() {
    chain_seed_prices();
    frozen_choices_ignore_clock();
    frozen_tree_purchases_ignore_clock();
    row_rule_switch_restarts_verify_walls();
    row_rule_switch_keeps_seeds();
    table_round_trip();
    std::printf("spec_frozen_costs PASS\n");
    return 0;
}
