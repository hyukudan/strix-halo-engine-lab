// Frozen speculation prices (strix/spec/verify_cost_table.hpp, BlockDrafter::seed_chain_ms).
//   1. Seeded chain walls price the chain under fixed_costs (nearest seeded depth + learned slope).
//   2. A run priced from a table makes the same row choices under any host-clock noise (the invariant the A/B of two
//      builds needs), while it still measures warm per-depth chain walls for the next run.
//   3. Cost-table round trip: verify + draft entries price a run (table_prices_run) only when every window is priced.
#include "strix/spec/verify_cost_table.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
    std::uint64_t calls = 0, seed = 0;
    double amp = 0.0, spike = 0.0;
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t p = stream + prefix.size() + j;
            out.push_back(mix(p * 31U + 7U) % 100U < 70U ? truth(p) : 7U);
        }
        g_clock += 1.0 + 0.4 * double(k) + amp * double(mix(seed * 1000003ULL + calls) % 1000U) / 1000.0 +
                   (calls == 0 ? spike : 0.0);
        ++calls;
    }
    void accept(std::span<const std::uint32_t> emitted) override { stream += emitted.size(); }
};

constexpr std::uint32_t kWindows[] = {1U, 2U, 4U, 8U};
double verify_wall(std::uint32_t rows) { return 9.0 + 1.6 * double(rows); }

sp::BlockConfig frozen_config(bool frozen) {
    sp::BlockConfig cfg;
    cfg.arbiter.windows.assign(std::begin(kWindows), std::end(kWindows));
    cfg.max_block = 8;
    cfg.chain_max_k = 7;
    cfg.copy = false;
    cfg.hybrid = false;
    cfg.fixed_costs = frozen;
    if (!frozen) cfg.arbiter.cold_walls = 1;
    if (frozen) sp::set_context_cost(cfg.arbiter, "off");
    cfg.clock_ms = [] { return g_clock; };
    return cfg;
}

struct Trace {
    std::vector<std::uint32_t> picks;  // source * 64 + drafts per pass
    sp::BlockDrafter::ChainWall k3{};
};

// One greedy run of n tokens; verify walls carry jitter in [0, amp) and the first pass of a width +30 ms.
Trace run(bool frozen, std::uint64_t seed, double amp, double spike, std::uint32_t n) {
    g_clock = 0.0;
    NoisyChain chain;
    chain.seed = seed;
    chain.amp = amp;
    chain.spike = spike;
    sp::BlockDrafter drafter(frozen_config(frozen), &chain);
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
        t.picks.push_back(std::uint32_t(b.source) * 64U + std::uint32_t(b.drafts.size()));
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
    table_round_trip();
    std::printf("spec_frozen_costs PASS\n");
    return 0;
}
