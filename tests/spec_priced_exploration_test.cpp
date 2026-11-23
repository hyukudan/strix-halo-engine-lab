// Frozen-price decision rules of the shared Arbiter (set_frozen_price_rules: priced exploration + cycle stop-loss),
// replayed against one device ticket of a learned per-depth drafter (three contents, 255 tokens each).
// - The cost model is checked first: the ticket's own pass ledger (passes x mean wall per width + draft wall) must give
//   the measured block / plain-decode ratio.
// - The acceptance stream: per-depth conditional rates realized on the ticket's drafted passes. Prose changes regime
//   at its turn boundary (emitted token 122), so its stream alternates x (1 + swing) / x (1 - swing) stretches of 124
//   tokens; the ticket's rule replayed on it must reproduce the device's pass mix and drafted depth counts.
// - Decisions use the frozen table only; the simulated device charges each content's warm walls, the one-time first
//   use of a width and of the drafter, against the measured plain decode of the same content.
// - Gates: the cycle rule gains on prose, never loses on i.i.d. prose, keeps >= 97 % of the code / structured gain,
//   reaches plain-decode speed on prose once verify widths and the drafter are installed before the decode clock, and
//   decides deterministically.
#include "strix/spec/drafting.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace strix::spec;

int failures = 0;
#define PE_CHECK(cond, ...)                                                                                            \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                                  \
            std::fprintf(stderr, __VA_ARGS__);                                                                         \
            std::fprintf(stderr, "\n");                                                                                \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

// The frozen table every block arm of the ticket loaded (verify r1 r2 r4 r8; draft k1 k3 k7 and the drafter's own
// extension between them). verify.r1 is ONE warm pass of the last fill content: the cache keeps the last writer.
const std::array<std::uint32_t, 4> kWindows{1, 2, 4, 8};
const std::map<std::uint32_t, double> kPrice{{1, 9.3665}, {2, 16.1154}, {4, 18.8677}, {8, 27.6129}};
const std::array<double, 7> kDraft{2.2112, 2.3190, 1.5737, 1.6815, 1.7894, 1.8972, 2.0051};
constexpr std::uint32_t kTokens = 255;
constexpr double kFirstDraftMs = 15.0;  // drafter first use: measured draft wall minus the priced walls of its passes

// One content: realized conditional acceptance per depth (depth_accepted / depth_offered), warm walls per width, the
// extra wall of a width's first pass (mean x n - warm x (n - 1) - warm), the measured plain decode and block ratio.
// R2 never ran on the device: its table price stands in for the wall and the R4 first use for its install.
struct Content {
    const char* name;
    std::array<double, 7> rate;
    std::map<std::uint32_t, double> wall;
    std::map<std::uint32_t, double> first;
    double off_tok_s;
    double measured_ratio;
    double swing = 0.0;
    std::uint32_t stretch = 1;
};

enum class Rule { ticket, cycle };
enum class FirstUse { all, plain_only };  // plain_only: verify widths and drafter installed before the decode clock

struct Run {
    double ms = 0.0, serial_ms = 0.0;
    std::map<std::uint32_t, std::uint32_t> rows;
    std::array<std::uint32_t, 7> offered{}, accepted{};
    std::string trace;
    [[nodiscard]] double ratio() const { return serial_ms / ms; }
};

struct Lcg {
    std::uint64_t s;
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

Run replay(const Content& c, Rule rule, std::uint64_t seed, FirstUse first_use) {
    ArbiterConfig config;
    config.windows.assign(kWindows.begin(), kWindows.end());
    config.learn_costs = false;  // frozen prices (the serial CLI with a complete cost table)
    set_context_cost(config, "off");
    if (rule == Rule::cycle) set_frozen_price_rules(config);
    else config.priced_exploration = true;  // the ticket's rule
    Arbiter arbiter(config);
    for (const auto& [rows, ms] : kPrice) arbiter.seed_cost(rows, ms);
    Lcg rng{seed};
    Run run;
    run.serial_ms = 1000.0 * double(kTokens) / c.off_tok_s;
    std::map<std::uint32_t, bool> used;
    bool drafted = false;
    std::uint32_t done = 0;
    std::vector<std::uint32_t> emitted;
    while (done < kTokens) {
        StepInput in;
        in.model_max_k = 7;
        in.model_draft_ms = std::span<const double>(kDraft);
        in.remaining = kTokens - done;
        in.max_rows = 8;
        in.context = 46 + done;
        const Decision d = arbiter.decide(in);
        Outcome o;
        o.source = d.source;
        o.rows = d.rows;
        o.context = in.context;
        std::uint32_t accepted = 0;
        if (d.source == Source::model) {
            o.proposed = d.k;
            const double f = (done / c.stretch) % 2U == 0U ? 1.0 + c.swing : 1.0 - c.swing;
            while (accepted < d.k) {
                ++run.offered[accepted];
                if (!(rng.next() < std::min(0.99, c.rate[accepted] * f))) break;
                ++run.accepted[accepted++];
            }
            o.accepted = accepted;
        }
        double ms = c.wall.at(d.rows);
        if (!used[d.rows] && (first_use == FirstUse::all || d.rows == 1)) ms += c.first.at(d.rows);
        used[d.rows] = true;
        o.verify_ms = ms;
        if (d.source == Source::model) {
            o.draft_ms = kDraft[d.k - 1] + (first_use == FirstUse::all && !drafted ? kFirstDraftMs : 0.0);
            drafted = true;
        }
        run.ms += ms + o.draft_ms;
        ++run.rows[d.rows];
        run.trace += std::to_string(d.rows) + (d.probe ? "p " : " ");
        emitted.assign(accepted + 1U, 7U);
        arbiter.observe(o, emitted);
        done += accepted + 1U;
    }
    return run;
}

struct Summary {
    double ratio = 0.0, lo = 1e9, hi = 0.0, installed = 0.0;
    std::map<std::uint32_t, double> rows;
    std::array<double, 5> offered{}, accepted{};
};

Summary summarize(const Content& c, Rule rule) {
    Summary s;
    constexpr int seeds = 16;
    for (int i = 0; i < seeds; ++i) {
        const auto seed = 0x9E3779B97F4A7C15ULL * std::uint64_t(i + 1);
        const Run a = replay(c, rule, seed, FirstUse::all);
        s.ratio += a.ratio() / seeds;
        s.lo = std::min(s.lo, a.ratio());
        s.hi = std::max(s.hi, a.ratio());
        s.installed += replay(c, rule, seed, FirstUse::plain_only).ratio() / seeds;
        for (const auto& [rows, n] : a.rows) s.rows[rows] += double(n) / seeds;
        for (std::size_t j = 0; j < s.offered.size(); ++j) {
            s.offered[j] += double(a.offered[j]) / seeds;
            s.accepted[j] += double(a.accepted[j]) / seeds;
        }
        PE_CHECK(replay(c, rule, seed, FirstUse::all).trace == a.trace, "%s: decisions are not deterministic", c.name);
    }
    std::printf("%-9s %-6s ratio %.3f [%.3f, %.3f] installed %.3f rows", c.name, rule == Rule::cycle ? "cycle" : "ticket",
                s.ratio, s.lo, s.hi, s.installed);
    for (const auto& [rows, n] : s.rows) std::printf(" R%u:%.0f", rows, n);
    std::printf(" offered");
    for (const auto v : s.offered) std::printf(" %.0f", v);
    std::printf(" accepted");
    for (const auto v : s.accepted) std::printf(" %.0f", v);
    std::printf("\n");
    return s;
}

}  // namespace

int main() {
    // Ticket receipts (block round 1 walls; ratios = mean of both rounds).
    const Content prose{"prose", {0.652, 0.489, 0.591, 0.75, 0.3, 0.5, 0.5},
                        {{1, 9.015}, {2, 16.115}, {4, 18.276}, {8, 27.083}},
                        {{1, 23.57}, {2, 29.30}, {4, 29.30}, {8, 16.53}}, 111.53, 0.928, 0.25, 124};
    Content prose_iid = prose;
    prose_iid.name = "prose-iid";
    prose_iid.swing = 0.0;
    const Content code{"code", {0.907, 0.816, 0.75, 0.889, 0.917, 0.773, 0.941},
                       {{1, 10.89}, {2, 16.115}, {4, 20.43}, {8, 31.74}},
                       {{1, 23.43}, {2, 30.32}, {4, 30.32}, {8, 16.56}}, 93.65, 1.463};
    const Content bstruct{"bstruct", {0.795, 0.707, 0.829, 0.765, 0.923, 0.917, 1.0},
                          {{1, 9.37}, {2, 16.115}, {4, 18.86}, {8, 27.73}},
                          {{1, 22.68}, {2, 31.62}, {4, 31.62}, {8, 16.50}}, 109.23, 1.297};

    // 1. Cost model: the prose ticket's pass ledger (R1 103 x 9.244, R4 61 x 18.756, R8 8 x 29.150, draft 127.04 ms).
    const double ledger_ms = 103 * 9.24397 + 61 * 18.7562 + 8 * 29.1503 + 127.036;
    const double ledger_ratio = 1000.0 * double(kTokens) / prose.off_tok_s / ledger_ms;
    PE_CHECK(std::abs(ledger_ratio / prose.measured_ratio - 1.0) <= 0.02, "ledger %.3f vs device %.3f", ledger_ratio,
             prose.measured_ratio);
    std::printf("prose     ledger %.3f (device %.3f)\n", ledger_ratio, prose.measured_ratio);

    std::map<const Content*, Summary> ticket, cycle;
    for (const Content* c : std::array<const Content*, 4>{&prose, &prose_iid, &code, &bstruct}) {
        ticket[c] = summarize(*c, Rule::ticket);
        cycle[c] = summarize(*c, Rule::cycle);
    }
    // 2. The stream: the ticket's rule reproduces the device's decisions (passes per width, drafted depth counts) and
    //    the device's single draw lies inside the replay's seed range. Device: R1 103 R4 61 R8 8, offered 69 45 22,
    //    accepted 45 22 13. The replay mean sits above the draw (fewer accepted at depth 3 on the device).
    const auto& t = ticket[&prose];
    PE_CHECK(std::abs(t.rows.at(4) - 61.0) <= 8.0 && std::abs(t.offered[0] - 69.0) <= 6.0 &&
                 std::abs(t.accepted[0] - 45.0) <= 5.0,
             "prose stream does not reproduce the device decisions (R4 %.0f, offered %.0f, accepted %.0f)", t.rows.at(4),
             t.offered[0], t.accepted[0]);
    PE_CHECK(prose.measured_ratio >= t.lo - 0.005 && prose.measured_ratio <= t.hi, "device %.3f outside [%.3f, %.3f]",
             prose.measured_ratio, t.lo, t.hi);
    // 3. Prose: the stop-loss gains on the regime stream and never loses on the i.i.d. one.
    PE_CHECK(cycle[&prose].ratio >= t.ratio + 0.012, "prose cycle %.3f vs ticket %.3f", cycle[&prose].ratio, t.ratio);
    PE_CHECK(cycle[&prose_iid].ratio >= ticket[&prose_iid].ratio - 0.003, "i.i.d. prose cycle %.3f vs ticket %.3f",
             cycle[&prose_iid].ratio, ticket[&prose_iid].ratio);
    // 4. With verify widths and the drafter installed before the decode clock, prose reaches plain decode.
    PE_CHECK(cycle[&prose].installed >= 1.0, "installed prose %.3f of plain decode", cycle[&prose].installed);
    // 5. Contents whose acceptance pays keep >= 97 % of the ticket rule's gain.
    for (const auto* c : {&code, &bstruct}) {
        const double kept = (cycle[c].ratio - 1.0) / (ticket[c].ratio - 1.0);
        PE_CHECK(kept >= 0.97, "%s keeps %.1f %% of the gain (%.3f -> %.3f)", c->name, 100.0 * kept, ticket[c].ratio,
                 cycle[c].ratio);
        std::printf("%-9s gain kept %.1f %% (device ticket %.3f)\n", c->name, 100.0 * kept, c->measured_ratio);
    }
    if (failures == 0) std::printf("spec priced exploration: ok\n");
    return failures == 0 ? 0 : 1;
}
