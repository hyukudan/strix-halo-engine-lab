// Bars of the shared Arbiter at frozen prices (set_frozen_price_rules: priced exploration + cycle stop-loss).
//   1. A barred model arm is not a candidate: the best remaining arm is bought (a lookup that beats serial) and the
//      decision equals the one of the same state without a drafter.
//   2. A barred arm admits no exploration, also while the configured minimum model-pass count is not met.
//   3. The stop-loss ledger ages on every observed emission: a long serial interval leaves no stale model loss.
#include "strix/spec/drafting.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace sp = strix::spec;

#define FB_CHECK(c, ...)                                                                                    \
    do {                                                                                                    \
        if (!(c)) {                                                                                         \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                       \
            std::fprintf(stderr, __VA_ARGS__);                                                              \
            std::fprintf(stderr, "\n");                                                                     \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

namespace {

constexpr std::array<double, 7> kDraft{1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6};

sp::ArbiterConfig frozen(std::uint32_t min_model_passes) {
    sp::ArbiterConfig c;
    c.windows = {1, 2, 4, 8};
    c.learn_costs = false;
    sp::set_context_cost(c, "off");
    sp::set_frozen_price_rules(c);
    c.min_model_passes = min_model_passes;
    return c;
}

void seed(sp::Arbiter& a) {
    a.seed_cost(1, 9.0);
    a.seed_cost(2, 10.0);
    a.seed_cost(4, 12.0);
    a.seed_cost(8, 16.0);
}

sp::Proposal proposal() {
    sp::Proposal p;
    p.tokens = {10, 11, 12, 13, 14, 15, 16};
    p.match_length = 8;
    return p;
}

sp::StepInput input(const sp::Proposal* lookup, bool model, std::uint64_t context) {
    sp::StepInput in;
    in.lookup = lookup;
    if (model) {
        in.model_max_k = 7;
        in.model_draft_ms = std::span<const double>(kDraft);
    }
    in.remaining = 1000;
    in.max_rows = 8;
    in.context = context;
    return in;
}

// Observes `d` as run: every draft accepted, `emitted_tokens` counted (the lookup proposal's own ids when it ran).
void observe(sp::Arbiter& a, const sp::Decision& d, const sp::StepInput& in, std::size_t emitted_tokens) {
    sp::Outcome o;
    o.source = d.source;
    o.rows = d.rows;
    o.context = in.context;
    o.proposed = d.k;
    o.accepted = d.k;
    if (d.source == sp::Source::lookup) o.bucket = in.lookup->bucket();
    std::vector<std::uint32_t> emitted;
    for (std::size_t i = 0; i < emitted_tokens; ++i)
        emitted.push_back(in.lookup && i < in.lookup->tokens.size() ? in.lookup->tokens[i] : 5000U + std::uint32_t(i));
    a.observe(o, emitted);
}

// Lookup evidence: passes that emit the proposal's first token resolve its pending predictions as hits.
void warm_lookup(sp::Arbiter& a, const sp::Proposal& p, std::uint64_t& ctx) {
    for (int i = 0; i < 6; ++i) {
        const auto in = input(&p, false, ctx);
        const auto d = a.decide(in);
        observe(a, d, in, d.source == sp::Source::lookup ? d.k + 1U : 1U);
        ctx += 2;
    }
}

// Model passes whose drafts are all accepted but that emit one token each: a realized loss at the frozen prices.
// Runs until the model arm is no longer chosen (the stop-loss bar), at most `limit` passes; returns the passes run.
int losing_model_passes(sp::Arbiter& a, const sp::Proposal* lookup, std::uint64_t& ctx, int limit) {
    int n = 0;
    for (; n < limit; ++n) {
        const auto in = input(lookup, true, ctx);
        const auto d = a.decide(in);
        if (d.source != sp::Source::model) break;
        observe(a, d, in, 1);
        ++ctx;
    }
    return n;
}

void barred_arm_leaves_the_ranking() {
    auto a = sp::Arbiter(frozen(3));
    seed(a);
    const auto p = proposal();
    std::uint64_t ctx = 100;
    warm_lookup(a, p, ctx);
    const int n = losing_model_passes(a, &p, ctx, 200);
    FB_CHECK(n > 0 && n < 200, "the model arm never ran or never stopped (%d passes)", n);
    // The model arm is now barred. The lookup arm beats serial: it is bought, as with no drafter in the input.
    sp::Arbiter without = a;
    const auto in = input(&p, true, ctx);
    const auto d = a.decide(in);
    const auto d0 = without.decide(input(&p, false, ctx));
    FB_CHECK(d.source == sp::Source::lookup && d.source == d0.source && d.k == d0.k && d.rows == d0.rows,
             "barred model: source %d k %u rows %u; without the drafter source %d k %u rows %u", int(d.source), d.k,
             d.rows, int(d0.source), d0.k, d0.rows);
    const auto& s = a.last_arm_scores();
    FB_CHECK(s[std::size_t(sp::Source::lookup)] > s[0], "lookup %.4f does not beat serial %.4f",
             s[std::size_t(sp::Source::lookup)], s[0]);
    FB_CHECK(!(s[std::size_t(sp::Source::model)] >= 0.0), "the barred model arm still has a score (%.4f)",
             s[std::size_t(sp::Source::model)]);
    std::printf("barred model arm: lookup bought after %d model passes (lookup %.3f > serial %.3f)\n", n,
                s[std::size_t(sp::Source::lookup)], s[0]);
}

void barred_arm_admits_no_exploration() {
    // 8 model passes before the drafter is compared: the bar (stop-loss at 4 decayed passes) lands before that count.
    auto a = sp::Arbiter(frozen(8));
    seed(a);
    std::uint64_t ctx = 100;
    const int n = losing_model_passes(a, nullptr, ctx, 200);
    FB_CHECK(n >= 4 && n < 8, "the bar did not land before the minimum model-pass count (%d passes)", n);
    for (int i = 0; i < 20; ++i) {
        const auto in = input(nullptr, true, ctx);
        const auto d = a.decide(in);
        FB_CHECK(d.source != sp::Source::model, "barred arm explored again %d passes after the bar", i);
        observe(a, d, in, 1);
        ++ctx;
    }
    std::printf("barred model arm: no exploration for 20 passes (bar after %d model passes, minimum 8)\n", n);
}

// A black-box drafter's pass wall is live: under frozen rules it cannot buy the arm unless the caller declares it frozen,
// and a barred arm admits no black-box probe either (the probe replaced a winning lookup and cleared the bar).
sp::StepInput black_box_input(const sp::Proposal* lookup, std::uint64_t context, bool frozen_price, bool per_depth) {
    auto in = input(lookup, per_depth, context);
    in.model_max_k = 7;
    in.model_pass_tokens = 3.0;
    in.model_pass_ms = 5.0;  // 0.6 tokens/ms against 0.11 for the plain row: attractive when admitted
    in.model_pass_frozen = frozen_price;
    return in;
}

void barred_arm_admits_no_black_box_probe() {
    auto a = sp::Arbiter(frozen(8));
    seed(a);
    const auto p = proposal();
    std::uint64_t ctx = 100;
    warm_lookup(a, p, ctx);
    const int n = losing_model_passes(a, &p, ctx, 200);
    FB_CHECK(n >= 1 && n < 8, "the bar did not land before the minimum model-pass count (%d passes)", n);
    sp::Arbiter without = a;
    for (int i = 0; i < 20; ++i) {
        for (const bool flag : {false, true}) {
            auto probe = a;
            const auto in = black_box_input(&p, ctx, flag, true);
            const auto d = probe.decide(in);
            const auto d0 = without.decide(input(&p, false, ctx));
            without = a;
            FB_CHECK(d.source != sp::Source::model, "barred arm bought a black-box probe %d passes after the bar (frozen price %d)",
                     i, int(flag));
            FB_CHECK(d.source == d0.source && d.k == d0.k && d.rows == d0.rows,
                     "barred black-box input changed the decision: source %d k %u vs %d k %u", int(d.source), d.k,
                     int(d0.source), d0.k);
        }
        const auto in = input(&p, true, ctx);
        const auto d = a.decide(in);
        observe(a, d, in, d.source == sp::Source::lookup ? d.k + 1U : 1U);
        ctx += 2;
        without = a;
    }
    std::printf("barred model arm: no black-box probe for 20 passes (bar after %d model passes, minimum 8)\n", n);
}

void black_box_needs_a_frozen_price() {
    auto live = sp::Arbiter(frozen(3));
    seed(live);
    auto frozen_price = live;
    sp::Decision d = live.decide(black_box_input(nullptr, 100, false, false));  // no per-depth prices, live wall: not offered
    FB_CHECK(d.source != sp::Source::model, "a live black-box wall bought the model arm under frozen rules");
    d = frozen_price.decide(black_box_input(nullptr, 100, true, false));
    FB_CHECK(d.source == sp::Source::model, "a declared frozen black-box price did not admit the arm (source %d)", int(d.source));
    std::printf("black-box arm: live wall not admitted under frozen rules, declared frozen price admitted\n");
}

void ledger_ages_on_every_emission() {
    auto a = sp::Arbiter(frozen(3));
    seed(a);
    std::uint64_t ctx = 100;
    // Four realized losses (3.9 decayed passes: just below the 4-pass minimum), a long serial interval, one more loss.
    for (int i = 0; i < 4; ++i) {
        const auto in = input(nullptr, true, ctx);
        const auto d = a.decide(in);
        FB_CHECK(d.source == sp::Source::model, "pass %d: the drafter is not chosen", i);
        observe(a, d, in, 1);
        ++ctx;
    }
    sp::Outcome serial;
    serial.rows = 1;
    serial.context = ctx;
    a.observe(serial, std::vector<std::uint32_t>(128, 7U));  // 4 half-lives of the 32-token ledger
    ctx += 128;
    const auto in = input(nullptr, true, ctx);
    const auto d = a.decide(in);
    FB_CHECK(d.source == sp::Source::model, "the drafter is not chosen after the interval");
    observe(a, d, in, 1);
    const auto after = a.decide(input(nullptr, true, ctx + 1));
    FB_CHECK(after.source == sp::Source::model, "stale model losses barred the arm after a 128-token interval");
    std::printf("ledger: four old losses decayed over a 128-token serial interval; no stale bar\n");
}

}  // namespace

int main() {
    barred_arm_leaves_the_ranking();
    barred_arm_admits_no_exploration();
    barred_arm_admits_no_black_box_probe();
    black_box_needs_a_frozen_price();
    ledger_ages_on_every_emission();
    std::printf("spec_frozen_bar PASS\n");
    return 0;
}
