// Shared drafting layer (strix/spec/drafting.hpp): index source/continuation vs brute force (ring wrap included), the
// copy cursor on a copy-with-one-edit stream, the arbiter's choices on synthetic acceptance/costs, and the stats line.
// A simulated greedy target checks that whatever the arbiter picks, the emitted ids equal the serial ids.
#include "strix/spec/drafting.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>

namespace sp = strix::spec;
namespace lk = strix::lookup;

namespace {
#define SP_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

struct Rng {
    std::uint64_t s;
    std::uint32_t next(std::uint32_t n) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return std::uint32_t((s >> 33) % n); }
};

// Every reported match is real: the suffix of length `length` ends at source_end, continuation = the stream after it.
void index_tests() {
    Rng r{11};
    std::uint64_t hits = 0;
    for (const std::uint32_t vocab : {3U, 7U, 40U}) {
        for (const std::uint32_t cap : {64U, 4096U}) {
            auto index = lk::make_ngram_index({cap, 64});
            std::vector<std::uint32_t> s;
            for (int i = 0; i < 3000; ++i) {
                const auto t = r.next(vocab);
                s.push_back(t);
                index->append(t);
                SP_CHECK(index->at(s.size() - 1) == t, "at(last)");
                SP_CHECK(index->oldest() == (s.size() > cap ? s.size() - cap : 0), "oldest");
                const auto m = index->longest({}, 2, 7);
                if (m.length == 0) continue;
                ++hits;
                const std::uint64_t cur = s.size() - 1, e = m.source_end;
                SP_CHECK(e < cur && cur - e == m.distance, "source_end/distance");
                SP_CHECK(e + 1 >= m.length + index->oldest(), "match inside the ring");
                for (std::uint32_t j = 0; j < m.length; ++j) SP_CHECK(s[e - j] == s[cur - j], "suffix at source_end");
                std::vector<std::uint32_t> out(40);
                const auto n = index->continuation(e, {}, out);
                SP_CHECK(n == std::min<std::size_t>(40, cur - e), "continuation length %zu", n);
                for (std::size_t j = 0; j < n; ++j) SP_CHECK(out[j] == s[e + 1 + j], "continuation token");
                for (std::uint32_t j = 0; j < m.available; ++j) SP_CHECK(m.next[j] == out[j], "next == continuation");
            }
            // extra: continuation crosses from the stream into the hypothetical tokens
            const std::vector<std::uint32_t> extra{1, 2, 3};
            std::vector<std::uint32_t> out(8);
            const auto n = index->continuation(index->tokens() - 2, extra, out);
            SP_CHECK(n == 4 && out[0] == s.back() && out[1] == 1 && out[3] == 3, "continuation into extra");
            SP_CHECK(index->continuation(0, {}, out) == (cap >= s.size() ? 8U : 0U), "positions older than the ring");
        }
    }
    SP_CHECK(hits > 1000, "too few matches %llu", static_cast<unsigned long long>(hits));
}

// Speculative loop against a known greedy truth: the arbiter decides, the "target" accepts the prefix equal to truth
// and emits the correction. Returns passes; checks emitted == truth (exactness of the protocol, any decision).
struct LoopResult { std::uint64_t passes = 0, lookup_passes = 0, resumed = 0; std::uint32_t max_k = 0; };
LoopResult run_loop(const std::vector<std::uint32_t>& prompt, const std::vector<std::uint32_t>& truth,
                    sp::ArbiterConfig config, double row_ms, sp::SpecStats* stats = nullptr) {
    sp::ContextProposer proposer;
    proposer.commit(prompt);
    sp::Arbiter arbiter(config);
    for (const auto w : arbiter.config().windows) arbiter.seed_cost(w, 19.4 + row_ms * double(w - 1));
    std::vector<std::uint32_t> emitted;
    LoopResult res;
    while (emitted.size() < truth.size()) {
        const auto proposal = proposer.propose({}, sp::max_proposal);
        sp::StepInput in;
        in.lookup = &proposal;
        in.remaining = std::uint32_t(truth.size() - emitted.size());
        in.context = prompt.size() + emitted.size();
        const auto d = arbiter.decide(in);
        SP_CHECK(d.k + 1 <= d.rows && d.k < in.remaining, "decision exceeds window/budget");
        SP_CHECK(d.source == sp::Source::serial || d.source == sp::Source::lookup, "no model offered");
        std::uint32_t accepted = 0;
        const auto base = emitted.size();
        if (d.source == sp::Source::lookup)
            while (accepted < d.k && proposal.tokens[accepted] == truth[base + accepted]) ++accepted;
        std::vector<std::uint32_t> now(truth.begin() + std::ptrdiff_t(base), truth.begin() + std::ptrdiff_t(base + accepted + 1));
        emitted.insert(emitted.end(), now.begin(), now.end());
        proposer.commit(now);
        sp::Outcome o{d.source, proposal.bucket(), d.k, accepted, d.rows, 19.4 + row_ms * double(d.rows - 1), 0.0, in.context};
        arbiter.observe(o, now);
        if (stats) stats->record(o, std::uint32_t(now.size()));
        ++res.passes;
        res.lookup_passes += d.source == sp::Source::lookup;
        res.resumed += proposal.resumed;
        res.max_k = std::max(res.max_k, d.k);
    }
    SP_CHECK(emitted == truth, "emitted ids differ from the serial truth");
    return res;
}

void copy_tests() {
    Rng r{5};
    std::vector<std::uint32_t> file(3000);
    for (auto& t : file) t = r.next(5000);
    std::vector<std::uint32_t> prompt{7, 8, 9};
    prompt.insert(prompt.end(), file.begin(), file.end());
    prompt.push_back(4999);
    auto truth = file;  // the answer: the whole file with one edit (3 tokens replaced)
    truth[1500] = 1; truth[1501] = 2; truth[1502] = 3;
    sp::ArbiterConfig config;
    config.windows = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64};
    sp::SpecStats stats;
    const auto res = run_loop(prompt, truth, config, 2.7, &stats);
    std::printf("copy 3000 tokens: %llu passes (%.1f tokens/pass), lookup %llu, resumed %llu, max k %u\n",
                static_cast<unsigned long long>(res.passes), 3000.0 / double(res.passes),
                static_cast<unsigned long long>(res.lookup_passes), static_cast<unsigned long long>(res.resumed), res.max_k);
    SP_CHECK(res.passes < 3000 / 20, "copy should commit > 20 tokens/pass, got %llu passes", static_cast<unsigned long long>(res.passes));
    SP_CHECK(res.max_k >= 31, "long windows unused (max k %u)", res.max_k);
    // Cheap rows (0.5 ms/row): the 64-row window wins.
    const auto resc = run_loop(prompt, truth, config, 0.5);
    std::printf("copy, 0.5 ms/row: %llu passes, max k %u\n", static_cast<unsigned long long>(resc.passes), resc.max_k);
    SP_CHECK(resc.max_k == 63 && resc.passes <= res.passes, "cheap rows: max k %u", resc.max_k);
    const auto line = stats.line();
    SP_CHECK(line.rfind("strix-spec-stats {\"passes\":", 0) == 0, "stats line prefix: %s", line.c_str());
    SP_CHECK(line.find("\"emitted\":3000") != std::string::npos, "stats emitted");
    SP_CHECK(line.find("\"64\":{\"passes\":") != std::string::npos, "rows histogram has the 64-row window");

    // Eight-row engines: same stream, capped windows.
    const auto res8 = run_loop(prompt, truth, sp::ArbiterConfig{}, 2.7);
    SP_CHECK(res8.max_k == 7 && res8.passes < 3000 / 6, "8-row copy: max k %u passes %llu", res8.max_k,
             static_cast<unsigned long long>(res8.passes));

    // New text (no repeats in the answer or the prompt): lookup never wins, serial throughout after the priors.
    std::vector<std::uint32_t> fresh(2000);
    for (auto& t : fresh) t = 10000 + r.next(50000);
    const auto resn = run_loop(prompt, fresh, config, 2.7);
    SP_CHECK(resn.lookup_passes < 20, "lookup used on new text: %llu passes", static_cast<unsigned long long>(resn.lookup_passes));

    // Expensive rows (12 ms/row on a 19.4 ms base): windows must stay shorter than with cheap rows.
    const auto rese = run_loop(prompt, truth, config, 12.0);
    SP_CHECK(rese.passes > res.passes, "expensive rows should cost passes (%llu vs %llu)",
             static_cast<unsigned long long>(rese.passes), static_cast<unsigned long long>(res.passes));
}

// Retreat over the evidence horizon (ArbiterConfig::retreat_horizon): at a 0.64 first-row rate (CUDA dense 27B Japanese
// prose) a streak of 6 first-row misses is routine over 256 passes and must not bar the drafter for probe_every serial
// passes; the per-pass form (horizon 0) bars it at 5. A streak of 16 still bars it in both forms.
void retreat_horizon_tests() {
    for (const double horizon : {0.0, 256.0}) {
        sp::ArbiterConfig config;
        config.windows = {1, 2, 3, 4, 5, 6, 7, 8};
        config.retreat_horizon = horizon;
        sp::Arbiter a(config);
        for (std::uint32_t w = 1; w <= 8; ++w) a.seed_cost(w, 13.30 + 0.113 * double(w - 1));
        const std::vector<double> draft{0.634, 1.268, 1.902, 2.536, 3.170, 3.804, 4.438};
        sp::StepInput in;
        in.model_max_k = 7;
        in.model_draft_ms = draft;
        const std::vector<std::uint32_t> tokens(8, 1);
        const auto pass = [&](std::uint32_t accepted) {
            a.observe({sp::Source::model, 0, 3, accepted, 4, 13.6, 1.9},
                      std::span<const std::uint32_t>(tokens.data(), accepted + 1));
        };
        for (std::uint32_t i = 0; i < 300; ++i) {  // first-row rate 0.64, then 0.5 per depth
            (void)a.decide(in);
            pass(i * 37 % 100 < 64 ? (i % 2 ? 1U : 2U) : 0U);
        }
        for (int i = 0; i < 6; ++i) { (void)a.decide(in); pass(0); }
        auto d = a.decide(in);
        if (horizon > 0.0) SP_CHECK(d.source == sp::Source::model, "horizon %.0f: 6 routine misses barred the drafter", horizon);
        else SP_CHECK(d.source == sp::Source::serial, "per-pass form: 6 misses at r1 0.64 bar the drafter (the old rule)");
        for (int i = 0; i < 16; ++i) { (void)a.decide(in); pass(0); }
        d = a.decide(in);
        SP_CHECK(d.source == sp::Source::serial, "horizon %.0f: 22 misses in a row must bar the drafter", horizon);
    }
}

void arbiter_tests() {
    sp::ArbiterConfig config;
    config.windows = {1, 2, 4, 8};
    sp::Arbiter a(config);
    for (const std::uint32_t w : {1U, 2U, 4U, 8U}) a.seed_cost(w, 20.0 + 3.0 * double(w - 1));
    SP_CHECK(a.verify_ms(3) > a.verify_ms(2) && a.verify_ms(3) < a.verify_ms(4), "interpolation");
    SP_CHECK(a.verify_ms(6) > a.verify_ms(4) && a.verify_ms(6) < a.verify_ms(8), "interpolation 6");
    // Model drafter p = 0.8 at every depth, 0.5 ms/draft token: the arbiter picks the model over serial.
    const std::vector<double> p(7, 0.8), draft{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    a.seed_model_survival(p);
    sp::StepInput in;
    in.model_max_k = 7;
    in.model_draft_ms = draft;
    auto d = a.decide(in);
    SP_CHECK(d.source == sp::Source::model && d.k >= 3 && d.rows >= d.k + 1, "model chosen (k %u)", d.k);
    // Budget: two tokens left -> at most one draft.
    in.remaining = 2;
    d = a.decide(in);
    SP_CHECK(d.k <= 1, "budget");
    in.remaining = 1;
    d = a.decide(in);
    SP_CHECK(d.source == sp::Source::serial && d.k == 0, "last token is serial");
    // Repeated first-row misses: the model's survival collapses and serial takes over.
    in.remaining = ~0U;
    for (int i = 0; i < 40; ++i) a.observe({sp::Source::model, 0, 7, 0, 8, 41.0, 3.5}, {});
    d = a.decide(in);
    SP_CHECK(d.source == sp::Source::serial || d.probe, "retreat after misses");
    // Probe: after probe_every passes without the model it is tried once.
    bool probed = false;
    for (int i = 0; i < 40 && !probed; ++i) {
        d = a.decide(in);
        probed = d.probe && d.source == sp::Source::model;
        a.observe({d.source, 0, d.k, 0, d.rows, 20.0, 0.0}, {});
    }
    SP_CHECK(probed, "model re-probed");
    SP_CHECK(d.k == 7, "probe runs the deepest legal k (%u)", d.k);
    // Per-step legal rows: max_rows = 3 keeps every window <= 3; zero budget / zero rows = nothing to do.
    in.max_rows = 3;
    for (int i = 0; i < 8; ++i) {
        d = a.decide(in);
        SP_CHECK(d.rows >= 1 && d.rows <= 3 && d.k + 1 <= d.rows, "max_rows (rows %u k %u)", d.rows, d.k);
    }
    in.max_rows = ~0U;
    in.remaining = 0;
    SP_CHECK(a.decide(in).rows == 0, "zero budget");
    in.remaining = ~0U;
    // A drafter must come with its measured cost.
    bool threw = false;
    sp::StepInput no_cost;
    no_cost.model_max_k = 3;
    try { (void)a.decide(no_cost); } catch (const std::invalid_argument&) { threw = true; }
    SP_CHECK(threw, "drafter without draft cost");
    // Validation.
    threw = false;
    try { sp::Arbiter bad(sp::ArbiterConfig{{2, 4}}); } catch (const std::invalid_argument&) { threw = true; }
    SP_CHECK(threw, "windows must start at 1");

    // Costs: monotone in rows even when measurements invert; separate per context bin.
    sp::Arbiter c(sp::ArbiterConfig{{1, 2, 4, 8, 16}});
    c.seed_cost(1, 20.0);
    c.seed_cost(4, 30.0);
    c.seed_cost(8, 28.0);  // inverted measurement
    SP_CHECK(c.verify_ms(8) >= c.verify_ms(4) && c.verify_ms(6) >= c.verify_ms(4), "monotone cost");
    SP_CHECK(c.verify_ms(16) >= c.verify_ms(8), "extrapolation stays monotone");
    c.observe({sp::Source::serial, 0, 0, 0, 1, 80.0, 0.0, 100000}, {});  // a 100K-context wall
    SP_CHECK(c.verify_ms(1, 100000) > c.verify_ms(1, 1000), "context bins are separate");

    // Lookup needs evidence first: a proposal is not run before min_lookup_evidence depth-1 observations.
    sp::Arbiter l(sp::ArbiterConfig{{1, 2, 4, 8}});
    for (const std::uint32_t w : {1U, 2U, 4U, 8U}) l.seed_cost(w, 20.0 + 1.0 * double(w - 1));
    sp::Proposal prop;
    prop.tokens = {5, 6, 7, 8};
    prop.match_length = 16;
    sp::StepInput lin;
    lin.lookup = &prop;
    SP_CHECK(l.decide(lin).source == sp::Source::serial, "lookup before evidence");

    // Black-box drafter (picks its own window): scored on its measured rate, no per-depth cost table needed.
    sp::Arbiter bb(sp::ArbiterConfig{});
    bb.seed_cost(1, 18.0);
    sp::StepInput bin;
    bin.model_max_k = 7;
    bin.model_pass_tokens = 2.6;
    bin.model_pass_ms = 24.0;  // 0.108 tok/ms > serial 1/18 = 0.056
    auto bd = bb.decide(bin);
    SP_CHECK(bd.source == sp::Source::model && bd.k == 7, "black-box drafter chosen (k %u)", bd.k);
    bin.model_pass_tokens = 1.1;  // 0.046 tok/ms < serial; after the 3 forced measurements serial wins
    for (int i = 0; i < 3; ++i) bb.observe({sp::Source::model, 0, 7, 1, 8, 20.0, 4.0}, {});
    bd = bb.decide(bin);
    SP_CHECK(bd.source == sp::Source::serial, "weak black-box drafter loses to serial");

    // Black-box drafter with ~50 % first-row acceptance whose rate beats serial: no retreat, it keeps running.
    sp::Arbiter bb2(sp::ArbiterConfig{});
    bb2.seed_cost(1, 18.0);
    sp::StepInput bin2;
    bin2.model_max_k = 7;
    bin2.model_pass_tokens = 1.7;
    bin2.model_pass_ms = 24.0;  // 0.071 tok/ms > serial 0.056
    std::uint32_t model_runs = 0;
    for (int i = 0; i < 200; ++i) {
        const auto d2 = bb2.decide(bin2);
        model_runs += d2.source == sp::Source::model;
        const std::uint32_t acc = (i % 4 < 2) ? 0U : 2U;  // two misses in a row every 4 passes
        bb2.observe({d2.source, 0, d2.k, d2.source == sp::Source::model ? acc : 0U, d2.rows, 20.0, 4.0}, {});
    }
    SP_CHECK(model_runs == 200, "black-box drafter barred by the retreat rule (%u of 200 passes)", model_runs);

    // Stats: lookup time quantiles.
    sp::SpecStats st;
    for (int i = 0; i < 99; ++i) st.lookup_time(3.0);
    st.lookup_time(500.0);
    SP_CHECK(st.lookup_us_quantile(0.5) <= 4.0 && st.lookup_us_quantile(0.999) >= 256.0, "lookup quantiles");
    SP_CHECK(st.json().find("\"lookup_us\":{\"count\":100") != std::string::npos, "lookup_us json");
}
// One chat copying its prompt, priced like the DeepSeek pair (verify 28.6 ms + 5.5 ms per extra row) against a black-box
// learned drafter with the same committed tokens per pass as a perfect copy window of `model_rows` and a 9.6 ms draft:
// the copy proposal (same survival, no draft cost) must win, and its expected tokens must not be shrunk per depth.
struct CopyRun { std::uint64_t passes = 0, lookup_passes = 0; double last_expected = 0.0; std::uint32_t last_k = 0; };
CopyRun ds_copy_run(std::uint32_t max_rows, std::uint32_t model_rows, std::uint32_t lookup_len) {
    std::vector<std::uint32_t> prompt(400);
    std::uint32_t x = 12345;
    for (auto& t : prompt) { x = x * 1103515245U + 12345U; t = 100 + (x >> 8) % 50000; }
    sp::ArbiterConfig config;
    config.windows.clear();
    for (std::uint32_t w = 1; w <= max_rows; ++w) config.windows.push_back(w);
    sp::Arbiter a(config);
    for (std::uint32_t w = 1; w <= max_rows; ++w) a.seed_cost(w, 28.6 + 5.5 * double(w - 1));
    sp::ContextProposer proposer;
    proposer.commit(prompt);
    sp::Proposal proposal;
    CopyRun run;
    std::size_t at = 0;  // the output copies the prompt
    while (at + 1 < prompt.size()) {
        proposer.propose(proposal, {}, lookup_len);
        sp::StepInput in;
        in.lookup = &proposal;
        in.model_max_k = model_rows - 1;
        in.model_pass_tokens = double(model_rows);
        in.model_pass_ms = 28.6 + 5.5 * double(model_rows - 1) + 9.6;
        in.remaining = std::uint32_t(prompt.size() - at);
        const auto d = a.decide(in);
        std::uint32_t accepted = 0;
        if (d.source == sp::Source::lookup)
            while (accepted < d.k && proposal.tokens[accepted] == prompt[at + accepted]) ++accepted;
        else if (d.source == sp::Source::model)
            accepted = std::min<std::uint32_t>(d.k, std::uint32_t(prompt.size() - at - 1));
        const std::uint32_t kept = std::min<std::uint32_t>(accepted + 1, std::uint32_t(prompt.size() - at));
        const std::span<const std::uint32_t> emitted(prompt.data() + at, kept);
        sp::Outcome o;
        o.source = d.source;
        o.bucket = std::min(proposal.bucket(), sp::lookup_buckets - 1);
        o.proposed = d.k;
        o.accepted = accepted;
        o.rows = d.rows;
        o.verify_ms = 28.6 + 5.5 * double(d.rows - 1);
        o.draft_ms = d.source == sp::Source::model ? 9.6 : 0.0;
        a.observe(o, emitted);
        proposer.commit(emitted);
        at += kept;
        ++run.passes;
        if (d.source == sp::Source::lookup) {
            ++run.lookup_passes;
            if (in.remaining > lookup_len + 1) run.last_expected = d.expected_tokens, run.last_k = d.k;  // not budget-bound
        }
    }
    return run;
}

void ds_arbiter_tests() {
    // 6 rows, the learned drafter as good as a perfect copy window: the copy (no draft) wins almost every pass
    const auto r6 = ds_copy_run(6, 6, 5);
    SP_CHECK(r6.lookup_passes * 10 >= r6.passes * 8, "copy lost to an equal drafter: %llu of %llu passes",
             static_cast<unsigned long long>(r6.lookup_passes), static_cast<unsigned long long>(r6.passes));
    SP_CHECK(r6.last_k == 5 && r6.last_expected >= 5.5, "6-row copy window predicts %.2f tokens (k %u)", r6.last_expected, r6.last_k);
    // 8-row copy windows beside the 6-row drafter: copy passes are deeper, fewer passes in total
    const auto r8 = ds_copy_run(8, 6, 7);
    SP_CHECK(r8.last_k == 7 && r8.passes < r6.passes, "8-row copy: k %u, %llu vs %llu passes", r8.last_k,
             static_cast<unsigned long long>(r8.passes), static_cast<unsigned long long>(r6.passes));
    // a perfect 16-candidate bucket predicts >= 15 committed tokens (cumulative survival bounded once)
    const auto r16 = ds_copy_run(16, 6, 15);
    SP_CHECK(r16.last_k == 15 && r16.last_expected >= 15.0, "16-row copy window predicts %.2f tokens (k %u)", r16.last_expected, r16.last_k);
}

// Frontier probe (ArbiterConfig::frontier_every). Replays the measured prose shape of the CUDA dense 27B MTP chain
// (q27 D2 receipts: verify ~13.2 ms + ~0.25 ms/row, 0.85 ms per chain step, conditional acceptance 0.759 at depth 1,
// 0.667 at depth 2, then 0.6) with the first probe missing at row 1, as in the receipt that stayed at k = 1 for 290 passes.
struct FrontierRun {
    std::uint32_t max_k_after_first = 0;
    double tokens = 0, ms = 0;
    std::uint64_t probes = 0, model = 0, k1 = 0, k_sum = 0;
    [[nodiscard]] double tok_s() const { return 1000.0 * tokens / ms; }
    [[nodiscard]] double mean_k() const { return model ? double(k_sum) / double(model) : 0.0; }
};
// One content shape: verify(rows) = v1 + per_row * (rows - 1), draft = per_call ms per drafted position (k MTP head calls),
// conditional acceptance rate[j] at depth j + 1, chain cap, and whether the first model pass misses at row 1.
struct Shape { double v1, per_row, per_call; std::vector<double> rate; std::uint32_t cap = 7; bool first_miss = false; };
// The controller forms: f0 = before cc_decode (no probe, flat prior), f8 = cc_decode 854325dbb (unpriced probe every 8,
// flat prior), now = the defaults (continuation prior, probe charged its cost).
sp::ArbiterConfig form(int f) {
    sp::ArbiterConfig c;
    c.windows.clear();
    for (std::uint32_t w = 1; w <= 32; ++w) c.windows.push_back(w);
    if (f == 0 || f == 8) { c.model_continuation = false; c.frontier_loss = 1.0; c.frontier_every = std::uint32_t(f); }
    return c;
}

FrontierRun replay(const sp::ArbiterConfig& config, const Shape& sh, std::uint64_t seed, int passes = 600) {
    sp::Arbiter a(config);
    const auto verify = [&](std::uint32_t rows) { return sh.v1 + sh.per_row * double(rows - 1); };
    for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U, 32U}) a.seed_cost(r, verify(r));
    std::vector<double> draft;
    for (std::uint32_t k = 1; k <= sh.cap; ++k) draft.push_back(sh.per_call * double(k));
    Rng rng{seed};
    FrontierRun run;
    std::vector<std::uint32_t> emitted;
    for (int pass = 0; pass < passes; ++pass) {
        sp::StepInput in;
        in.model_max_k = sh.cap;
        in.model_draft_ms = draft;
        const auto d = a.decide(in);
        SP_CHECK(d.rows >= d.k + 1 && d.k <= sh.cap, "illegal decision rows %u k %u", d.rows, d.k);
        std::uint32_t acc = 0;
        if (d.source == sp::Source::model) {
            if (pass > 0 || !sh.first_miss)  // first_miss: the first probe misses at row 1 (the stuck receipt's start)
                while (acc < d.k && double(rng.next(1000)) / 1000.0 < sh.rate[std::min<std::size_t>(acc, sh.rate.size() - 1)]) ++acc;
            if (pass > 0) {
                run.max_k_after_first = std::max(run.max_k_after_first, d.k);
                ++run.model;
                run.k1 += d.k == 1;
                run.k_sum += d.k;
            }
            run.probes += d.probe;
        }
        const double ms = verify(d.rows) + (d.source == sp::Source::model ? draft[d.k - 1] : 0.0);
        emitted.assign(acc + 1, 1U);
        a.observe({d.source, 0, d.k, acc, d.rows, verify(d.rows), d.source == sp::Source::model ? draft[d.k - 1] : 0.0}, emitted);
        run.tokens += double(acc + 1);
        run.ms += ms;
    }
    return run;
}

// The q27 D2 prose shape that stayed at k = 1 (first probe misses at row 1).
const Shape stuck_prose{13.2, 0.25, 0.85, {0.759, 0.667, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5}, 7, true};
FrontierRun frontier_run(std::uint32_t frontier_every, std::uint64_t seed) { return replay(form(int(frontier_every)), stuck_prose, seed); }

// Shapes of the cd1-cd5 receipts (RTX PRO 6000, 1,200 tokens): verify by rows from the step lines, 0.83 ms per drafted
// position, conditional acceptance by depth from the stats depth_offered / depth_accepted (f8 runs).
const Shape ja_f8{13.65, 0.05, 0.83, {0.638, 0.469, 0.475, 0.338, 0.5}};
const Shape en_f8{13.45, 0.08, 0.83, {0.756, 0.578, 0.497, 0.463, 0.47}};
const Shape code_f8{13.70, 0.06, 0.83, {0.941, 0.918, 0.900, 0.855, 0.891, 0.909, 0.857, 0.87}};
const Shape think_f8{13.85, 0.05, 0.83, {0.844, 0.801, 0.825, 0.829, 0.826, 0.780, 0.782, 0.79}};

// Pricing: the model reproduces the measured f8 tok/s within 10 % (ja 120.9, en 134.6, code 283.4, think 209.4: replay E is
// chain-only, so code/think replays sit above the receipts by their serial/lookup passes), and the new form never loses
// to f8 by more than noise; Japanese must come back to >= the f0 receipt shape (k = 3 instead of 4-5).
void pricing_tests() {
    struct Case { const char* name; const Shape* sh; double measured_f8; double tol; };
    const Case cases[] = {{"ja", &ja_f8, 120.9, 0.10}, {"en", &en_f8, 134.6, 0.10}, {"code", &code_f8, 283.4, 0.12},
                          {"think", &think_f8, 209.4, 0.12}};
    for (const auto& c : cases) {
        double f8 = 0, now = 0, k8 = 0, kn = 0;
        for (const std::uint64_t seed : {1ULL, 7ULL, 42ULL}) {
            const auto a = replay(form(8), *c.sh, seed, 1200), b = replay(form(-1), *c.sh, seed, 1200);
            f8 += a.tok_s() / 3.0; now += b.tok_s() / 3.0; k8 += a.mean_k() / 3.0; kn += b.mean_k() / 3.0;
        }
        std::printf("pricing %s: f8 replay %.1f tok/s (k %.2f; measured %.1f) | now %.1f tok/s (k %.2f) %+.1f %%\n", c.name, f8, k8,
                    c.measured_f8, now, kn, 100.0 * (now / f8 - 1.0));
        SP_CHECK(std::abs(f8 / c.measured_f8 - 1.0) <= c.tol, "%s: f8 replay %.1f not within %.0f %% of the receipt %.1f", c.name, f8,
                 100.0 * c.tol, c.measured_f8);
        SP_CHECK(now >= 0.99 * f8, "%s: new pricing %.1f below f8 %.1f", c.name, now, f8);
    }
    double ja_now = 0, ja_f8_rate = 0, ja_k = 0;
    for (const std::uint64_t seed : {1ULL, 7ULL, 42ULL}) {
        const auto b = replay(form(-1), ja_f8, seed, 1200);
        ja_now += b.tok_s() / 3.0; ja_k += b.mean_k() / 3.0;
        ja_f8_rate += replay(form(8), ja_f8, seed, 1200).tok_s() / 3.0;
    }
    SP_CHECK(ja_now >= 1.03 * ja_f8_rate && ja_k < 3.6, "ja: now %.1f tok/s k %.2f vs f8 %.1f (expected >= +3 %%, k ~3)", ja_now, ja_k,
             ja_f8_rate);
    // Deeper caps are priced, not taken blindly: code with a cheap draft (0.45 ms, subset head) and cap 15 runs deeper than 7
    // and gains; with today's 0.83 ms the cap-15 arm must not lose against cap 7.
    for (const double per_call : {0.45, 0.83}) {
        Shape deep = code_f8, capped = code_f8;
        deep.per_call = capped.per_call = per_call;
        deep.cap = 15;
        double r15 = 0, r7 = 0, k15 = 0;
        for (const std::uint64_t seed : {1ULL, 7ULL, 42ULL}) {
            const auto a = replay(form(-1), deep, seed, 1200);
            r15 += a.tok_s() / 3.0; k15 += a.mean_k() / 3.0;
            r7 += replay(form(-1), capped, seed, 1200).tok_s() / 3.0;
        }
        std::printf("cap code %.2f ms/call: cap 7 %.1f tok/s | cap 15 %.1f tok/s (k %.2f)\n", per_call, r7, r15, k15);
        SP_CHECK(r15 >= 0.99 * r7, "cap 15 loses: %.1f vs %.1f", r15, r7);
        if (per_call < 0.5) SP_CHECK(k15 > 7.5 && r15 > 1.03 * r7, "cheap draft: cap 15 k %.2f, %.1f vs %.1f", k15, r15, r7);
    }
}

// Context pricing (ArbiterConfig::row_kv_ms_per_kpos). Receipts: RTX PRO, CUDA dense 27B, fixed arm sub15d (cap 15),
// seeded V(r) = 13.30 + 0.113 (r - 1), draft 0.494 ms per position, at every context. Walls at length L fit on the 28K
// code / think / prose cells: V(r, L) = 11.66 + 0.113 (r - 1) + 0.0699 (L / 1024) r (+ 0.08 host); it predicts the 8K code
// cells within 0.4 %. Survival by depth = depth_accepted / depth_offered of each run; measured draft ms per position.
struct CtxCase { const char* name; std::vector<double> rate; double ctx, draft, measured, measured_k; };
// Device truth for the replays, fitted on the per-row verify walls of the RTX PRO receipts (stats "rows", Qwen 27B dense
// NVFP4, 8K and 28K, 14 runs): a flat 12.2 ms, a step of 1.1 ms from 9 rows and 4.0 ms from 17 rows (row tiles), and the
// K/V walk kappa L/1,024 per row (0.0686). shared = the marginal row's share of the K/V walk (1 = every row re-reads it,
// 0.45 = rows share one walk, attention x0.52 at 8 rows), + 0.08 ms host.
double device_wall(std::uint32_t rows, double ctx, double shared = 1.0, double kappa = 0.0686) {
    const double r = double(rows);
    return 12.2 + (rows >= 9 ? 1.1 : 0.0) + (rows >= 17 ? 4.0 : 0.0) + kappa * (ctx / 1024.0) * (1.0 + shared * (r - 1.0)) + 0.08;
}
// mode: spec::set_context_cost value ("off" = the previous context-blind seeds, "online" = the default fit, "0.0699" = fixed).
FrontierRun replay_ctx(const CtxCase& c, const std::string& mode, std::uint64_t seed, double shared = 1.0, int passes = 600) {
    auto config = form(-1);
    config.learn_costs = false;  // BlockConfig::fixed_costs
    sp::set_context_cost(config, mode);
    sp::Arbiter a(config);
    for (std::uint32_t r = 1; r <= 32; ++r) a.seed_cost(r, 13.30 + 0.113 * double(r - 1));
    std::vector<double> seeded_draft;
    for (std::uint32_t k = 1; k <= 15; ++k) seeded_draft.push_back(0.494 * double(k));
    const auto wall = [&](std::uint32_t rows) { return device_wall(rows, c.ctx, shared); };
    Rng rng{seed};
    FrontierRun run;
    std::vector<std::uint32_t> emitted;
    for (int pass = 0; pass < passes; ++pass) {
        sp::StepInput in;
        in.model_max_k = 15;
        in.model_draft_ms = seeded_draft;
        in.context = std::uint64_t(c.ctx);
        const auto d = a.decide(in);
        std::uint32_t acc = 0;
        if (d.source == sp::Source::model) {
            while (acc < d.k && double(rng.next(1000)) / 1000.0 < c.rate[std::min<std::size_t>(acc, c.rate.size() - 1)]) ++acc;
            ++run.model;
            run.k_sum += d.k;
        }
        const double draft = d.source == sp::Source::model ? c.draft * double(d.k) : 0.0;
        emitted.assign(acc + 1, 1U);
        a.observe({d.source, 0, d.k, acc, d.rows, wall(d.rows), draft, in.context}, emitted);
        run.tokens += double(acc + 1);
        run.ms += wall(d.rows) + draft;
    }
    return run;
}

// The long-context receipts (survival by depth, context, draft ms per position, tok/s and k of the off arm).
const CtxCase ctx_cases[] = {
    {"code 28K", {0.818, 0.728, 0.814, 0.8125, 0.784, 0.786, 0.727, 0.75, 0.7}, 28387, 0.583, 120.6, 7.43},
    {"think 28K", {0.831, 0.797, 0.764, 0.854, 0.829, 0.966, 0.741, 0.789, 0.727, 0.667, 0.7}, 27911, 0.591, 129.3, 8.32},
    {"prose 28K", {0.724, 0.556, 0.615, 0.48, 0.5}, 27781, 0.673, 100.2, 3.91},
    {"code 8K", {0.782, 0.779, 0.789, 0.747, 0.727, 0.846, 0.583, 0.583, 0.55}, 8300, 0.456, 188.8, 6.95},
};

void context_pricing_tests() {
    // The term: seeded model only, exact formula, nothing at context 0, ignored by learned costs.
    {
        sp::ArbiterConfig config = form(-1);
        config.learn_costs = false;
        config.row_kv_ms_per_kpos = 0.0699;
        sp::Arbiter a(config);
        auto plain_cfg = config;
        plain_cfg.row_kv_ms_per_kpos = 0.0;
        sp::Arbiter plain(plain_cfg);
        auto learned_cfg = config;
        learned_cfg.learn_costs = true;
        sp::Arbiter learned(learned_cfg);
        for (std::uint32_t r = 1; r <= 32; ++r) {
            a.seed_cost(r, 13.30 + 0.113 * double(r - 1));
            plain.seed_cost(r, 13.30 + 0.113 * double(r - 1));
            learned.seed_cost(r, 13.30 + 0.113 * double(r - 1));
        }
        for (const std::uint32_t r : {1U, 4U, 9U, 16U}) {
            SP_CHECK(a.verify_ms(r, 0) == plain.verify_ms(r, 0), "context 0 adds nothing (rows %u)", r);
            const double want = plain.verify_ms(r, 28672) + 0.0699 * double(r) * 28.0;
            SP_CHECK(std::abs(a.verify_ms(r, 28672) - want) < 1e-9, "term rows %u: %.6f vs %.6f", r, a.verify_ms(r, 28672), want);
            SP_CHECK(learned.verify_ms(r, 28672) == plain.verify_ms(r, 28672), "learned costs ignore the term (rows %u)", r);
        }
    }
    // Replays: kv = 0 reproduces the receipts (tok/s and k); kv = 0.0699 drafts shallower at 28K and gains; at 8K it
    // never loses; at a short context the decisions are identical (same tokens and walls).
    // off = the previous form (reproduces the receipts); online = the default fit; fixed 0.0699 = the measured A/B arm.
    // Each also on a short context (1,200 positions, the chart prompts) and with a shared K/V walk (rows x0.52 at 8).
    for (const auto& c : ctx_cases) {
        CtxCase shortc = c;
        shortc.ctx = 1200;
        struct Arm { double tok = 0, k = 0; };
        const auto run3 = [](const CtxCase& cc, const std::string& mode, double shared) {
            Arm m;
            for (const std::uint64_t seed : {1ULL, 7ULL, 42ULL}) {
                const auto r = replay_ctx(cc, mode, seed, shared);
                m.tok += r.tok_s() / 3.0;
                m.k += r.mean_k() / 3.0;
            }
            return m;
        };
        const Arm off = run3(c, "off", 1.0), on = run3(c, "online", 1.0), fix = run3(c, "0.0699", 1.0);
        const Arm s_off = run3(shortc, "off", 1.0), s_on = run3(shortc, "online", 1.0);
        const Arm w_fix = run3(c, "0.0699", 0.45), w_on = run3(c, "online", 0.45);
        std::printf("context pricing %s: off %.1f tok/s k %.2f (receipt %.1f k %.2f) | online %.1f k %.2f %+.1f %% | fixed 0.0699 "
                    "%.1f k %.2f %+.1f %% | short off %.1f online %.1f %+.2f %% | shared walk: fixed %.1f k %.2f, online %.1f k %.2f "
                    "%+.1f %%\n",
                    c.name, off.tok, off.k, c.measured, c.measured_k, on.tok, on.k, 100.0 * (on.tok / off.tok - 1.0), fix.tok, fix.k,
                    100.0 * (fix.tok / off.tok - 1.0), s_off.tok, s_on.tok, 100.0 * (s_on.tok / s_off.tok - 1.0), w_fix.tok, w_fix.k,
                    w_on.tok, w_on.k, 100.0 * (w_on.tok / w_fix.tok - 1.0));
        SP_CHECK(std::abs(off.tok / c.measured - 1.0) <= 0.10, "%s: off replay %.1f not within 10 %% of the receipt %.1f", c.name,
                 off.tok, c.measured);
        SP_CHECK(on.tok >= 0.995 * off.tok, "%s: online loses to off (%.1f vs %.1f)", c.name, on.tok, off.tok);
        SP_CHECK(on.tok >= 0.99 * fix.tok, "%s: online %.1f below the fixed arm %.1f", c.name, on.tok, fix.tok);
        if (c.ctx > 20000) SP_CHECK(on.tok >= 1.03 * off.tok && on.k < off.k, "%s: expected a gain at 28K (%.1f vs %.1f, k %.2f vs %.2f)",
                                    c.name, on.tok, off.tok, on.k, off.k);
        SP_CHECK(s_on.tok >= 0.995 * s_off.tok, "%s short: online loses (%.1f vs %.1f)", c.name, s_on.tok, s_off.tok);
        // Shared walk: the marginal row is cheaper, the fit follows by itself (deeper than per-row, not below the lab constant).
        SP_CHECK(w_on.k > on.k && w_on.tok >= 0.995 * w_fix.tok, "%s shared walk: online k %.2f (per-row %.2f), %.1f vs fixed %.1f",
                 c.name, w_on.k, on.k, w_on.tok, w_fix.tok);
    }
}

// The fit on the measured per-row walls of the receipts (stats "rows": rows, mean verify ms, passes; one run each).
struct RowWalls { const char* name; double ctx; std::vector<std::array<double, 3>> rows; };
const RowWalls receipt_walls[] = {
    {"c28b", 28387, {{4, 20.11, 1}, {5, 21.51, 2}, {6, 23.90, 3}, {8, 27.40, 42}, {9, 30.42, 44}, {10, 32.24, 6}, {11, 33.93, 1}}},
    {"c28k", 28387, {{1, 13.99, 1}, {2, 15.93, 1}, {3, 18.20, 1}, {4, 19.83, 2}, {5, 21.65, 93}, {6, 23.96, 12}, {7, 25.59, 4}}},
    {"e28b", 27781, {{2, 15.85, 1}, {4, 19.91, 41}, {5, 21.68, 100}, {6, 23.32, 14}, {7, 25.57, 5}, {8, 27.27, 2}}},
    {"t28b", 27911, {{1, 14.02, 1}, {4, 19.92, 2}, {7, 25.66, 3}, {8, 27.39, 5}, {9, 30.25, 30}, {10, 32.51, 32}, {11, 34.18, 6}}},
    {"n8b", 8150, {{1, 12.42, 1}, {5, 14.48, 2}, {6, 15.38, 4}, {7, 16.27, 1}, {8, 16.43, 10}, {9, 18.04, 50}, {10, 18.53, 2}}},
};

void cost_fit_tests() {
    // The switch: one parser and one printed name for every binder.
    {
        sp::ArbiterConfig c;
        SP_CHECK(c.cost_fit && c.row_kv_ms_per_kpos == 0.0, "default = online");
        c.learn_costs = false;
        SP_CHECK(sp::context_cost_name(c) == "online", "name online");
        sp::set_context_cost(c, "off");
        SP_CHECK(!c.cost_fit && sp::context_cost_name(c) == "off", "off");
        sp::set_context_cost(c, "0.0699");
        SP_CHECK(!c.cost_fit && c.row_kv_ms_per_kpos == 0.0699 && sp::context_cost_name(c) == "fixed 0.0699", "fixed");
        sp::set_context_cost(c, "0");
        SP_CHECK(!c.cost_fit && c.row_kv_ms_per_kpos == 0.0, "0 = off (the previous flag value)");
        sp::set_context_cost(c, "online");
        SP_CHECK(c.cost_fit && c.row_kv_ms_per_kpos == 0.0, "back to online");
        for (const char* bad : {"", "on", "-1", "11", "0.07x", "nan"}) {
            bool threw = false;
            try { sp::set_context_cost(c, bad); } catch (const std::invalid_argument&) { threw = true; }
            SP_CHECK(threw, "--cost-kv '%s' must be rejected", bad);
        }
        c.learn_costs = true;
        SP_CHECK(sp::context_cost_name(c) == "learned", "learned costs: walls per window");
    }
    const auto seeded = [](const std::string& mode, bool learn = false) {
        auto config = form(-1);
        config.learn_costs = learn;
        sp::set_context_cost(config, mode);
        sp::Arbiter a(config);
        for (std::uint32_t r = 1; r <= 32; ++r) a.seed_cost(r, 13.30 + 0.113 * double(r - 1));
        return a;
    };
    const auto feed = [](sp::Arbiter& a, const RowWalls& w) {  // passes interleaved across rows (a run's order is unknown)
        std::vector<std::array<double, 3>> left = w.rows;
        for (bool any = true; any;) {
            any = false;
            for (auto& r : left)
                if (r[2] > 0.0) {
                    a.observe({sp::Source::model, 0, std::uint32_t(r[0]) - 1U, 0, std::uint32_t(r[0]), r[1], 0.0, std::uint64_t(w.ctx)},
                              std::vector<std::uint32_t>(1, 1U));
                    r[2] -= 1.0;
                    any = true;
                }
        }
    };
    // Recovery: each receipt's own walls predict its per-row walls (rows with >= 4 passes) within 6 % and kappa lands in the
    // flat band of the decision (0.05-0.10 ms per row per 1K positions at 28K); before any pass = the seeds.
    for (const auto& w : receipt_walls) {
        auto a = seeded("online");
        SP_CHECK(a.verify_ms(5, std::uint64_t(w.ctx)) == 13.30 + 0.113 * 4.0, "%s: no pass = the seeds", w.name);
        feed(a, w);
        const auto f = a.cost_fit();
        double worst = 0.0;
        for (const auto& r : w.rows)
            if (r[2] >= 4.0) worst = std::max(worst, std::abs(a.verify_ms(std::uint32_t(r[0]), std::uint64_t(w.ctx)) / r[1] - 1.0));
        const double slope = (a.verify_ms(8, std::uint64_t(w.ctx)) - a.verify_ms(4, std::uint64_t(w.ctx))) / 4.0;
        std::printf("cost fit %s (ctx %.0f): c %+.3f beta %+.4f kappa %.4f weight %.1f | priced slope %.3f ms/row | worst row %.1f %%\n",
                    w.name, w.ctx, f.c, f.beta, f.kappa, f.weight, slope, 100.0 * worst);
        SP_CHECK(std::string(f.mode) == "online" && worst <= 0.06, "%s: fit misses a measured row by %.1f %%", w.name, 100.0 * worst);
        if (w.ctx > 20000) SP_CHECK(slope / (w.ctx / 1024.0) > 0.05 && slope / (w.ctx / 1024.0) < 0.10, "%s: slope %.3f", w.name, slope);
    }
    // A one-off stall (5x the wall) after the fit holds is not folded: the 5-row price does not move.
    {
        auto a = seeded("online");
        feed(a, receipt_walls[1]);
        const double before = a.verify_ms(5, 28387);
        a.observe({sp::Source::model, 0, 4, 0, 5, 5.0 * 21.65, 0.0, 28387}, std::vector<std::uint32_t>(1, 1U));
        const double after = a.verify_ms(5, 28387);
        SP_CHECK(after == before && a.cost_fit().skipped == 1, "stall moved the price %.4f -> %.4f", before, after);
    }
    // A device whose walls ARE its seeds: the default fit decides exactly like the previous form (same tokens, walls, k).
    {
        const CtxCase c{"seed device", {0.8, 0.75, 0.7, 0.7, 0.65, 0.6}, 4000, 0.5, 0, 0};
        const auto run = [&](const std::string& mode) {
            auto config = form(-1);
            config.learn_costs = false;
            sp::set_context_cost(config, mode);
            sp::Arbiter a(config);
            for (std::uint32_t r = 1; r <= 32; ++r) a.seed_cost(r, 13.30 + 0.113 * double(r - 1));
            std::vector<double> dms;
            for (std::uint32_t k = 1; k <= 15; ++k) dms.push_back(0.494 * double(k));
            Rng rng{9};
            FrontierRun out;
            for (int pass = 0; pass < 300; ++pass) {
                sp::StepInput in;
                in.model_max_k = 15;
                in.model_draft_ms = dms;
                in.context = std::uint64_t(c.ctx);
                const auto d = a.decide(in);
                std::uint32_t acc = 0;
                if (d.source == sp::Source::model)
                    while (acc < d.k && double(rng.next(1000)) / 1000.0 < c.rate[std::min<std::size_t>(acc, c.rate.size() - 1)]) ++acc;
                out.k_sum += d.k;
                out.tokens += double(acc + 1);
                const double v = 13.30 + 0.113 * double(d.rows - 1);
                out.ms += v;
                a.observe({d.source, 0, d.k, acc, d.rows, v, 0.0, in.context}, std::vector<std::uint32_t>(acc + 1, 1U));
            }
            return out;
        };
        const auto off = run("off"), on = run("online");
        SP_CHECK(off.tokens == on.tokens && off.ms == on.ms && off.k_sum == on.k_sum, "walls == seeds: online must decide as off");
    }
    // Learned costs (DS41 dspark, Gemma4, HIP 'learned'): measured walls price windows; the fit never applies there.
    {
        auto plain = seeded("off", true), fit = seeded("online", true);
        feed(plain, receipt_walls[0]);
        feed(fit, receipt_walls[0]);
        for (const std::uint32_t r : {1U, 5U, 9U, 16U})
            SP_CHECK(plain.verify_ms(r, 28387) == fit.verify_ms(r, 28387), "learned: fit changed rows %u", r);
        SP_CHECK(std::string(fit.cost_fit().mode) == "learned", "learned mode name");
    }
    // The stats line carries the fit (old parsers: a new trailing key only).
    {
        sp::SpecStats st;
        SP_CHECK(st.json().find("cost_fit") == std::string::npos, "no fit field before a pass");
        auto a = seeded("online");
        feed(a, receipt_walls[1]);
        st.cost_fit = a.cost_fit();
        st.has_cost_fit = true;
        const std::string j = st.json();
        SP_CHECK(j.find(",\"cost_fit\":{\"mode\":\"online\",\"c\":") != std::string::npos && j.back() == '}', "stats json: %s", j.c_str());
    }
}

// Identification of the context term (ArbiterConfig::cost_fit_condition). Per-row walls of measured runs
// (RTX PRO 6000 at 600 W, Qwen 27B dense NVFP4, arm sub15d, n = 1 each; stats "rows": rows, mean verify ms, passes),
// prompt length and generated tokens: the context grows over each run.
struct KitWalls { const char* name; double ctx0, gen; bool long_ctx; std::vector<std::array<double, 3>> rows; };
const KitWalls kit_walls[] = {
    {"soff en-prose", 74, 1200, false, {{3, 12.26, 5}, {4, 12.88, 191}, {5, 13.00, 258}, {6, 13.01, 22}, {7, 12.99, 6}, {8, 12.91, 4}}},
    {"soff ja-prose", 79, 1200, false, {{3, 12.59, 60}, {4, 12.89, 503}, {5, 12.91, 17}}},
    {"soff code", 59, 1200, false, {{9, 14.18, 39}, {10, 14.70, 21}, {11, 14.96, 33}, {12, 15.14, 17}, {13, 15.51, 26}, {14, 15.66, 25},
                                    {15, 15.97, 7}, {16, 15.39, 12}}},
    {"soff think", 129, 1200, false, {{7, 13.26, 28}, {8, 13.52, 111}, {9, 14.51, 65}, {10, 15.04, 26}, {11, 15.19, 16}, {12, 15.31, 6},
                                      {13, 14.92, 3}, {16, 15.44, 3}}},
    {"son en-prose", 74, 1200, false, {{3, 12.27, 5}, {4, 12.88, 259}, {5, 13.00, 195}, {6, 13.03, 24}, {7, 12.87, 4}}},
    {"son ja-prose", 79, 1200, false, {{3, 12.76, 408}, {4, 12.87, 103}, {5, 13.06, 86}}},
    {"son code", 59, 1200, false, {{8, 13.29, 41}, {9, 14.37, 99}, {10, 14.66, 19}, {11, 15.09, 21}, {12, 15.53, 16}}},
    {"son think", 129, 1200, false, {{7, 13.22, 132}, {8, 13.48, 69}, {9, 14.44, 21}, {10, 15.04, 14}, {11, 15.19, 11}, {12, 15.46, 6},
                                     {13, 15.17, 5}, {14, 15.55, 3}, {16, 15.77, 6}}},
    {"n8off code", 6641, 300, true, {{6, 15.41, 4}, {8, 16.41, 10}, {9, 18.00, 50}}},
    {"n8on code", 6641, 300, true, {{5, 14.43, 3}, {6, 15.43, 3}, {7, 16.13, 14}, {8, 16.31, 44}, {9, 17.78, 5}}},
    {"c28off", 28187, 400, true, {{6, 23.84, 3}, {8, 27.29, 42}, {9, 30.35, 44}, {10, 32.21, 6}}},
    {"c28on", 28187, 400, true, {{4, 19.94, 6}, {5, 21.61, 89}, {6, 23.94, 11}, {7, 25.65, 7}}},
    {"t28off", 27711, 400, true, {{7, 25.64, 3}, {8, 27.32, 5}, {9, 30.34, 30}, {10, 32.42, 32}, {11, 34.01, 6}}},
    {"t28on", 27711, 400, true, {{4, 19.87, 3}, {5, 21.57, 33}, {6, 23.45, 50}, {7, 25.61, 11}}},
    {"e28off", 27581, 400, true, {{4, 19.90, 41}, {5, 21.65, 100}, {6, 23.34, 14}, {7, 25.63, 5}}},
    {"e28on", 27581, 400, true, {{3, 17.59, 167}, {4, 19.90, 10}, {5, 21.67, 5}}},
};

sp::Arbiter seeded_arbiter(const std::string& mode, double condition = 2.0) {
    auto config = form(-1);
    config.learn_costs = false;  // BlockConfig::fixed_costs
    sp::set_context_cost(config, mode);
    config.cost_fit_condition = condition;
    sp::Arbiter a(config);
    for (std::uint32_t r = 1; r <= 32; ++r) a.seed_cost(r, 13.30 + 0.113 * double(r - 1));
    return a;
}
// A receipt's own walls, passes interleaved across rows, the context growing from the prompt to prompt + generated.
void feed_kit(sp::Arbiter& a, const KitWalls& w) {
    double total = 0.0, done = 0.0;
    for (const auto& r : w.rows) total += r[2];
    std::vector<std::array<double, 3>> left = w.rows;
    for (bool any = true; any;) {
        any = false;
        for (auto& r : left)
            if (r[2] > 0.0) {
                const auto ctx = std::uint64_t(w.ctx0 + w.gen * done / total);
                a.observe({sp::Source::model, 0, std::uint32_t(r[0]) - 1U, 0, std::uint32_t(r[0]), r[1], 0.0, ctx},
                          std::vector<std::uint32_t>(1, 1U));
                r[2] -= 1.0;
                done += 1.0;
                any = true;
            }
    }
}
// Every decision of a run (source, k, rows), the context growing by every emitted token from ctx0; walls = device_wall
// at that context (row tiles at 9 / 17 rows, K/V 0.0686) x (1 + drift x progress), and one stall at pass stall_pass: stall_ms,
// or stall_x x that pass's own wall when stall_x > 0 (stall_pass past the run = no stall).
struct Trace {
    FrontierRun run;
    std::vector<std::array<std::uint32_t, 3>> picks;
    std::uint32_t passes = 0, open = 0;
    sp::Arbiter::CostFit fit;
};
Trace replay_trace(const std::string& mode, double condition, const std::vector<double>& rate, double draft, double ctx0,
                   double gen, double drift, std::uint64_t seed, double stall_ms = 150.0, std::uint32_t stall_pass = 100,
                   double stall_x = 0.0) {
    auto a = seeded_arbiter(mode, condition);
    std::vector<double> dms;
    for (std::uint32_t k = 1; k <= 15; ++k) dms.push_back(0.494 * double(k));
    Rng rng{seed};
    Trace t;
    while (t.run.tokens < gen) {
        sp::StepInput in;
        in.model_max_k = 15;
        in.model_draft_ms = dms;
        in.context = std::uint64_t(ctx0 + t.run.tokens);
        const auto d = a.decide(in);
        std::uint32_t acc = 0;
        if (d.source == sp::Source::model) {
            while (acc < d.k && double(rng.next(1000)) / 1000.0 < rate[std::min<std::size_t>(acc, rate.size() - 1)]) ++acc;
            ++t.run.model;
            t.run.k_sum += d.k;
        }
        const double dev = device_wall(d.rows, double(in.context)) * (1.0 + drift * t.run.tokens / gen);
        const double wall = t.passes != stall_pass ? dev : stall_x > 0.0 ? stall_x * dev : stall_ms;
        const double dr = d.source == sp::Source::model ? draft * double(d.k) : 0.0;
        a.observe({d.source, 0, d.k, acc, d.rows, wall, dr, in.context}, std::vector<std::uint32_t>(acc + 1, 1U));
        t.picks.push_back({std::uint32_t(d.source), d.k, d.rows});
        t.run.tokens += double(acc + 1);
        t.run.ms += wall + dr;
        ++t.passes;
        if (a.cost_fit().kv_open) ++t.open;
    }
    t.fit = a.cost_fit();
    return t;
}

void context_gate_tests() {
    // The condition: seeds given at every context stand for bin 0 [0, 2,048): q = L / (L - 2,048) <= 2 from 4,096 on;
    // a table given for one bin never identifies the term inside that bin; 0 = no test (the previous form).
    {
        auto a = seeded_arbiter("online");
        for (const std::uint64_t l : {0ULL, 1329ULL, 2048ULL, 4095ULL}) SP_CHECK(!a.kv_identified(l), "identified at %llu", (unsigned long long)l);
        for (const std::uint64_t l : {4096ULL, 6641ULL, 28187ULL}) SP_CHECK(a.kv_identified(l), "not identified at %llu", (unsigned long long)l);
        auto prev = seeded_arbiter("online", 0.0);
        SP_CHECK(prev.kv_identified(0) && prev.kv_identified(1329), "condition 0 = every context");
        auto cfg = form(-1);
        cfg.learn_costs = false;
        sp::Arbiter bin2(cfg);
        for (std::uint32_t r = 1; r <= 32; ++r) bin2.seed_cost(r, 20.0 + 1.9 * double(r - 1), 20000);
        SP_CHECK(!bin2.kv_identified(28672) && !bin2.kv_identified(1000), "a bin-2 table never identifies inside bin 2");
        auto fixed = seeded_arbiter("0.0699"), off = seeded_arbiter("off");
        for (const std::uint32_t r : {1U, 5U, 9U, 16U}) {
            SP_CHECK(fixed.verify_ms(r, 1329) == off.verify_ms(r, 1329), "fixed KV at a short context = off (rows %u)", r);
            SP_CHECK(std::abs(fixed.verify_ms(r, 28672) - off.verify_ms(r, 28672) - 0.0699 * double(r) * 28.0) < 1e-9, "fixed term");
        }
        bool threw = false;
        try { cfg.cost_fit_condition = 0.5; sp::Arbiter bad(cfg); (void)bad; } catch (const std::invalid_argument&) { threw = true; }
        SP_CHECK(threw, "cost_fit_condition in (0, 1) must be rejected");
    }
    // 1. Every kit receipt's own walls: the term is closed on the eight short runs (the price stays the seed table, bit for
    //    bit) and open on the eight 8K / 28K runs.
    for (const auto& w : kit_walls) {
        auto a = seeded_arbiter("online"), off = seeded_arbiter("off");
        feed_kit(a, w);
        const auto f = a.cost_fit();
        const auto end = std::uint64_t(w.ctx0 + w.gen);
        bool same = true;
        for (std::uint32_t r = 1; r <= 16; ++r) same = same && a.verify_ms(r, end) == off.verify_ms(r, end);
        std::printf("context gate %s (ctx %.0f -> %llu): kv %s, mean ctx %.0f, kappa %.4f c %+.3f beta %+.4f, price %s\n", w.name,
                    w.ctx0, (unsigned long long)end, f.kv_open ? "open" : "closed", f.context, f.kappa, f.c, f.beta,
                    same ? "== seeds" : "fitted");
        SP_CHECK(f.kv_open == w.long_ctx && same == !w.long_ctx, "%s: gate %s", w.name, f.kv_open ? "open" : "closed");
    }
    // 2. Decisions on the four short chart contents (survival of the short-context receipts, prompts 59-129, 1,200 tokens),
    //    row tiles, drift 0 / 3 / 6 %, one 150 ms stall: online and fixed decide exactly as off (every pass: source, k,
    //    rows; same tokens, same walls). The previous form (condition 0) is printed for the record.
    struct ShortCase { const char* name; const Shape* sh; double ctx0; };
    const ShortCase shorts[] = {{"ja-prose", &ja_f8, 79}, {"en-prose", &en_f8, 74}, {"code", &code_f8, 59}, {"think", &think_f8, 129}};
    for (const auto& s : shorts) {
        std::uint32_t prev_diff = 0, passes = 0;
        double prev_tok = 0.0, off_tok = 0.0;
        for (const double drift : {0.0, 0.03, 0.06})
            for (const std::uint64_t seed : {1ULL, 7ULL, 42ULL}) {
                const auto off = replay_trace("off", 2.0, s.sh->rate, 0.5, s.ctx0, 1200, drift, seed);
                const auto on = replay_trace("online", 2.0, s.sh->rate, 0.5, s.ctx0, 1200, drift, seed);
                const auto fix = replay_trace("0.0699", 2.0, s.sh->rate, 0.5, s.ctx0, 1200, drift, seed);
                const auto prev = replay_trace("online", 0.0, s.sh->rate, 0.5, s.ctx0, 1200, drift, seed);
                SP_CHECK(on.picks == off.picks && on.run.tokens == off.run.tokens && on.run.ms == off.run.ms && on.open == 0,
                         "short %s drift %.2f seed %llu: online decided differently from off", s.name, drift, (unsigned long long)seed);
                SP_CHECK(fix.picks == off.picks, "short %s: fixed KV decided differently from off", s.name);
                for (std::size_t i = 0; i < std::min(prev.picks.size(), off.picks.size()); ++i) prev_diff += prev.picks[i] != off.picks[i] ? 1U : 0U;
                passes += off.passes;
                prev_tok += prev.run.tok_s() / 9.0;
                off_tok += off.run.tok_s() / 9.0;
            }
        std::printf("short %s: online == off on %u passes (9 runs); previous form differed on %u passes, %+.2f %% tok/s\n", s.name,
                    passes, prev_diff, 100.0 * (prev_tok / off_tok - 1.0));
    }
    // 3. Long context (28K three contents, 8K; 400 generated tokens, 3 % drift, one 150 ms stall): the gate is open from the
    //    first fitted pass and the decisions keep the gain of the previous form within 1 %.
    for (const auto& c : ctx_cases) {
        double off = 0.0, on = 0.0, prev = 0.0;
        std::uint32_t closed = 0;
        for (const std::uint64_t seed : {1ULL, 7ULL, 42ULL}) {
            off += replay_trace("off", 2.0, c.rate, c.draft, c.ctx, 400, 0.03, seed).run.tok_s() / 3.0;
            const auto t = replay_trace("online", 2.0, c.rate, c.draft, c.ctx, 400, 0.03, seed);
            on += t.run.tok_s() / 3.0;
            closed += t.passes - t.open;
            prev += replay_trace("online", 0.0, c.rate, c.draft, c.ctx, 400, 0.03, seed).run.tok_s() / 3.0;
        }
        std::printf("long %s: off %.1f | online %.1f (%+.1f %%) | previous form %.1f (%+.1f %%) | closed passes %u\n", c.name, off, on,
                    100.0 * (on / off - 1.0), prev, 100.0 * (prev / off - 1.0), closed);
        SP_CHECK(std::abs(on / prev - 1.0) <= 0.01 && closed == 0, "%s: online %.1f vs previous form %.1f, %u closed", c.name, on, prev,
                 closed);
        if (c.ctx > 20000) SP_CHECK(on >= 1.03 * off, "%s: expected the 28K gain (%.1f vs %.1f)", c.name, on, off);
    }
    // 4. One 150 ms stall leaves the fit bit-identical to the stream without it; the next pass folds as usual; two in a row
    //    are a level change: the second is folded.
    {
        auto a = seeded_arbiter("online"), b = seeded_arbiter("online");
        feed_kit(a, kit_walls[11]);
        feed_kit(b, kit_walls[11]);
        const auto same = [](const sp::Arbiter::CostFit& x, const sp::Arbiter::CostFit& y) {
            return x.c == y.c && x.beta == y.beta && x.kappa == y.kappa && x.weight == y.weight && x.context == y.context;
        };
        const sp::Outcome stall{sp::Source::model, 0, 4, 0, 5, 150.0, 0.0, 28587}, normal{sp::Source::model, 0, 4, 0, 5, 21.6, 0.0, 28587};
        a.observe(stall, std::vector<std::uint32_t>(1, 1U));
        SP_CHECK(same(a.cost_fit(), b.cost_fit()) && a.cost_fit().skipped == 1, "one stall moved the fit");
        a.observe(normal, std::vector<std::uint32_t>(1, 1U));
        b.observe(normal, std::vector<std::uint32_t>(1, 1U));
        SP_CHECK(same(a.cost_fit(), b.cost_fit()), "the pass after a stall folds as usual");
        a.observe(stall, std::vector<std::uint32_t>(1, 1U));
        a.observe(stall, std::vector<std::uint32_t>(1, 1U));
        SP_CHECK(a.cost_fit().weight > b.cost_fit().weight && a.cost_fit().skipped == 2, "a second stall in a row is folded");
    }
    // 5. A short request's walls never price a long request's K/V (the fit's mean context must identify it too), and the
    //    stats line says which side of the gate the term is on.
    {
        auto a = seeded_arbiter("online"), off = seeded_arbiter("off");
        feed_kit(a, kit_walls[2]);
        SP_CHECK(a.verify_ms(5, 28387) == off.verify_ms(5, 28387), "short walls priced a long context");
        sp::SpecStats st;
        st.cost_fit = a.cost_fit();
        st.has_cost_fit = true;
        SP_CHECK(st.json().find("\"kv\":\"closed\"") != std::string::npos, "stats: %s", st.json().c_str());
        feed_kit(a, kit_walls[11]);
        st.cost_fit = a.cost_fit();
        SP_CHECK(a.cost_fit().kv_open && a.verify_ms(5, 28387) != off.verify_ms(5, 28387) &&
                     st.json().find("\"kv\":\"open\"") != std::string::npos, "long walls after short: %s", st.json().c_str());
    }
    // 6. A stall in the warm-up (one 2x or 5x wall at pass 1 or pass 3; code 6.6K x 300, code / reasoning 28K x 400; 3 %
    //    drift; five seeds, the same stall in both arms): online keeps >= off - 0.5 % (seed mean) and its mean k stays within
    //    1.0 of the stall-free online run. The stall-free runs skip no pass, so they decide exactly as before the warm-up
    //    test (it acts only on a skipped pass).
    {
        struct StallCase { const char* name; const CtxCase* c; double ctx, gen; };
        const StallCase cases[] = {{"code 6.6K", &ctx_cases[3], 6600, 300}, {"code 28K", &ctx_cases[0], 28387, 400},
                                   {"think 28K", &ctx_cases[1], 27911, 400}};
        const std::uint64_t seeds[] = {1, 7, 42, 9, 23};
        for (const auto& s : cases) {
            double free_k = 0.0;
            for (const auto seed : seeds) {
                const auto t = replay_trace("online", 2.0, s.c->rate, s.c->draft, s.ctx, s.gen, 0.03, seed, 150.0, 1U << 30);
                SP_CHECK(t.fit.skipped == 0, "%s seed %llu: a stall-free run skipped %u passes", s.name, (unsigned long long)seed,
                         unsigned(t.fit.skipped));
                free_k += t.run.mean_k() / 5.0;
            }
            for (const std::uint32_t pass : {1U, 3U})
                for (const double x : {2.0, 5.0}) {
                    double off = 0.0, on = 0.0, k = 0.0;
                    for (const auto seed : seeds) {
                        off += replay_trace("off", 2.0, s.c->rate, s.c->draft, s.ctx, s.gen, 0.03, seed, 0.0, pass, x).run.tok_s() / 5.0;
                        const auto t = replay_trace("online", 2.0, s.c->rate, s.c->draft, s.ctx, s.gen, 0.03, seed, 0.0, pass, x);
                        on += t.run.tok_s() / 5.0;
                        k += t.run.mean_k() / 5.0;
                    }
                    std::printf("warm-up stall %s, %.0fx at pass %u: off %.1f | online %.1f (%+.1f %%) | k %.2f (stall-free %.2f)\n",
                                s.name, x, pass, off, on, 100.0 * (on / off - 1.0), k, free_k);
                    SP_CHECK(on >= 0.995 * off && std::abs(k - free_k) <= 1.0, "%s %.0fx at pass %u: online %.1f vs off %.1f, k %.2f vs %.2f",
                             s.name, x, pass, on, off, k, free_k);
                }
        }
    }
    // 7. A level is not a stall: stall-free code 131K x 300 (3 % drift, five seeds). The first walls are 2-10x the
    //    context-blind seeds' prediction; the stall test must learn that level, not drop every other pass. Reference: the
    //    same replay without a warm-up stall test (the rule acting only after the warm-up never fires here, skipped 0):
    //    online / off = 56.9 / 43.6 = 1.305. Online keeps >= 98 % of that and skips <= 2 passes (seed mean); dropping
    //    every other level pass gives 1.065 with 6.6 skipped.
    {
        const std::uint64_t seeds[] = {1, 7, 42, 9, 23};
        double off = 0.0, on = 0.0, skipped = 0.0;
        for (const auto seed : seeds) {
            off += replay_trace("off", 2.0, ctx_cases[0].rate, ctx_cases[0].draft, 131072, 300, 0.03, seed, 150.0, 1U << 30).run.tok_s() / 5.0;
            const auto t = replay_trace("online", 2.0, ctx_cases[0].rate, ctx_cases[0].draft, 131072, 300, 0.03, seed, 150.0, 1U << 30);
            on += t.run.tok_s() / 5.0;
            skipped += double(t.fit.skipped) / 5.0;
        }
        std::printf("level 131K x 300 stall-free: off %.1f | online %.1f (x%.3f, reference x1.305) | skipped %.1f\n", off, on,
                    on / off, skipped);
        SP_CHECK(on >= 0.98 * 1.305 * off && skipped <= 2.0, "level 131K: online %.1f vs off %.1f (x%.3f, need >= x%.3f), skipped %.1f",
                 on, off, on / off, 0.98 * 1.305, skipped);
    }
    // 8. Regime rule: the default limit (ArbiterConfig::cost_fit_condition 1.5) opens the term for bin-0 seeds from 6,144
    //    positions. A 4-5K run (prompt about 3.5-3.65K + 1,200 tokens, where the limit 2 lost on code) decides exactly as
    //    off on every pass; the 6.6K and 28K runs decide exactly as under the limit 2 (their receipts stand).
    {
        sp::ArbiterConfig defaults;  // not const: the check reads the shipped default at run time
        const double def = defaults.cost_fit_condition;
        SP_CHECK(def == 1.5, "default cost_fit_condition %.2f, want 1.5", def);
        auto a = seeded_arbiter("online", def);
        for (const std::uint64_t l : {4096ULL, 4847ULL, 6143ULL}) SP_CHECK(!a.kv_identified(l), "limit 1.5: identified at %llu", (unsigned long long)l);
        for (const std::uint64_t l : {6144ULL, 6641ULL, 28187ULL}) SP_CHECK(a.kv_identified(l), "limit 1.5: not identified at %llu", (unsigned long long)l);
        struct RegimeCase { const char* name; const Shape* sh; double ctx0, gen; bool open; };
        const RegimeCase cases[] = {{"code 4-5K", &code_f8, 3600, 1200, false}, {"think 4-5K", &think_f8, 3650, 1200, false},
                                    {"prose 4-5K", &en_f8, 3500, 1200, false}, {"code 6.6K", &code_f8, 6641, 300, true},
                                    {"code 28K", &code_f8, 28187, 400, true}};
        for (const auto& c : cases) {
            std::uint32_t open = 0, passes = 0;
            for (const std::uint64_t seed : {1ULL, 7ULL, 42ULL}) {
                const auto on = replay_trace("online", def, c.sh->rate, 0.5, c.ctx0, c.gen, 0.03, seed);
                const auto ref = replay_trace(c.open ? "online" : "off", c.open ? 2.0 : def, c.sh->rate, 0.5, c.ctx0, c.gen, 0.03, seed);
                SP_CHECK(on.picks == ref.picks, "%s seed %llu: decisions differ from %s", c.name, (unsigned long long)seed,
                         c.open ? "the limit 2" : "off");
                open += on.open;
                passes += on.passes;
            }
            std::printf("regime %s (ctx %.0f + %.0f): term open on %u of %u passes, decisions == %s\n", c.name, c.ctx0, c.gen, open,
                        passes, c.open ? "limit 2" : "off");
            SP_CHECK(c.open ? open > 0U : open == 0U, "%s: term open on %u passes", c.name, open);
        }
    }
}

void frontier_tests() {
    for (const std::uint64_t seed : {1ULL, 7ULL, 42ULL}) {
        const auto old_form = frontier_run(0, seed);
        const auto fixed = frontier_run(sp::ArbiterConfig{}.frontier_every, seed);
        // The previous form stays at k = 1 (the receipt; only a rare retreat + deep re-probe can free it); the frontier
        // probe learns deeper depths and leaves k = 1.
        SP_CHECK(double(old_form.k1) >= 0.8 * double(old_form.model), "old form: k = 1 on %llu of %llu model passes",
                 static_cast<unsigned long long>(old_form.k1), static_cast<unsigned long long>(old_form.model));
        SP_CHECK(fixed.max_k_after_first >= 2 && double(fixed.k1) < 0.5 * double(fixed.model), "frontier: max k %u, k = 1 on %llu of %llu",
                 fixed.max_k_after_first, static_cast<unsigned long long>(fixed.k1), static_cast<unsigned long long>(fixed.model));
        const double old_rate = old_form.tokens / old_form.ms, new_rate = fixed.tokens / fixed.ms;
        std::printf("frontier seed %llu: off %.2f tok/step %.1f tok/s | on %.2f tok/step %.1f tok/s (max k %u, probes %llu)\n",
                    static_cast<unsigned long long>(seed), old_form.tokens / 600.0, 1000.0 * old_rate, fixed.tokens / 600.0,
                    1000.0 * new_rate, fixed.max_k_after_first, static_cast<unsigned long long>(fixed.probes));
        SP_CHECK(new_rate > 1.10 * old_rate, "frontier gain %.3f below +10 %%", new_rate / old_rate);
        // The continuation prior prices an unexplored depth at r_1^2, so the k = 1 absorbing state is gone even with the probe
        // off; the defaults (continuation + charged probe) keep the f8 gain on this shape.
        auto no_probe = form(-1);
        no_probe.frontier_every = 0;
        const auto cont = replay(no_probe, stuck_prose, seed), now = replay(form(-1), stuck_prose, seed);
        std::printf("frontier seed %llu: continuation, probe off %.1f tok/s (k = 1 on %llu of %llu) | defaults %.1f tok/s\n",
                    static_cast<unsigned long long>(seed), cont.tok_s(), static_cast<unsigned long long>(cont.k1),
                    static_cast<unsigned long long>(cont.model), now.tok_s());
        SP_CHECK(double(cont.k1) < 0.5 * double(cont.model) && cont.tok_s() > 1.10 * 1000.0 * old_rate, "continuation stays stuck");
        SP_CHECK(now.tok_s() >= 0.98 * 1000.0 * new_rate, "defaults %.1f below f8 %.1f on the stuck shape", now.tok_s(), 1000.0 * new_rate);
    }
    // Black-box drafters (model_pass_ms) are priced on their measured rate: no frontier probe changes their k.
    sp::ArbiterConfig config;
    sp::Arbiter a(config);
    for (int pass = 0; pass < 64; ++pass) {
        sp::StepInput in;
        in.model_max_k = 7;
        in.model_pass_tokens = 3.0;
        in.model_pass_ms = 20.0;
        const auto d = a.decide(in);
        SP_CHECK(d.source != sp::Source::model || d.k == 7, "black-box k %u", d.k);
        a.observe({d.source, 0, d.k, 1, d.rows, 20.0, 0.0}, std::vector<std::uint32_t>(2, 1U));
    }
}
// Exploration probe charge (ArbiterConfig::charge_probe). CUDA dense 27B sub15d shape: seeds 13.30 + 0.113 (r - 1) ms,
// K/V 0.0699 ms per verify row per 1,024 positions (28K receipt), draft 0.824 ms per position, windows up to 16 rows (k 15).
// The device is the priced model itself (fixed K/V term) and acceptance is a deterministic draw at survival p per depth.
struct ProbeRun {
    std::vector<std::uint32_t> ks;  // k of every pass (0 = serial)
    double tokens = 0.0, ms = 0.0, serial_ms = 0.0, budget = 0.0, max_excess = -1e300;
    std::uint32_t probes = 0, max_probe_k = 0, min_probe_k = ~0U;
};
ProbeRun probe_run(bool charge, std::uint64_t ctx, double p, int passes) {
    sp::ArbiterConfig config;
    config.windows = {1, 2, 3, 4, 6, 8, 12, 16};
    config.learn_costs = false;
    config.row_kv_ms_per_kpos = 0.0699;
    config.charge_probe = charge;
    sp::Arbiter a(config);
    for (std::uint32_t r = 1; r <= 16; ++r) a.seed_cost(r, 13.30 + 0.113 * double(r - 1));
    const std::vector<double> survival(15, p);
    a.seed_model_survival(survival);
    sp::StepInput in;
    in.model_max_k = 15;
    std::vector<double> draft_ms;
    for (std::uint32_t i = 1; i <= 15; ++i) draft_ms.push_back(0.824 * double(i));
    in.model_draft_ms = draft_ms;
    in.context = ctx;
    ProbeRun run;
    run.serial_ms = a.verify_ms(1, ctx);
    run.budget = config.frontier_loss * double(config.probe_every) * run.serial_ms;
    std::uint64_t s = 0x9E3779B97F4A7C15ULL;
    const auto draw = [&s] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(s >> 11) * 0x1.0p-53;
    };
    for (int i = 0; i < passes; ++i) {
        const auto d = a.decide(in);
        run.ks.push_back(d.source == sp::Source::serial ? 0U : d.k);
        if (d.source == sp::Source::serial) {
            run.tokens += 1.0;
            run.ms += run.serial_ms;
            a.observe({sp::Source::serial, 0, 0, 0, 1, run.serial_ms, 0.0, ctx}, {});
            continue;
        }
        std::uint32_t acc = 0;
        while (acc < d.k && draw() < p) ++acc;
        const double v = a.verify_ms(d.rows, ctx), dr = 0.824 * double(d.k);
        run.tokens += double(acc + 1U);
        run.ms += v + dr;
        if (d.probe) {
            ++run.probes;
            run.max_probe_k = std::max(run.max_probe_k, d.k);
            run.min_probe_k = std::min(run.min_probe_k, d.k);
            if (d.k > 1) run.max_excess = std::max(run.max_excess, d.expected_ms - d.expected_tokens * run.serial_ms);
        }
        a.observe({sp::Source::model, 0, d.k, acc, d.rows, v, dr, ctx}, {});
    }
    return run;
}

void probe_charge_tests() {
    // Short context (2K, serial wins at p 0.05): the deepest probe prices below its budget, so the charged arbiter makes
    // the previous decisions pass for pass.
    {
        const auto on = probe_run(true, 2048, 0.05, 320), off = probe_run(false, 2048, 0.05, 320);
        SP_CHECK(on.ks == off.ks, "short context: the charged probe changed a decision");
        SP_CHECK(off.probes >= 8 && off.max_probe_k == 15 && on.max_probe_k == 15, "short context: deep probes (%u, k %u)",
                 off.probes, off.max_probe_k);
    }
    // 128K (serial wins at p 0.4: 22.2 ms vs k 1 at 32.1 ms for 1.4 tokens): the previous probe runs k 15 (~170 ms); the
    // charged one stays within its budget (frontier_loss x probe_every serial passes) and keeps >= 95 % of serial's rate.
    {
        const auto on = probe_run(true, 131072, 0.4, 640), off = probe_run(false, 131072, 0.4, 640);
        const double serial_rate = 1.0 / on.serial_ms, rate_on = on.tokens / on.ms, rate_off = off.tokens / off.ms;
        SP_CHECK(off.max_probe_k == 15, "128K: the previous probe runs the deepest k (%u)", off.max_probe_k);
        SP_CHECK(on.probes >= 16 && on.max_probe_k < 15 && on.min_probe_k >= 1, "128K: charged probes %u, k %u..%u", on.probes,
                 on.min_probe_k, on.max_probe_k);
        SP_CHECK(on.max_excess <= on.budget + 1e-9, "128K: priced probe excess %.3f ms > budget %.3f ms", on.max_excess, on.budget);
        SP_CHECK(rate_on > rate_off && rate_on >= 0.95 * serial_rate,
                 "128K: charged %.2f tok/s, previous %.2f tok/s, serial %.2f tok/s", 1000.0 * rate_on, 1000.0 * rate_off,
                 1000.0 * serial_rate);
        std::printf("probe charge 128K p 0.4: serial %.2f tok/s, previous probe %.2f (k %u), charged %.2f (k %u..%u), %u probes\n",
                    1000.0 * serial_rate, 1000.0 * rate_off, off.max_probe_k, 1000.0 * rate_on, on.min_probe_k, on.max_probe_k,
                    on.probes);
    }
}

} // namespace

int main() {
    index_tests();
    copy_tests();
    arbiter_tests();
    retreat_horizon_tests();
    ds_arbiter_tests();
    frontier_tests();
    pricing_tests();
    context_pricing_tests();
    cost_fit_tests();
    context_gate_tests();
    probe_charge_tests();
    std::printf("PASS strix-spec-drafting-tests\n");
    return 0;
}
