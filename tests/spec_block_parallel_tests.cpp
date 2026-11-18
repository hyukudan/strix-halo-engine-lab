// Parallel block drafter contract (strix/spec/block_parallel.hpp): block layout, row visibility, presets, the controller
// binding through ChainDrafter with a synthetic drafter of controlled accuracy (emitted ids == serial greedy decode for
// every policy / accuracy / source mix, ONE forward per draft, anchor position tracked through every step kind), the flat
// cost config, and the MoE verify pricing used in the HY4 break-even table.
#include "strix/spec/block_parallel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace sp = strix::spec;

namespace {
#define BP_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

double g_clock = 0.0;  // virtual ms

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

struct World {
    std::vector<std::uint32_t> prompt, response;
};
World make_world(std::uint32_t n, std::uint64_t seed) {
    World w;
    for (std::uint32_t i = 0; i < 40; ++i) w.prompt.push_back(std::uint32_t(mix(seed * 977 + i) % 5000U) + 10U);
    // phrases repeat every so often, so the copy source fires too
    for (std::uint32_t i = 0; i < n + 64; ++i) {
        const std::uint64_t h = mix(seed + i);
        w.response.push_back(h % 4 == 0 && i >= 24 ? w.response[i - 24] : std::uint32_t(h % 5000U) + 10U);
    }
    return w;
}

struct Target final : sp::VerifyTarget {
    const World& w;
    std::uint32_t committed = 0;  // response rows in the target (the anchor excluded)
    std::vector<std::uint32_t> last;
    explicit Target(const World& world) : w(world) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        greedy.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) greedy.push_back(w.response[committed + i + 1]);
        last.assign(tokens.begin(), tokens.end());
        g_clock += 10.0 + 0.4 * double(tokens.size());
    }
    void commit(std::uint32_t k) override {
        BP_CHECK(k >= 1 && k <= last.size(), "commit %u of %zu", k, last.size());
        committed += k;
    }
};

// Synthetic drafter: knows the response, each position is right with probability `acc` (deterministic per position).
struct Forward final : sp::BlockForward {
    const World& w;
    sp::BlockPolicy p;
    double acc;
    std::uint64_t anchor_pos;  // its own view, advanced by accept()
    std::uint64_t forwards = 0, proposals = 0;
    Forward(const World& world, const sp::BlockPolicy& policy, double a)
        : w(world), p(policy), acc(a), anchor_pos(world.prompt.size()) {}
    void forward(const sp::BlockRequest& req, std::vector<std::uint32_t>& out, std::vector<float>& conf) override {
        ++forwards;
        BP_CHECK(req.anchor_pos == anchor_pos, "anchor pos %llu vs %llu", (unsigned long long)req.anchor_pos, (unsigned long long)anchor_pos);
        const std::uint64_t a = anchor_pos - w.prompt.size();
        BP_CHECK(req.ids.size() == sp::rows_needed(p, req.prefix, req.k), "rows %zu", req.ids.size());
        BP_CHECK(req.ids[0] == w.response[a], "row 0 must be the anchor");
        for (std::size_t r = 1 + req.prefix; r < req.ids.size(); ++r) BP_CHECK(req.ids[r] == p.fill_token, "fill row %zu", r);
        BP_CHECK(req.k >= 1 && req.k <= sp::max_drafts(p, req.prefix), "k %u", req.k);
        for (std::uint32_t j = 0; j < req.k; ++j) {
            const std::uint64_t pos = a + 1 + req.prefix + j;
            const bool right = double(mix(pos * 0x9E3779B97F4A7C15ULL + 7) % 1000U) < acc * 1000.0;
            out.push_back(right ? w.response[pos] : 3U);
            conf.push_back(right ? 1.0f : 0.0f);
        }
        proposals += req.k;
        g_clock += 2.0;  // one pass, any k
    }
    void accept(std::span<const std::uint32_t> emitted) override {
        const std::uint64_t a = anchor_pos - w.prompt.size();
        for (std::size_t i = 0; i < emitted.size(); ++i) BP_CHECK(emitted[i] == w.response[a + 1 + i], "accepted id %zu", i);
        anchor_pos += emitted.size();
    }
};

void layout_tests() {
    for (const auto& p : {sp::hy4_dfly2(), sp::ds41_dspark(), sp::g4_dflash()}) sp::validate(p);
    const auto h = sp::hy4_dfly2();
    BP_CHECK(sp::max_drafts(h) == 5 && sp::proposal_row(h, 0) == 1 && sp::proposal_row(h, 4) == 5, "dfly2 rows");
    BP_CHECK(sp::max_drafts(sp::ds41_dspark()) == 5 && sp::proposal_row(sp::ds41_dspark(), 0) == 0, "dspark rows");
    BP_CHECK(sp::max_drafts(sp::g4_dflash()) == 7, "dflash rows");
    BP_CHECK(h.mask == sp::BlockMask::causal && !h.correction, "dfly2 measured numerics: causal block, correction off");
    auto b = h;
    b.mask = sp::BlockMask::bidirectional;
    std::vector<std::uint32_t> ids;
    sp::block_ids(b, 77, {}, 2, ids);  // bidirectional: the whole block runs even for k = 2
    BP_CHECK(ids.size() == 6 && ids[0] == 77 && ids[1] == 120009 && ids[5] == 120009, "dfly2 ids");
    BP_CHECK(sp::row_visible(b, 1, 5) && sp::row_visible(b, 5, 0) && !sp::row_visible(b, 6, 0), "bidirectional");
    auto c = h;
    c.mask = sp::BlockMask::causal;
    c.prefix_in_block = true;
    sp::block_ids(c, 77, {}, 2, ids);
    BP_CHECK(ids.size() == 3, "causal k=2 runs anchor + 2 rows, got %zu", ids.size());
    const std::uint32_t pre[2] = {5, 6};
    sp::block_ids(c, 77, pre, 3, ids);
    BP_CHECK(ids.size() == 6 && ids[1] == 5 && ids[2] == 6 && ids[3] == 120009, "prefix rows");
    BP_CHECK(sp::max_drafts(c, 2) == 3 && !sp::row_visible(c, 1, 2) && sp::row_visible(c, 2, 1), "causal");
    bool threw = false;
    try { (void)sp::max_drafts(h, 1); } catch (const std::invalid_argument&) { threw = true; }
    BP_CHECK(threw, "dfly2 takes no prefix");
    auto bad = h;
    bad.target_layer_count = 4;
    threw = false;
    try { sp::validate(bad); } catch (const std::invalid_argument&) { threw = true; }
    BP_CHECK(threw, "share needs one source per draft layer");
    bad = sp::g4_dflash();
    bad.ring = 0;
    threw = false;
    try { sp::validate(bad); } catch (const std::invalid_argument&) { threw = true; }
    BP_CHECK(threw, "own ring needs entries");
}

void prefix_tests() {  // hybrid extension: copy tokens fill the leading rows, the block holds the rest (clamped)
    auto c = sp::hy4_dfly2();
    c.mask = sp::BlockMask::causal;
    c.prefix_in_block = true;
    const auto w = make_world(64, 5);
    Forward f(w, c, 1.0);
    sp::ParallelBlockDrafter pbd(c, f);
    pbd.start(w.response[0], w.prompt.size());
    const std::uint32_t pre[2] = {w.response[1], w.response[2]};
    std::vector<std::uint32_t> out;
    pbd.draft(pre, 5, out);
    BP_CHECK(out.size() == 3 && out[0] == w.response[3] && out[2] == w.response[5] && pbd.passes() == 1, "prefix draft %zu", out.size());
    BP_CHECK(pbd.last_confidence().size() == 3, "confidence per proposal");
    pbd.draft({}, 0, out);
    BP_CHECK(out.empty() && pbd.passes() == 1, "k = 0 runs no pass");
    sp::ParallelBlockDrafter strict(sp::hy4_dfly2(), f);
    bool threw = false;
    try { strict.draft(pre, 3, out); } catch (const std::invalid_argument&) { threw = true; }
    BP_CHECK(threw, "a no-prefix policy rejects a prefix");
}

struct Stats { std::uint64_t steps = 0, model = 0, hybrid = 0, forwards = 0, proposals = 0; };

Stats run(const World& w, std::uint32_t n, const sp::BlockPolicy& p, double acc, bool copy, bool fixed) {
    Forward f(w, p, acc);
    sp::ParallelBlockDrafter pbd(p, f);
    sp::BlockConfig base;
    base.copy = copy;
    base.fixed_costs = fixed;
    base.clock_ms = [] { return g_clock; };
    const auto cfg = sp::parallel_block_config(p, 2.0, base);
    sp::BlockDrafter drafter(cfg, &pbd);
    for (std::uint32_t r = 1; r <= 32; ++r) drafter.seed_verify_ms(r, 10.0 + 0.4 * double(r));
    Target t(w);
    drafter.commit(w.prompt);
    std::vector<std::uint32_t> out{w.response[0]};
    drafter.commit(std::span<const std::uint32_t>(out.data(), 1));
    pbd.start(out[0], w.prompt.size());
    sp::BlockSpeculator spec(drafter, t);
    spec.start(out[0]);
    Stats s;
    while (out.size() < n) {
        sp::StepParams sp_params;
        sp_params.remaining = n - std::uint32_t(out.size());
        sp_params.context = w.prompt.size() + out.size();
        const auto forwards = f.forwards;
        const auto rec = spec.step(out, sp_params);
        BP_CHECK(rec.emitted >= 1, "no progress");
        ++s.steps;
        s.model += rec.source == sp::Source::model;
        s.hybrid += rec.copy > 0 && rec.ext > 0;
        BP_CHECK(f.forwards - forwards <= 1, "one forward per draft, got %llu", (unsigned long long)(f.forwards - forwards));
        BP_CHECK(rec.ext <= sp::max_drafts(p, rec.copy > 0 && p.prefix_in_block ? rec.copy : 0), "ext %u", rec.ext);
        BP_CHECK(t.committed + 1 == out.size(), "committed %u emitted %zu", t.committed, out.size());
        BP_CHECK(pbd.anchor_pos() == w.prompt.size() + t.committed, "drafter anchor position");
    }
    for (std::uint32_t i = 0; i < n; ++i) BP_CHECK(out[i] == w.response[i], "token %u differs", i);
    s.forwards = f.forwards;
    s.proposals = f.proposals;
    BP_CHECK(pbd.passes() == f.forwards, "passes");
    return s;
}

void controller_tests() {
    auto causal = sp::hy4_dfly2();
    causal.mask = sp::BlockMask::causal;
    causal.prefix_in_block = true;
    const sp::BlockPolicy policies[] = {sp::hy4_dfly2(), sp::ds41_dspark(), sp::g4_dflash(), causal};
    int cases = 0;
    std::uint64_t hybrids = 0;
    for (std::uint32_t pi = 0; pi < 4; ++pi)
        for (const double acc : {0.0, 0.6, 0.95})
            for (const bool copy : {false, true})
                for (const bool fixed : {false, true}) {
                    const auto w = make_world(500, 11 + pi);
                    g_clock = 0.0;
                    const auto s = run(w, 500, policies[pi], acc, copy, fixed);
                    if (acc >= 0.95 && !copy) BP_CHECK(s.model > 0 && s.steps < 250, "acc .95: %llu steps", (unsigned long long)s.steps);
                    if (pi < 3) BP_CHECK(s.hybrid == 0, "no-prefix policy must not run hybrid blocks");
                    hybrids += s.hybrid;
                    ++cases;
                }
    std::printf("controller: %d cases PASS (hybrid blocks with a prefix policy: %llu)\n", cases, (unsigned long long)hybrids);
}

void cost_tests() {
    const auto cfg = sp::parallel_block_config(sp::hy4_dfly2(), 2.5);
    BP_CHECK(cfg.chain_max_k == 5 && cfg.chain_seed_flush_ms == 2.5 && cfg.chain_seed_step_ms == 0.0 && !cfg.hybrid, "config");
    auto fixed = cfg;
    fixed.fixed_costs = true;
    sp::BlockDrafter d(fixed, nullptr);
    BP_CHECK(d.chain_cost_ms(1) == 2.5 && d.chain_cost_ms(5) == 2.5, "flat cost in k");
}

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

void pricing_tests() {
    BP_CHECK(near(sp::uniform_distinct_experts(512, 16, 1), 16.0, 1e-9), "1 row");
    const double d6 = sp::uniform_distinct_experts(512, 16, 6);
    BP_CHECK(near(d6, 88.80, 0.01) && d6 < 96.0, "6 rows %.3f", d6);
    // HY4 one node (hy4_hipbind-STATE s4 bytes): fixed 7.60 GB, experts 41 layers x 4.26 MB (2.80 GB for 16), KV ~46 MB/row.
    sp::MoeVerifyBytes m;
    m.fixed_gb = 7.60 - 0.046;
    m.per_row_gb = 0.046;
    m.expert_gb = 2.80 / (16.0 * 41.0);
    m.experts = 512; m.top_k = 16; m.moe_layers = 41;
    BP_CHECK(near(sp::verify_gb(m, 1), 10.40, 1e-6), "plain bytes %.4f", sp::verify_gb(m, 1));
    std::printf("pricing: rows | distinct experts | verify GB | break-even E (bytes, BF16 draft 2.43 GB)\n");
    for (std::uint32_t r = 1; r <= 6; ++r) {
        const double gb = sp::verify_gb(m, r);
        std::printf("  %u | %.1f | %.2f | %.2f\n", r, sp::uniform_distinct_experts(512, 16, r), gb,
                    sp::break_even_emitted(sp::verify_gb(m, 1), gb, r > 1 ? 2.43 : 0.0));
    }
    BP_CHECK(near(sp::verify_gb(m, 6), 23.37, 0.01), "6-row verify %.3f", sp::verify_gb(m, 6));
    BP_CHECK(near(sp::break_even_emitted(10.40, 23.37, 2.43), 2.481, 0.001), "break-even");
    BP_CHECK(near(sp::expected_emitted(0.75, 6), 3.2881, 1e-4) && sp::expected_emitted(0.5, 1) == 1.0, "emitted");
}

// HY4 verify seed table: bytes give the shape, the measured plain wall (57.7 ms, EXACT one node) the scale. Invariant:
// priced with the measured DFly2 survival (CPU reference), prose never pays for 3+ verify rows and copy-heavy text does.
void seed_tests() {
    const sp::MoeVerifyBytes m = sp::hy4_verify_bytes();
    BP_CHECK(near(sp::verify_gb(m, 1), 10.56, 0.01), "hy4 plain bytes %.4f (trace 10.56)", sp::verify_gb(m, 1));
    BP_CHECK(near(m.expert_gb * 1e3, 4.5896, 1e-3), "expert MB %.4f", m.expert_gb * 1e3);
    const std::vector<double> ms = sp::moe_verify_seed_ms(m, 57.7, 207.0, 8);
    BP_CHECK(ms.size() == 8U && near(ms[0], 57.7, 1e-9), "row 1 = plain wall %.4f", ms[0]);
    for (std::size_t r = 1; r < ms.size(); ++r) BP_CHECK(ms[r] > ms[r - 1], "monotone at %zu", r + 1);
    // Each extra row reads ~15 new experts of 41 layers: +12..+15 ms per row, verify(4) ~ 1.7 x plain (not ~1.1 x).
    BP_CHECK(ms[1] - ms[0] > 12.0 && ms[1] - ms[0] < 15.0, "row 2 marginal %.3f", ms[1] - ms[0]);
    BP_CHECK(ms[3] > 1.6 * ms[0] && ms[3] < 1.8 * ms[0], "verify(4) %.3f", ms[3]);
    const double prose[5] = {0.53, 0.10, 0.02, 0.0, 0.0}, golden[5] = {0.90, 0.83, 0.66, 0.45, 0.21};
    const double draft_ms = 13.0;  // DFly2 device pass prior: 2.43 GB at ~207 GB/s + launches
    const auto best = [&](const double (&run)[5], double& score) {
        std::uint32_t pick = 1;
        score = 1.0 / ms[0];
        double emitted = 1.0;
        for (std::uint32_t r = 2; r <= 6; ++r) {
            emitted += run[r - 2];
            const double s = emitted / (ms[r - 1] + draft_ms);
            if (s > score) score = s, pick = r;
        }
        return pick;
    };
    double sp_, sg = 0.0;
    const std::uint32_t rp = best(prose, sp_), rg = best(golden, sg);
    std::printf("hy4 seeds ms R1..8:");
    for (const double v : ms) std::printf(" %.1f", v);
    std::printf("\nhy4 best rows: prose %u (%.1f tok/s), golden %u (%.1f tok/s), plain %.1f\n", rp, 1e3 * sp_, rg, 1e3 * sg,
                1e3 / ms[0]);
    BP_CHECK(rp <= 2U && 1e3 * sp_ < 1.10 * 1e3 / ms[0], "prose picks %u at %.2f tok/s", rp, 1e3 * sp_);
    BP_CHECK(rg >= 4U && rg <= 6U && sg > 1.6 / ms[0], "golden picks %u at %.2f tok/s", rg, 1e3 * sg);
    bool threw = false;
    try { (void)sp::moe_verify_seed_ms(m, 40.0, 207.0, 4); } catch (const std::invalid_argument&) { threw = true; }
    BP_CHECK(threw, "bytes above the plain wall must throw");
}
}  // namespace

int main() {
    layout_tests();
    cost_tests();
    prefix_tests();
    pricing_tests();
    seed_tests();
    controller_tests();
    std::printf("spec block_parallel tests PASS\n");
    return 0;
}
