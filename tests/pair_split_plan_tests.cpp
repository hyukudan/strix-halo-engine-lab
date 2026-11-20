// Host tests of strix/pair/split_plan.hpp (one stage description, per-regime split plan):
// - the dense column-split binding described as stages gives the same estimate as its hand-built tp::StepModel;
// - routed slots: balanced assignment with an always-present unit; slot rows exchanged unreduced and combined in slot
//   order on every rank == the one-rank combine, bitwise;
// - the regime decides the form (decode -> tensor, prefill windows -> layers, slow link -> single);
// - bit-changing joins are refused unless allowed.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "strix/pair/split_plan.hpp"

namespace split = strix::pair::split;
namespace tp = strix::pair::tp;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// The qwen35 dense TP binding's own model (apps/hip_qwen35_decode_main.hip before the re-binding): everything split over
// 1024 units, 4 gathers per layer + the argmax pick, latency-only trips.
static void test_dense_binding_equivalence() {
    const std::uint32_t layers = 64U;
    const double total = 13.9e9 + 0.21e9 + 64.0 * 1024.0 * 2048.0;
    for (const auto& rates : {std::vector<double>{189.5e6, 189.5e6}, std::vector<double>{213e6, 180e6}, std::vector<double>{200e6}}) {
        for (const double trip_us : {39.0, 400.0}) {
            tp::StepModel m;
            m.split_bytes = total;
            m.trips.assign(4U * layers + 1U, tp::Trip{});
            const tp::TripCost link{trip_us / 1000.0, 0.0};
            const auto hand = tp::estimate(m, rates, tp::fractions(tp::split_units(1024U, rates)), link);
            const auto stages = split::uniform_layers(layers, total / layers, 1024U, 4U, 0.0, 1U);
            const split::Costs c{rates, link, 0.0, false};
            const auto t = split::tensor(stages, split::Pass{}, c);
            const auto p = split::plan(stages, split::Pass{}, c);
            const double one = p.single.ms;
            check(std::fabs(one - hand.one_rank_ms) <= 1e-9 * hand.one_rank_ms, "dense binding: one rank == hand model");
            if (rates.size() > 1U) {
                check(t.trips == m.trips.size(), "dense binding: trips == 4 x layers + 1");
                check(std::fabs(t.compute_ms - hand.compute_ms) <= 1e-9 * hand.compute_ms, "dense binding: compute == hand model");
                check(std::fabs(t.link_ms - hand.link_ms) <= 1e-9 * hand.link_ms, "dense binding: link == hand model");
                check((t.ms < one) == hand.group_wins, "dense binding: same decision as the hand model");
                check(t.exact, "dense binding: all-gathers are exact");
            }
        }
    }
}

static void test_assign_slots() {
    const double two[2] = {243e6, 243e6};
    const auto a = split::assign_slots(16U, 4.26e6, 15.05e6, two);
    check(a.size() == 2U && a[0] == 6U && a[1] == 10U, "16 slots + shared expert: rank 0 shared + 6, rank 1 10");
    const auto b = split::assign_slots(16U, 4.26e6, 0.0, two);
    check(b[0] == 8U && b[1] == 8U, "16 slots, no extra: 8 + 8");
    const double one[1] = {243e6};
    check(split::assign_slots(16U, 4.26e6, 15.05e6, one)[0] == 16U, "one rank: all slots");
    const double three[3] = {243e6, 243e6, 243e6};
    const auto c3 = split::assign_slots(16U, 4.26e6, 15.05e6, three);
    check(c3[0] + c3[1] + c3[2] == 16U, "three ranks: every slot once");
    split::Stage s;
    s.axis = split::Axis::slots;
    s.units = 16U;
    s.pool = 512U;
    check(split::touched_slots(s, 1U) == 16U, "one row touches its 16 slots");
    const auto r4 = split::touched_slots(s, 4U);
    check(r4 >= 61U && r4 <= 64U, "4 rows touch the expected union (<= 4 x 16)");
}

// bf16-valued slot rows (the one-rank engine rounds each slot's down output to bf16 before the combine).
static float bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u = (u + 0x7FFFU + ((u >> 16) & 1U)) & 0xFFFF0000U;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
}
static void combine(const std::vector<float>& rows, const std::vector<float>& w, std::uint32_t slots, std::uint32_t width, std::vector<float>& out) {
    for (std::uint32_t c = 0; c < width; ++c) {
        float acc = 0.0f;
        for (std::uint32_t k = 0; k < slots; ++k) acc = bf16(acc + w[k] * rows[std::size_t{k} * width + c]);  // slot order
        out[c] = acc;
    }
}
static void test_slot_join_bitwise() {
    const std::uint32_t K = 16U, W = 1536U;
    std::uint32_t seed = 12345U;
    const auto rnd = [&] { seed = seed * 1664525U + 1013904223U; return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f; };
    std::vector<float> rows(std::size_t{K} * W), w(K);
    for (auto& x : rows) x = bf16(rnd() * 8.0f);
    for (auto& x : w) x = rnd() + 0.5f;
    std::vector<float> one(W), joined(W);
    combine(rows, w, K, W, one);
    // two ranks: rank 0 computes slots [0, 6), rank 1 [6, 16); each writes its rows into the exchange buffer at the slot
    // index; after the exchange every rank combines the full buffer in slot order
    const double two[2] = {243e6, 243e6};
    const auto count = split::assign_slots(K, 4.26e6, 15.05e6, two);
    std::vector<float> buf(rows.size(), 0.0f);
    std::uint32_t at = 0U;
    for (std::uint32_t r = 0; r < 2U; ++r) {
        for (std::uint32_t k = at; k < at + count[r]; ++k) std::memcpy(&buf[std::size_t{k} * W], &rows[std::size_t{k} * W], W * sizeof(float));
        at += count[r];
    }
    check(at == K, "every slot computed by exactly one rank");
    combine(buf, w, K, W, joined);
    check(std::memcmp(one.data(), joined.data(), W * sizeof(float)) == 0, "slot rows exchanged unreduced, combined in slot order == one rank, bitwise");
    // the reduce form (each rank combines its own slots, partials added) is a different summation: not the one-rank bits
    std::vector<float> p0(W), p1(W), red(W);
    std::vector<float> r0(rows.begin(), rows.begin() + std::ptrdiff_t{6} * W), r1(rows.begin() + std::ptrdiff_t{6} * W, rows.end());
    std::vector<float> w0(w.begin(), w.begin() + 6), w1(w.begin() + 6, w.end());
    combine(r0, w0, 6U, W, p0);
    combine(r1, w1, 10U, W, p1);
    std::uint32_t differ = 0U;
    for (std::uint32_t c = 0; c < W; ++c) {
        red[c] = bf16(p0[c] + p1[c]);
        differ += std::memcmp(&red[c], &one[c], sizeof(float)) != 0 ? 1U : 0U;
    }
    check(differ > 0U, "partial sums added (reduce) change bits: it must stay a FAST form");
    std::printf("slot join: unreduced == one rank; reduce differs in %u of %u columns\n", differ, W);
}

// A 42-layer latent-MoE trunk (MLA attention split by heads, routed slots, replicated routers / latents) with the shape
// of the receipts in the pair_plan STATE (class efficiencies, launch-bound ms per layer).
static std::vector<split::Stage> latent_moe_trunk() {
    using split::Axis;
    using split::Join;
    std::vector<split::Stage> s;
    const double q8 = 207.0 / 243.0, ex = 222.0 / 243.0, hr = 172.0 / 243.0;
    for (std::uint32_t l = 0; l < 42U; ++l) {
        s.push_back({.name = "attn.latent_in", .layer = l, .bytes = 0.219e9 / 42.0, .efficiency = q8, .fixed_ms = (3.9 + 4.7) / 42.0});
        s.push_back({.name = "attn.heads", .layer = l, .axis = Axis::heads, .units = 64U, .bytes = 3.129e9 / 42.0,
                     .ms_per_row = 1.8 / 42.0, .efficiency = q8, .join = Join::gather, .join_bytes_per_row = 14336.0 * 2.0});
        s.push_back({.name = "attn.o", .layer = l, .axis = Axis::columns, .units = 48U, .bytes = 1.925e9 / 42.0, .efficiency = q8,
                     .join = Join::gather, .join_bytes_per_row = 3072.0 * 4.0});
        if (l == 0U) {
            s.push_back({.name = "dense", .layer = l, .axis = Axis::columns, .units = 48U, .bytes = 0.135e9, .efficiency = q8,
                         .join = Join::gather, .join_bytes_per_row = 3072.0 * 4.0});
            continue;
        }
        s.push_back({.name = "moe.latent", .layer = l, .bytes = (0.411e9 + 0.101e9 * 4.0 / 41.0) / 41.0, .efficiency = q8});
        s.push_back({.name = "moe.router", .layer = l, .bytes = 0.258e9 / 41.0, .efficiency = hr});
        s.push_back({.name = "moe.slots", .layer = l, .axis = Axis::slots, .units = 16U, .bytes = 4.26e6, .extra_slot_bytes = 15.05e6,
                     .pool = 512U, .efficiency = ex, .join = Join::slots, .join_bytes_per_row = 1536.0 * 2.0, .extra_join_bytes = 3072.0 * 4.0});
    }
    s.push_back({.name = "head", .layer = 41U, .axis = Axis::vocab, .units = 1888U, .bytes = 0.742e9, .efficiency = hr,
                 .join = Join::pick, .join_bytes_per_row = 8.0});
    return s;
}

static void test_regimes() {
    const auto s = latent_moe_trunk();
    const split::Costs host_join{{243e6, 243e6}, {0.052, 4.0e6}, 3072.0 * 4.0 * 4.0, false};
    const auto d = split::plan(s, split::Pass{}, host_join);
    check(d.chosen.form == split::Form::tensor && d.chosen.exact, "decode, 52 us exchanges: exact tensor split wins");
    check(d.layers.ms > d.single.ms, "decode: a layer pipeline of one stream is slower than one rank");
    check(d.tensor.trips == 42U * 2U + 1U + 41U + 1U, "trips: 2 attention gathers per layer, dense gather, 41 slot exchanges, pick");
    std::printf("decode: single %.1f ms, layers %.1f ms, tensor %.1f ms (compute %.1f + %u trips %.1f), rank bytes %.2f / %.2f GB\n",
                d.single.ms, d.layers.ms, d.tensor.ms, d.tensor.compute_ms, d.tensor.trips, d.tensor.link_ms,
                d.tensor.rank_bytes[0] / 1e9, d.tensor.rank_bytes[1] / 1e9);
    const split::Costs slow{{243e6, 243e6}, {2.0, 4.0e6}, 3072.0 * 4.0 * 4.0, false};
    check(split::plan(s, split::Pass{}, slow).chosen.form == split::Form::single, "decode, 2 ms exchanges: one rank");
    // prefill: 16 windows of 1024 rows, compute-bound (0.95 ms/row for the whole trunk on one rank)
    auto pf = s;
    for (auto& st : pf)
        if (st.axis != split::Axis::replicated) st.ms_per_row += 0.95 / static_cast<double>(pf.size());
    const split::Pass win{split::Regime::prefill, 1024U, 16U};
    const auto p = split::plan(pf, win, host_join);
    check(p.chosen.form == split::Form::layers, "prefill windows: the layer flow shop wins (no per-sublayer exchange of 1024 rows)");
    check(p.layers.cut >= 18U && p.layers.cut <= 24U, "prefill: the cut balances the halves");
    std::printf("prefill 16 x 1024: single %.0f ms, layers %.0f ms (cut %u), tensor %.0f ms\n", p.single.ms, p.layers.ms, p.layers.cut, p.tensor.ms);
    // verify R rows: the split carries R rows per exchange; slot bytes grow with the union
    const auto v = split::plan(s, split::Pass{split::Regime::verify, 4U, 1U}, host_join);
    check(v.tensor.rank_bytes[1] > d.tensor.rank_bytes[1], "verify 4 rows: more distinct slots read");
    std::printf("verify 4 rows: single %.1f ms, tensor %.1f ms\n", v.single.ms, v.tensor.ms);
}

static void test_fast_refused() {
    std::vector<split::Stage> s(2);
    s[0] = {.name = "attn", .layer = 0U, .axis = split::Axis::heads, .units = 64U, .bytes = 1e8, .join = split::Join::reduce,
            .join_bytes_per_row = 12288.0};
    s[1] = {.name = "ffn", .layer = 1U, .axis = split::Axis::columns, .units = 64U, .bytes = 1e8, .join = split::Join::gather,
            .join_bytes_per_row = 12288.0};
    split::Costs c{{243e6, 243e6}, {0.01, 4e6}, 12288.0, false};
    const auto p = split::plan(s, split::Pass{}, c);
    check(!p.tensor.exact && !p.tensor.admissible && p.chosen.form != split::Form::tensor, "reduce join: refused without allow_fast");
    c.allow_fast = true;
    const auto f = split::plan(s, split::Pass{}, c);
    check(f.tensor.admissible && f.chosen.form == split::Form::tensor && !f.chosen.exact, "reduce join: admitted as FAST when allowed");
    bool threw = false;
    try {
        split::Costs bad{{243e6, -1.0}, {0.01, 4e6}, 0.0, false};
        (void)split::plan(s, split::Pass{}, bad);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "negative rate refused");
    threw = false;
    try {
        std::vector<split::Stage> z(1);
        z[0].axis = split::Axis::columns;
        z[0].units = 0U;
        (void)split::plan(z, split::Pass{}, c);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "divided stage without units refused");
}

// A measured one-rank pass: the residual over the stage model is one replicated fixed stage, carried by single AND tensor.
static void test_measured_single() {
    std::vector<split::Stage> s(3);
    s[0] = {.name = "rep", .layer = 0U, .axis = split::Axis::replicated, .units = 1U, .bytes = 0.8e9};
    s[1] = {.name = "heads", .layer = 0U, .axis = split::Axis::heads, .units = 64U, .bytes = 4.8e9, .join = split::Join::gather,
            .join_bytes_per_row = 4096.0};
    s[2] = {.name = "rows", .layer = 0U, .axis = split::Axis::columns, .units = 64U, .bytes = 4.8e9, .join = split::Join::gather,
            .join_bytes_per_row = 4096.0};
    const split::Costs c{{100e6, 100e6}, {0.0, 4e6}, 0.0, false};
    auto m = s;
    const auto cal = split::with_measured_single(m, split::Pass{}, 100e6, 124.0);
    check(std::fabs(cal.model_ms - 104.0) < 1e-9 && std::fabs(cal.fixed_ms - 20.0) < 1e-9, "residual = measured - model");
    check(m.size() == 4U && m.back().axis == split::Axis::replicated && m.back().join == split::Join::none, "one replicated stage appended");
    const auto p = split::plan(m, split::Pass{}, c);
    check(std::fabs(p.single.ms - 124.0) < 1e-9, "single form = the measured pass");
    check(std::fabs(p.tensor.compute_ms - 76.0) < 1e-9 && p.tensor.trips == 2U, "tensor form carries the same fixed ms");
    auto n = s;
    const auto neg = split::with_measured_single(n, split::Pass{}, 100e6, 90.0);
    check(neg.fixed_ms == 0.0 && std::fabs(split::plan(n, split::Pass{}, c).single.ms - 104.0) < 1e-9, "a negative residual is not priced");
    bool threw = false;
    try { (void)split::with_measured_single(n, split::Pass{}, 100e6, 0.0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a pass that was not measured is refused");
}

int main() {
    test_dense_binding_equivalence();
    test_assign_slots();
    test_slot_join_bitwise();
    test_regimes();
    test_fast_refused();
    test_measured_single();
    std::printf("pair split plan tests: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
