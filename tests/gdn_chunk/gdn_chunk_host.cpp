// Host oracle of the FAST chunked GDN (strix/cuda_core/gdn_chunk.hpp, gdn-chunk16-f32-v1). Invariants:
//  1. the chunk algebra equals the serial recurrence: chunk(float) vs serial(double) relative RMS of o and of the final state is
//     within the bound and of the same order as the incumbent serial(float) error (rows 1..300, partial chunks, decays from
//     ~0.05 to ~0.999, betas 0..1);
//  2. padding rows (beta 0, g 0, zero q/k/v) leave the state unchanged and emit nothing: chunk calls of any length compose with
//     each other and with serial decode rows (prefill -> decode hand-off);
//  3. workspace layout: blocks float4-aligned, workspace_floats covers every (chunk, head) block.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "gdn_ref.hpp"

namespace gc = strix::cuda_core::gdn_chunk;
using gdnref::D;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++fails;
}
template <class A, class B>
static double rel(const std::vector<A>& a, const std::vector<B>& b) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

static gdnref::Problem make(int rows, unsigned seed, float g_lo, float g_hi) {
    gdnref::Problem p;
    p.rows = rows;
    std::mt19937 rng(seed);
    std::normal_distribution<float> n01(0.f, 1.f);
    std::uniform_real_distribution<float> ug(g_lo, g_hi), ub(-3.f, 3.f);
    auto unit = [&](std::vector<float>& x, int heads, float scale) {
        x.resize(static_cast<std::size_t>(rows) * heads * D);
        for (std::size_t o = 0; o < x.size(); o += D) {
            double ss = 0;
            for (int e = 0; e < D; ++e) ss += static_cast<double>((x[o + e] = n01(rng))) * x[o + e];
            const float inv = static_cast<float>(scale / std::sqrt(ss + 1e-6));
            for (int e = 0; e < D; ++e) x[o + e] *= inv;
        }
    };
    unit(p.q, p.key_heads, 1.0f / std::sqrt(128.0f));
    unit(p.k, p.key_heads, 1.0f);
    p.v.resize(static_cast<std::size_t>(rows) * p.value_heads * D);
    for (float& x : p.v) x = n01(rng);
    p.g.resize(static_cast<std::size_t>(rows) * p.value_heads);
    p.beta.resize(p.g.size());
    for (std::size_t i = 0; i < p.g.size(); ++i) {
        p.g[i] = -std::exp(ug(rng));                        // log decay in [-exp(g_hi), -exp(g_lo)]
        p.beta[i] = 1.0f / (1.0f + std::exp(-ub(rng)));     // sigmoid(b)
    }
    return p;
}
static std::vector<float> init_state(int vh, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.f, 0.3f);
    std::vector<float> s(static_cast<std::size_t>(vh) * D * D);
    for (float& x : s) x = n(rng);
    return s;
}

int main() {
    // 3. layout
    {
        using L = gc::Layout;
        const bool al = L::wt % 4 == 0 && L::u % 4 == 0 && L::pt % 4 == 0 && L::g % 4 == 0 && L::block % 4 == 0;
        const bool cover = gc::workspace_floats(48, 4096) == gc::block_offset(255, 47, 48) + L::block &&
                           gc::workspace_floats(48, 4097) == gc::block_offset(256, 47, 48) + L::block && gc::chunks(1) == 1 &&
                           gc::chunks(16) == 1 && gc::chunks(17) == 2;
        check(al && cover && L::block == 4368, "layout: float4-aligned blocks, workspace covers every (chunk, head), 4,368 floats/block");
    }
    // 1. chunk algebra vs serial, several lengths and decay regimes
    struct Case { int rows; float glo, ghi; const char* name; };
    const Case cases[] = {{1, -7.f, -1.f, "rows 1"},          {15, -7.f, -1.f, "rows 15 (one partial chunk)"},
                          {16, -7.f, -1.f, "rows 16"},        {17, -7.f, -1.f, "rows 17"},
                          {300, -7.f, 1.f, "rows 300 mixed decays 0.07..0.999"},
                          {300, -9.f, -5.f, "rows 300 long memory (decay > 0.993)"}};
    unsigned seed = 1;
    for (const Case& cs : cases) {
        const auto p = make(cs.rows, seed++, cs.glo, cs.ghi);
        const auto s0 = init_state(p.value_heads, seed++);
        std::vector<double> sd(s0.begin(), s0.end()), od;
        std::vector<float> sf = s0, of, sc = s0, oc;
        gdnref::serial<double>(p, 0, p.rows, sd, od);
        gdnref::serial<float>(p, 0, p.rows, sf, of);
        gdnref::chunked(p, 0, p.rows, sc, oc);
        const double eo = rel(oc, od), es = rel(sc, sd), fo = rel(of, od), fs = rel(sf, sd);
        char msg[200];
        std::snprintf(msg, sizeof msg, "%s: chunk o %.2e state %.2e | serial-float o %.2e state %.2e (bound 2e-5 and <= 8x serial-float + 1e-6)",
                      cs.name, eo, es, fo, fs);
        check(eo < 2e-5 && es < 2e-5 && eo <= 8 * fo + 1e-6 && es <= 8 * fs + 1e-6, msg);
    }
    // 2. composition: chunk calls of 37 + 64 + 23 rows, then 20 serial decode rows == serial(double) over all 144 rows
    {
        const auto p = make(144, 99, -7.f, 0.5f);
        const auto s0 = init_state(p.value_heads, 98);
        std::vector<double> sd(s0.begin(), s0.end()), od;
        gdnref::serial<double>(p, 0, 144, sd, od);
        std::vector<float> sc = s0, oc;
        gdnref::chunked(p, 0, 37, sc, oc);
        gdnref::chunked(p, 37, 101, sc, oc);
        gdnref::chunked(p, 101, 124, sc, oc);
        gdnref::serial<float>(p, 124, 144, sc, oc);
        char msg[160];
        std::snprintf(msg, sizeof msg, "composition 37+64+23 chunk rows + 20 serial rows: o %.2e state %.2e (bound 2e-5)", rel(oc, od),
                      rel(sc, sd));
        check(rel(oc, od) < 2e-5 && rel(sc, sd) < 2e-5, msg);
    }
    std::printf("gdn_chunk_host: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
