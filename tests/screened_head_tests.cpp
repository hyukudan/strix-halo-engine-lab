// EXACT screened LM head (src/core/screened_head.hpp), CPU: plane builder and the bound that makes the screen exact.
// For random, adversarial and degenerate rows / activations the test emulates the incumbent head (FP32 sums in several
// summation orders, FMA and non-FMA, then bf16 + softcap and the first-max picker) and checks
//   1. every incumbent logit lies in [L_i, U_i] of the screen (the invariant);
//   2. the candidate set G(U_i) >= G(max L) contains every row tied with the true argmax, so the candidate pick equals
//      the full pick (token and status);
//   3. whenever the overflow / non-finite guard passes, every incumbent logit is finite.
// It prints the mean candidate counts (the screen's tightness) for Gaussian-like and peaked rows.
#include "core/screened_head.hpp"
#include "core/screened_head_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {

namespace sh = strix::core::screened_head;
using strix::weight::scalar_float::bf16_encode;
using strix::weight::scalar_float::bf16_round;

int failures = 0;
void expect(bool ok, const char* what, unsigned a = 0, unsigned b = 0) {
    if (ok) return;
    ++failures;
    if (failures <= 20) std::fprintf(stderr, "FAIL: %s (%u, %u)\n", what, a, b);
}

constexpr float kCap = 30.0F;

float butterfly(float* lane) {
    for (unsigned m = 16U; m > 0U; m >>= 1U) {
        float next[32];
        for (unsigned l = 0; l < 32U; ++l) next[l] = lane[l] + lane[l ^ m];
        std::memcpy(lane, next, sizeof next);
    }
    return lane[0];
}

float pairwise(const float* w, const float* x, unsigned n) {
    if (n == 1U) return w[0] * x[0];
    const unsigned h = n / 2U;
    return pairwise(w, x, h) + pairwise(w + h, x + h, n - h);
}

constexpr unsigned kOrders = 5U;
// The incumbent's dot product in several summation forms (any form is inside the bound).
float incumbent_dot(unsigned order, const float* w, const float* x, unsigned n) {
    float acc = 0.0F;
    switch (order) {
    case 0: for (unsigned k = 0; k < n; ++k) acc = std::fma(w[k], x[k], acc); return acc;
    case 1: for (unsigned k = 0; k < n; ++k) { const volatile float p = w[k] * x[k]; acc = acc + p; } return acc;
    case 2: for (unsigned k = n; k-- > 0U;) acc = std::fma(w[k], x[k], acc); return acc;
    case 3: {  // the device tight row: 32 lanes, 16-element chunks round-robin, butterfly
        float lane[32] = {};
        for (unsigned l = 0; l < 32U; ++l)
            for (unsigned c = l; c < n / 16U; c += 32U)
                for (unsigned e = 0; e < 16U; ++e) { const volatile float p = w[c * 16U + e] * x[c * 16U + e]; lane[l] = lane[l] + p; }
        return butterfly(lane);
    }
    default: return pairwise(w, x, n);
    }
}

// Pass 1 as the device computes it (unit of 8, inner FMA chain, scale FMA, lanes, butterfly).
float plane_dot(const sh::Plane& plane, unsigned row, const float* x) {
    float lane[32] = {};
    const unsigned units = plane.layout.width / sh::kUnit;
    for (unsigned l = 0; l < 32U; ++l)
        for (unsigned u = l; u < units; u += 32U) {
            float inner = 0.0F;
            for (unsigned j = 0; j < sh::kUnit; ++j) {
                const unsigned k = u * sh::kUnit + j;
                inner = std::fma(static_cast<float>(sh::plane_code(plane, row, k)), x[k], inner);
            }
            lane[l] = std::fma(sh::plane_scale(plane, row, u * sh::kUnit / sh::kGroup), inner, lane[l]);
        }
    return butterfly(lane);
}

struct Scenario {
    const char* name;
    unsigned rows, width;
    float weight_sigma, x_sigma;
    float planted_cos;    // > 0: one row is aligned with x with this cosine
    bool duplicates;      // copy a few rows (exact ties) and one-bit perturbed copies (near ties)
    bool zero_rows;
    float x_scale;        // multiplies x after generation
};

struct Stats { double max_ratio = 0; double candidates = 0; unsigned runs = 0, fallbacks = 0, guarded_finite_checks = 0; };

void run(const Scenario& sc, unsigned bits, unsigned seed, Stats& stats) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0F, 1.0F);
    const unsigned n = sc.rows, k = sc.width;
    std::vector<float> x(k);
    for (auto& v : x) v = gauss(rng) * sc.x_sigma * sc.x_scale;
    std::vector<std::uint16_t> w(std::size_t{n} * k);
    for (auto& v : w) v = bf16_encode(gauss(rng) * sc.weight_sigma);
    if (sc.planted_cos > 0.0F) {
        double xn = 0.0, wn = 0.0;
        for (float v : x) xn += static_cast<double>(v) * v;
        for (unsigned e = 0; e < k; ++e) wn += std::pow(strix::weight::scalar_float::u2f(std::uint32_t{w[5U * k + e]} << 16U), 2.0);
        const float a = static_cast<float>(sc.planted_cos * std::sqrt(wn) / std::sqrt(xn));
        for (unsigned e = 0; e < k; ++e)
            w[7U * k + e] = bf16_encode(strix::weight::scalar_float::u2f(std::uint32_t{w[5U * k + e]} << 16U) * std::sqrt(1.0F - sc.planted_cos * sc.planted_cos) +
                                          a * x[e] / std::max(sc.x_scale, 1e-30F) * sc.x_scale);
    }
    if (sc.duplicates) {
        for (unsigned i = 0; i < 6U && 20U + 3U * i < n; ++i) {
            std::copy_n(w.begin() + std::size_t{7U} * k, k, w.begin() + std::size_t{20U + 3U * i} * k);       // exact ties
            std::copy_n(w.begin() + std::size_t{7U} * k, k, w.begin() + std::size_t{21U + 3U * i} * k);
            w[std::size_t{21U + 3U * i} * k + (13U * i) % k] ^= 1U;                                              // near tie
        }
    }
    if (sc.zero_rows)
        for (unsigned i = 40U; i < 44U && i < n; ++i) std::fill_n(w.begin() + std::size_t{i} * k, k, std::uint16_t{0});

    const auto plane = sh::build_plane(w, k, n, bits);
    expect(plane.usable, "plane usable");
    const auto bound = sh::make_bound(k);
    const float nx = sh::inflated_norm(x, bound);
    const bool guard = sh::guard_ok(nx, plane.max_amplitude);

    std::vector<float> row(k), y(n), yhat(n), g(n), lo(n), hi(n);
    std::vector<float> wf(k);
    ++stats.runs;
    for (unsigned order = 0; order < kOrders; ++order) {
        for (unsigned i = 0; i < n; ++i) {
            for (unsigned e = 0; e < k; ++e) wf[e] = strix::weight::scalar_float::u2f(std::uint32_t{w[std::size_t{i} * k + e]} << 16U);
            y[i] = incumbent_dot(order, wf.data(), x.data(), k);
            g[i] = sh::softcap_g(y[i], kCap);
        }
        const auto full = sh::pick_reference(g);
        if (guard) {
            ++stats.guarded_finite_checks;
            for (unsigned i = 0; i < n; ++i) expect(std::isfinite(y[i]) && std::isfinite(g[i]), "guard passed but logit non-finite", i, order);
            expect(full.status == 0U, "guard passed but status != 0", full.status, order);
        } else {
            if (order == 0U) ++stats.fallbacks;
            continue;  // the device runs the incumbent on all rows
        }
        float lstar = -std::numeric_limits<float>::infinity();
        for (unsigned i = 0; i < n; ++i) {
            yhat[i] = plane_dot(plane, i, x.data());
            const float e = sh::error_bound(nx, plane.residual[i], plane.amplitude[i], bound);
            sh::interval(yhat[i], e, lo[i], hi[i]);
            expect(lo[i] <= y[i] && y[i] <= hi[i], "incumbent logit outside [L, U]", i, order);
            if (e > 0.0F) stats.max_ratio = std::max(stats.max_ratio, static_cast<double>(std::fabs(y[i] - yhat[i])) / e);
            lstar = std::max(lstar, lo[i]);
        }
        const float threshold = sh::softcap_g(lstar, kCap);
        std::vector<std::uint32_t> ids;
        std::vector<float> cand;
        for (unsigned i = 0; i < n; ++i)
            if (sh::softcap_g(hi[i], kCap) >= threshold) { ids.push_back(i); cand.push_back(g[i]); }
        const auto screened = sh::pick_reference(cand, ids);
        expect(screened.token == full.token && screened.status == full.status, "candidate pick differs", screened.token, full.token);
        expect(std::find(ids.begin(), ids.end(), full.token) != ids.end(), "argmax not a candidate", full.token, order);
        for (unsigned i = 0; i < n; ++i)  // every row tied with the max is a candidate
            if (g[i] == g[full.token] && std::find(ids.begin(), ids.end(), i) == ids.end()) expect(false, "tied row dropped", i, order);
        if (order == 0U) stats.candidates += static_cast<double>(ids.size());
    }
}

void softcap_monotone_host() {
    for (const float cap : {30.0F, 1.0F, 0.37F, 100.0F}) {
        std::vector<std::pair<float, float>> v;
        for (std::uint32_t b = 0; b < 65536U; ++b) {
            const float f = strix::weight::scalar_float::u2f(b << 16U);
            if (!std::isfinite(f)) continue;
            v.emplace_back(f, sh::softcap_g(f, cap));
        }
        std::sort(v.begin(), v.end());
        for (std::size_t i = 1; i < v.size(); ++i)
            if (v[i].first > v[i - 1].first) expect(v[i].second >= v[i - 1].second, "host softcap not monotone", static_cast<unsigned>(i));
    }
}

void builder_tests() {
    expect(sh::plane_layout(2304U, 4U).row_bytes == 1296U, "layout 2304x4");
    expect(sh::plane_layout(2304U, 8U).row_bytes == 2448U, "layout 2304x8");
    expect(sh::plane_layout(4096U, 8U).row_bytes % 2048U != 0U, "pitch never a 2 KiB multiple");
    for (unsigned width : {32U, 64U, 128U, 256U, 1024U, 2048U, 4096U, 6144U, 8192U})
        for (unsigned bits : {4U, 8U}) {
            const auto l = sh::plane_layout(width, bits);
            expect(l.row_bytes % 16U == 0U && l.row_bytes % 2048U != 0U, "layout pitch rule", width, bits);
        }
    expect(!sh::layout_supported(100U, 4U) && !sh::layout_supported(64U, 5U) && !sh::layout_supported(0U, 4U), "unsupported shapes");
    std::mt19937 rng(3);
    std::normal_distribution<float> gauss(0.0F, 0.05F);
    const unsigned rows = 50U, k = 96U;
    std::vector<std::uint16_t> w(std::size_t{rows} * k);
    for (auto& v : w) v = bf16_encode(gauss(rng));
    for (unsigned bits : {4U, 8U}) {
        const auto plane = sh::build_plane(w, k, rows, bits);
        expect(plane.usable && plane.rows == rows && plane.data.size() == rows * plane.layout.row_bytes, "plane shape");
        for (unsigned r = 0; r < rows; ++r) {
            long double d2 = 0.0L, w2 = 0.0L, h2 = 0.0L;
            for (unsigned e = 0; e < k; ++e) {
                const float wf = strix::weight::scalar_float::u2f(std::uint32_t{w[std::size_t{r} * k + e]} << 16U);
                const float wh = sh::plane_element(plane, r, e);
                d2 += static_cast<long double>(wf - wh) * (wf - wh);
                w2 += static_cast<long double>(wf) * wf;
                h2 += static_cast<long double>(wh) * wh;
                const float s = sh::plane_scale(plane, r, e / sh::kGroup);
                expect(std::fabs(wf - wh) <= 1.0001F * s, "code error beyond one step", r, e);
            }
            expect(static_cast<long double>(plane.residual[r]) >= std::sqrt(d2), "R rounded up", r, bits);
            expect(static_cast<long double>(plane.amplitude[r]) >= std::sqrt(w2) + std::sqrt(h2), "A rounded up", r, bits);
        }
        for (unsigned threads : {2U, 3U, 7U, 64U}) {  // rows built concurrently: the same bytes and norms
            const auto split = sh::build_plane(w, k, rows, bits, threads);
            expect(split.usable && split.data == plane.data && split.residual == plane.residual &&
                       split.amplitude == plane.amplitude && split.max_amplitude == plane.max_amplitude,
                   "threaded plane == one-thread plane", threads, bits);
        }
    }
    auto bad = w;
    bad[17] = 0x7f80U;  // +inf
    expect(!sh::build_plane(bad, k, rows, 4U).usable, "non-finite weight -> plane unusable");
    expect(!sh::build_plane(bad, k, rows, 4U, 3U).usable, "non-finite weight -> plane unusable (threads)");
    bad[17] = 0x7fc0U;  // NaN
    expect(!sh::build_plane(bad, k, rows, 8U).usable, "NaN weight -> plane unusable");
    expect(!sh::build_plane(w, 100U, rows, 4U).usable, "unsupported width -> unusable");
    // The picker reference: ties to the lowest token, zeros equal, non-finite flagged.
    const float logits[] = {1.0F, 3.0F, 3.0F, -0.0F, 3.0F};
    expect(sh::pick_reference(logits).token == 1U && sh::pick_reference(logits).status == 0U, "tie -> lowest token");
    const float zeros[] = {0.0F, -0.0F};
    expect(sh::pick_reference(zeros).token == 0U, "zeros are equal");
    const float nanv[] = {std::numeric_limits<float>::quiet_NaN(), 2.0F};
    expect(sh::pick_reference(nanv).token == 1U && sh::pick_reference(nanv).status == 1U, "non-finite sets bit0");
    // Guard: non-finite x and overflow-prone x fall back.
    const auto b = sh::make_bound(k);
    std::vector<float> xs(k, 1.0F);
    expect(sh::guard_ok(sh::inflated_norm(xs, b), 1.0F), "guard passes on a unit-scale activation");
    xs[3] = std::numeric_limits<float>::quiet_NaN();
    expect(!sh::guard_ok(sh::inflated_norm(xs, b), 1.0F), "NaN x -> fallback");
    xs[3] = std::numeric_limits<float>::infinity();
    expect(!sh::guard_ok(sh::inflated_norm(xs, b), 1.0F), "inf x -> fallback");
    std::fill(xs.begin(), xs.end(), 1.0e19F);
    expect(!sh::guard_ok(sh::inflated_norm(xs, b), 1.0F), "x^2 overflow -> fallback");
    std::fill(xs.begin(), xs.end(), 1.0e-30F);
    expect(sh::guard_ok(sh::inflated_norm(xs, b), 1.0F) && sh::inflated_norm(xs, b) > 1.0e-30F * 9.0F, "tiny x keeps a sound norm floor");
}

// The pass-1 schedule (screened_head_plan: row tiles, one-wave grid, lane groups folded into the block index) computes
// every (row, lane) bound exactly once, for every lane count the resident screen admits, any grid cap and any resident
// count, including clipped tail tiles. The device bodies walk exactly this loop.
void pass1_schedule_tests() {
    namespace plan = strix::core::screened_head_plan;
    expect(plan::pass1_tile_rows(1U) == 1U && plan::pass1_tile_rows(2U) == 1U && plan::pass1_tile_rows(4U) == 2U &&
               plan::pass1_tile_rows(8U) == 4U,
           "tile rows 1, 1, 2, 4 for 1, 2, 4, 8 lanes per group");
    expect(plan::pass1_blocks(1504U, 0U, 1U) == 1504U && plan::pass1_blocks(1504U, 1504U, 1U) == 1504U,
           "one lane keeps the owner's grid");
    expect(plan::pass1_blocks(1504U, 376U, 1U) == 376U, "eight lanes: one wave of resident blocks");
    expect(plan::pass1_blocks(1504U, 375U, 4U) == 372U && plan::pass1_blocks(1504U, 3U, 4U) == 4U &&
               plan::pass1_blocks(2U, 0U, 4U) == 4U,
           "whole lane groups, at least one block per group");
    const unsigned row_counts[] = {1U, 3U, 7U, 64U, 1001U, 20011U};
    const unsigned residents[] = {0U, 3U, 376U, 564U, 1504U};
    const std::size_t limits[] = {48U * 1024U, 227U * 1024U};
    for (const std::size_t limit : limits)
        for (unsigned lanes = 1U; lanes <= plan::kMaxLanes; ++lanes) {
            const unsigned per_group = plan::pass1_lanes(2304U, lanes, limit);
            const unsigned groups = plan::pass1_groups(lanes, per_group);
            expect(per_group != 0U && plan::pass1_lanes_valid(per_group, lanes), "lane form admitted", lanes, per_group);
            expect(groups * per_group >= lanes && (groups - 1U) * per_group < lanes, "groups cover the lanes", lanes,
                   groups);
            const unsigned tile_rows = plan::pass1_tile_rows(per_group);
            for (const unsigned resident : residents)
                for (const unsigned rows : row_counts) {
                    const unsigned blocks = plan::pass1_blocks(1504U, resident, groups);
                    expect(blocks % groups == 0U && blocks >= groups, "blocks are whole groups", blocks, groups);
                    std::vector<unsigned> seen(std::size_t{rows} * groups, 0U);
                    const unsigned tiles = (rows + tile_rows - 1U) / tile_rows;
                    for (unsigned block = 0; block < blocks; ++block) {
                        const auto map = plan::pass1_map(block, blocks, groups);
                        for (unsigned warp = 0; warp < plan::kPass1Waves; ++warp)
                            for (unsigned tile = plan::pass1_first_tile(map, warp); tile < tiles;
                                 tile += plan::pass1_tile_step(map))
                                for (unsigned r = 0; r < tile_rows && tile * tile_rows + r < rows; ++r)
                                    ++seen[std::size_t{tile * tile_rows + r} * groups + map.group];
                    }
                    const bool once = std::all_of(seen.begin(), seen.end(), [](unsigned v) { return v == 1U; });
                    expect(once, "every (row, lane group) bound computed exactly once", rows, lanes);
                }
        }
    std::puts("screened-head pass1 schedule: lanes 1..32, tiles 1/2/4, one-wave grids, folded groups ok");
}

}  // namespace

int main() {
    builder_tests();
    pass1_schedule_tests();
    softcap_monotone_host();
    const Scenario scenarios[] = {
        {"gaussian", 1500, 256, 0.02F, 1.0F, 0.0F, false, false, 1.0F},
        {"gaussian-k2304", 1200, 2304, 0.02F, 1.0F, 0.0F, false, false, 1.0F},
        {"peaked cos0.30", 1500, 2304, 0.02F, 1.0F, 0.30F, false, false, 1.0F},
        {"peaked cos0.15", 1500, 2304, 0.02F, 1.0F, 0.15F, false, false, 1.0F},
        {"peaked cos0.08", 1500, 2304, 0.02F, 1.0F, 0.08F, false, false, 1.0F},
        {"ties+near ties", 800, 256, 0.02F, 1.0F, 0.2F, true, true, 1.0F},
        {"softcap saturation", 800, 256, 2.0F, 8.0F, 0.0F, true, false, 1.0F},
        {"tiny x", 600, 256, 0.02F, 1.0F, 0.0F, false, false, 1.0e-12F},
        {"subnormal-scale x", 600, 256, 0.02F, 1.0F, 0.0F, false, false, 1.0e-37F},
        {"x^2 overflow (guard)", 600, 256, 0.02F, 1.0F, 0.0F, false, false, 1.0e25F},
        {"logit overflow (guard)", 600, 256, 1.0e19F, 1.0e19F, 0.0F, false, false, 1.0F},
        {"large weights", 600, 256, 1.0e6F, 1.0e6F, 0.0F, false, false, 1.0F},
    };
    for (const auto& sc : scenarios)
        for (const unsigned bits : {4U, 8U}) {
            Stats stats;
            const unsigned trials = sc.width > 1000U ? 3U : 6U;
            for (unsigned t = 0; t < trials; ++t) run(sc, bits, 1000U * t + bits, stats);
            const unsigned used = stats.runs - stats.fallbacks;
            std::printf("screened-head %-20s K=%-4u rows=%-4u bits=%u: mean candidates %.1f of %u (%u runs, %u fallback), max |y-yhat|/e %.3g\n", sc.name,
                        sc.width, sc.rows, bits, used ? stats.candidates / used : -1.0, sc.rows, stats.runs, stats.fallbacks, stats.max_ratio);
        }
    if (failures != 0) {
        std::fprintf(stderr, "screened_head_tests: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("screened_head_tests: ok");
    return 0;
}
