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
#include <functional>
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

void rescore_plan_tests() {
    namespace plan = strix::core::screened_head_plan;
    // The order key is monotone over every non-NaN float (bf16 grid, both zeros, both infinities).
    std::vector<float> grid;
    for (std::uint32_t b = 0; b < 65536U; ++b) {
        const float f = strix::weight::scalar_float::u2f(b << 16U);
        if (!std::isnan(f)) grid.push_back(f);
    }
    std::sort(grid.begin(), grid.end());
    for (std::size_t i = 1; i < grid.size(); ++i)
        if (grid[i] > grid[i - 1]) expect(plan::order_key(grid[i]) > plan::order_key(grid[i - 1]), "order key monotone");
    const float negative_nan = std::bit_cast<float>(0xffc00000U);
    const float positive_nan = std::bit_cast<float>(0x7fc12345U);
    const float signaling_nan = std::bit_cast<float>(0x7f800001U);
    const float infinity = std::numeric_limits<float>::infinity();
    const float mixed[] = {negative_nan, 1.0F, positive_nan, signaling_nan, infinity};
    const auto nan_cutoff = plan::rescore_threshold(mixed, 1U);
    unsigned selected_nans = 0U;
    for (float v : mixed) selected_nans += plan::order_key(v) >= nan_cutoff ? 1U : 0U;
    expect(plan::order_key(negative_nan) == plan::order_key(positive_nan) &&
               plan::order_key(positive_nan) == plan::order_key(signaling_nan) &&
               plan::order_key(negative_nan) > plan::order_key(infinity),
           "NaN signs and payloads share the highest key");
    expect(selected_nans == 3U, "all NaNs selected at k=1, overflowing a capacity of two", selected_nans);
    const float zeros[] = {-0.0F, 0.0F, -1.0F};
    expect(plan::order_key(zeros[0]) == plan::order_key(zeros[1]) &&
               plan::order_key(zeros[0]) >= plan::rescore_threshold(zeros, 1U),
           "signed zeros share the numeric tie at k=1");
    // The threshold selects the exact top-k plus the rows sharing the k-th key's 24-bit prefix, never fewer than k.
    std::mt19937 rng(11U);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    for (const unsigned k : {1U, 2U, 64U, 300U}) {
        for (unsigned trial = 0; trial < 4U; ++trial) {
            std::vector<float> s(5003U);
            for (auto& v : s) v = normal(rng);
            if (trial == 1U) std::fill(s.begin() + 100, s.begin() + 900, s[0]);  // a block of equal scores
            if (trial == 2U) s[17] = std::numeric_limits<float>::quiet_NaN();    // NaN: the largest key
            const auto t = plan::rescore_threshold(s, k);
            expect((t & 0xffU) == 0U, "threshold low byte clear", t);
            std::vector<std::uint32_t> keys(s.size());
            std::transform(s.begin(), s.end(), keys.begin(), [](float v) { return plan::order_key(v); });
            std::vector<std::uint32_t> sorted = keys;
            std::sort(sorted.begin(), sorted.end(), std::greater<>{});
            expect(t == (sorted[k - 1U] & ~0xffU), "threshold = k-th key prefix", t, sorted[k - 1U]);
            unsigned selected = 0U, above = 0U;
            for (const auto key : keys) {
                selected += key >= t ? 1U : 0U;
                above += key > sorted[k - 1U] ? 1U : 0U;
            }
            expect(selected >= k && above < k, "selection holds the exact top-k", selected, above);
            for (const auto key : keys)
                if (key >= t) expect((key >> 8U) >= (sorted[k - 1U] >> 8U), "selected rows share or pass the prefix");
        }
    }
    expect(plan::rescore_threshold(std::vector<float>(40U, 1.0F), 64U) == 0U, "fewer rows than k: every row");
    // Scratch layout: 256-byte parts, the cleared prefix ends where the ids start.
    const auto l = plan::rescore_layout(200058U, plan::kCapacity, 16U);
    expect(l.hist % 256U == 0U && l.ids % 256U == 0U && l.y % 256U == 0U && l.cleared == l.ids, "rescore layout parts");
    expect(l.hist >= 16U * sizeof(plan::RescoreState) && l.bytes >= l.y + 16U * 200058U * 4U, "rescore layout sizes");
    expect(plan::rescore_blocks(200058U) == 98U && plan::rescore_blocks(1U) == 1U &&
               plan::rescore_blocks(1U << 30U) == 128U,
           "rescore blocks");
    constexpr unsigned huge = std::numeric_limits<unsigned>::max();
    expect(plan::rescore_blocks(huge) == 128U, "histogram grid rounds without wrap");
    // Host storage stands in for device ranges. Put lane 1 at a scratch-aligned address.
    alignas(256) static std::uint8_t scratch[plan::rescore_layout(32U, 16U, 2U).bytes];
    alignas(256) static std::uint16_t weights[32U * 16U];
    alignas(256) static float x[128];
    alignas(256) static float scores[128];
    const plan::RescoreLaunch ok{{weights, 32U, 16U, 30.0F}, x, scores, scratch, 2U, 64U, 64U, 4U, 16U, 64U,
                                 sizeof(scratch)};
    expect(plan::rescore_launch_valid(ok), "valid rescore launch");
    auto bad = ok;
    bad.k = 0U;
    expect(!plan::rescore_launch_valid(bad), "k 0");
    bad = ok;
    bad.capacity = 3U;
    expect(!plan::rescore_launch_valid(bad), "capacity < k");
    bad = ok;
    bad.head.softcap = 0.0F;
    expect(!plan::rescore_launch_valid(bad), "softcap 0");
    bad = ok;
    bad.pitch = 31U;
    expect(!plan::rescore_launch_valid(bad), "pitch < rows");
    bad = ok;
    bad.x_stride = 6U;
    expect(!plan::rescore_launch_valid(bad), "x stride < width");
    bad = ok;
    for (unsigned width : {8U, 20U}) {
        bad.head.width = width;
        expect(!plan::rescore_launch_valid(bad), "width leaves a mapped BF16 tail", width);
    }
    bad = ok;
    bad.head.rows = huge;
    bad.pitch = huge;
    expect(!plan::rescore_launch_valid(bad), "row loops and all-row mapping cannot wrap");
    bad = ok;
    bad.pass2_blocks = huge / 128U + 1U;
    expect(!plan::rescore_launch_valid(bad), "pass-2 grid must fit the 32-bit work-item limit");
    bad = ok;
    bad.scratch = scratch + 16;
    expect(!plan::rescore_launch_valid(bad), "scratch alignment");
    bad = ok;
    --bad.scratch_bytes;
    expect(!plan::rescore_launch_valid(bad), "undersized scratch");
    for (const void* overlap : {static_cast<const void*>(scores + 64), static_cast<const void*>(x + 64),
                                static_cast<const void*>(weights)}) {
        bad = ok;
        bad.scratch = reinterpret_cast<std::uint8_t*>(const_cast<void*>(overlap));
        expect(!plan::rescore_launch_valid(bad), "scratch overlaps a live input or score lane");
    }
    bad = ok;
    bad.scratch = reinterpret_cast<std::uint8_t*>(std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{255});
    expect(!plan::rescore_launch_valid(bad), "scratch address range cannot wrap");
    // The score image is rewritten in place: it may not reach the weights or the activations. One arena so adjacency is
    // a deliberate address: weights [0, 1024), activations [1024, 1344) (lane 1 at 64 floats, 16 wide).
    alignas(256) static std::uint8_t arena[4096];
    bad = ok;
    bad.scores = reinterpret_cast<float*>(weights);
    expect(!plan::rescore_launch_valid(bad), "scores over the weights");
    bad = ok;
    bad.scores = x;
    expect(!plan::rescore_launch_valid(bad), "scores over the activations");
    auto near = ok;
    near.head.weights = reinterpret_cast<const std::uint16_t*>(arena);
    near.scores = reinterpret_cast<float*>(arena + 1024);
    expect(plan::rescore_launch_valid(near), "scores starting where the weights end");
    near.scores = reinterpret_cast<float*>(arena + 1020);
    expect(!plan::rescore_launch_valid(near), "scores on the last weight bytes");
    near = ok;
    near.x = reinterpret_cast<const float*>(arena + 1024);
    near.scores = reinterpret_cast<float*>(arena + 1344);
    expect(plan::rescore_launch_valid(near), "scores starting where the activations end");
    near.scores = reinterpret_cast<float*>(arena + 1340);
    expect(!plan::rescore_launch_valid(near), "scores on the last activation");
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

}  // namespace

int main() {
    builder_tests();
    softcap_monotone_host();
    rescore_plan_tests();
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
