// No-model check + timing of attention_decode_prepared_split_bf16 against the prepared flash step path (rows = 1, device
// position) on the Inkling 2 Mini decode shape (24 query heads, 8 KV heads, d128, relative bias extent 1024, sinks).
// Per case: both outputs vs a double reference of the same contract (FP32 Q, unrounded P), split determinism over two
// launches (also proves the ticket reset), then microseconds per launch (cudaEvent, `--iters`). Prints RESULT PASS|FAIL.
// bits / flash_bits / rows: output digests of the split, the one-row flash and host-position flash blocks of 16 / 17 / 128 /
// 129 rows (both sides of the 16-row warp slice and of the 128-row query tile), for comparing two builds case by case.
// Every launch is checked (a failed launch exits rc 2); outputs start as a sentinel (again before the second split
// launch) and every live element must be written and finite;
// the blocks are checked against the double reference on sampled rows and must repeat bitwise after re-poisoning. --inject-unwritten skips the 17-row launch (negative
// control: the run must FAIL).
// The one-row flash also runs with the swz3 pipe (three stages, prefetch distance 2): its digest must equal the pad2 one
// (layouts change addresses, never arithmetic) and TIME reports both pipes with GB/s of the unique K/V bytes walked.
// --inject-stale skips the second split launch and --inject-nonfinite stores the same infinity in every one-row output:
// both are negative controls of the overwrite and finite checks (the run must FAIL).
// Before any CUDA call every launch of the run is checked against the engine's host admission predicates (ARGS line; a
// refused launch exits rc 3 with no device work). --check-args stops after that check: a kit dry run proves the arguments
// without touching the GPU.
// Usage: cuda_core_decode_split_bench [--iters N] [--check-only] [--check-args] [--inject-unwritten] [--inject-stale]
//        [--inject-nonfinite] [--inject-stale-rows] (skip only the repeated 17-row launch; must FAIL)
#include "strix/cuda_core/attention.hpp"
#include "strix/attention_split_plan.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace cc = strix::cuda_core;

namespace {

constexpr unsigned kHeads = 24, kKv = 8, kDim = 128, kExtent = 1024;
constexpr float kScale = 0.08838834764831845f;
bool g_fail = false;
bool g_inject_unwritten = false;
bool g_inject_stale_rows = false;  // negative control: only the second 17-row launch is skipped
bool g_inject_stale = false;     // negative control: the second split launch is skipped
bool g_inject_nonfinite = false;  // negative control: the same infinity in every one-row output
constexpr std::uint32_t kSentinel = 0xffffffffU;  // a NaN no kernel stores

// NaN-aware running maximum (std::max keeps its left side against a NaN).
void raise_max(double& m, double v) { m = (v != v || m != m) ? v + m : std::max(m, v); }

// Written elements that are not finite (the sentinel is counted by unwritten): an infinity or NaN in both builds
// would otherwise pass byte equality and an infinite tolerance.
std::size_t nonfinite(const std::vector<float>& v, std::size_t live) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < live; ++i) {
        std::uint32_t u = 0;
        std::memcpy(&u, &v[i], sizeof u);
        n += u != kSentinel && !std::isfinite(v[i]) ? 1U : 0U;
    }
    return n;
}

std::size_t unwritten(const std::vector<float>& v, std::size_t live) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < live; ++i) {
        std::uint32_t u = 0;
        std::memcpy(&u, &v[i], sizeof u);
        n += u == kSentinel ? 1U : 0U;
    }
    return n;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <class T>
struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { check(cudaMalloc(&p, count * sizeof(T) + 16), "malloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void poison() { check(cudaMemset(p, 0xff, n * sizeof(T)), "sentinel"); }
    void put(const std::vector<T>& h) { check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d"); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        check(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
        return h;
    }
};

float bf2f(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}
std::uint16_t f2bf(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    u += 0x7fffU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

struct Case {
    const char* name;
    unsigned capacity, window, position;
    bool wrapped;
};

// sliding visits (window 1024, wrapped ring) and full visits (prefix, unwrapped) at 0.1K / 1K / 2.5K / 8K / 16K
constexpr Case kCases[] = {
    {"sliding", 1280, 1024, 37, true},   {"sliding", 1280, 1024, 1023, true}, {"sliding", 1280, 1024, 2520, true},
    {"sliding", 1280, 1024, 8191, true}, {"full", 16384, 0, 37, false},      {"full", 16384, 0, 100, false},
    {"full", 16384, 0, 255, false},      {"full", 16384, 0, 256, false},     {"full", 16384, 0, 511, false},
    {"full", 16384, 0, 1023, false},     {"full", 16384, 0, 2520, false},    {"full", 16384, 0, 8191, false},
    {"full", 16384, 0, 16383, false},
    // position-canonical plan: aligned sliding row (4 chunks), short window straddling a unit (one pass from first),
    // past 16K keys (512-key aligned chunks)
    {"sliding", 1280, 1024, 2559, true}, {"short", 256, 200, 3000, true}, {"full", 20480, 0, 20000, false},
};
// host-position flash blocks: both sides of the 16-row warp slice and of the 128-row query tile
constexpr unsigned kBlockRows[] = {16U, 17U, 128U, 129U};
bool block_fits(const Case& c, unsigned rows) { return !c.wrapped || c.window + rows - 1U <= c.capacity; }
unsigned block_pos0(const Case& c, unsigned rows) { return c.position + 1U >= rows ? c.position + 1U - rows : 0U; }

// Plain prompt flash (Qwen's prefill form) serves head_dim cc::kPrefillFlashHeadDim only: its own geometry, 24 q / 8 kv heads.
constexpr unsigned kPlainCapacity = 2048U, kPlainMaxRows = 1000U;
constexpr float kPlainScale = 0.0625f;  // 1 / sqrt(256)
struct Plain { unsigned pos0, rows; };
constexpr Plain kPlainCases[] = {{0U, 17U}, {0U, 129U}, {700U, 1000U}, {1023U, 300U}};
cc::AttentionDims plain_dims() { return {kHeads, kKv, cc::kPrefillFlashHeadDim, kPlainScale, kPlainCapacity}; }

// Every launch of the run against the engine's host admission predicates, with aligned distinct host stand-ins for the device
// pointers (the predicates read only alignment, nullness and aliasing). No CUDA call. Returns the refused launch count.
unsigned check_args(unsigned& launches) {
    alignas(16) static float q[4], out[4], bias[4], sinks[4];
    alignas(16) static std::uint16_t kc[8], vc[8];
    alignas(16) static std::uint32_t pos[4];
    alignas(16) static unsigned char ws[16];
    const auto* k = reinterpret_cast<const __nv_bfloat16*>(kc);
    const auto* v = reinterpret_cast<const __nv_bfloat16*>(vc);
    unsigned refused = 0;
    const auto admit = [&](bool ok, const char* what, unsigned a, unsigned b) {
        ++launches;
        if (!ok) {
            ++refused;
            std::printf("ARGS FAIL %s %u %u\n", what, a, b);
        }
    };
    for (const auto& c : kCases) {
        const cc::AttentionDims dims{kHeads, kKv, kDim, kScale, c.capacity};
        const cc::PreparedBf16Flash prepared{bias, sinks, kExtent, c.window, c.wrapped, false};
        for (const auto pipe : {cc::FlashPipe::pad2, cc::FlashPipe::swz3})
            admit(cc::can_launch_prepared_bf16_flash(dims, prepared, q, k, v, c.position, 1U, out, pipe), "flash", c.position, 1U);
        const std::size_t ws_bytes = cc::decode_split_workspace_bytes(dims, c.window);
        admit(cc::can_launch_prepared_decode_split(dims, prepared, q, k, v, pos, out, ws, ws_bytes), "split", c.position, c.window);
        for (const unsigned rows : kBlockRows)
            if (block_fits(c, rows))
                admit(cc::can_launch_prepared_bf16_flash(dims, prepared, q, k, v, block_pos0(c, rows), rows, out,
                                                         cc::FlashPipe::pad2), "flash rows", c.position, rows);
    }
    for (const auto& p : kPlainCases)
        for (int tail = 0; tail < 2; ++tail) admit(cc::can_launch_prefill_flash(plain_dims(), p.pos0, p.rows), "plain", p.pos0, p.rows);
    // control: the predicate refuses the prepared forms' head_dim on the plain flash (the argument t3 passed)
    admit(!cc::can_launch_prefill_flash({kHeads, kKv, kDim, kScale, kPlainCapacity}, 0U, 17U), "plain control d128", 0U, 17U);
    // controls: no query head (a zero-job grid), and a row range whose 32-bit tile ceiling would wrap to zero
    admit(!cc::can_launch_prefill_flash({0U, 1U, cc::kPrefillFlashHeadDim, kScale, 1U}, 0U, 1U), "plain control heads 0", 0U, 1U);
    constexpr std::uint32_t kAll = 0xFFFFFFFFu;
    admit(!cc::can_launch_prefill_flash({1U, 1U, cc::kPrefillFlashHeadDim, kScale, kAll}, 0U, kAll), "plain control rows 2^32-1", 0U,
          kAll);
    admit(!cc::can_launch_prefill_flash({1U, 1U, cc::kPrefillFlashHeadDim, kScale, kAll}, kAll - 100U, 50U), "plain control key ceiling",
          kAll - 100U, 50U);
    return refused;
}

void run_case(const Case& c, int iters, bool check_only) {
    std::mt19937 rng(1234U + c.position);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    const std::size_t cache_elems = static_cast<std::size_t>(c.capacity) * kKv * kDim;
    std::vector<std::uint16_t> hk(cache_elems), hv(cache_elems);
    for (auto& x : hk) x = f2bf(uni(rng) * 2.0f);
    for (auto& x : hv) x = f2bf(uni(rng));
    std::vector<float> hq(kHeads * kDim), hb(kHeads * kExtent), hs(kHeads);
    for (auto& x : hq) x = uni(rng) * 2.0f;
    for (auto& x : hb) x = uni(rng);
    for (auto& x : hs) x = uni(rng) * 2.0f;

    Dev<std::uint16_t> dk(cache_elems), dv(cache_elems);
    Dev<float> dq(hq.size()), db(hb.size()), ds(hs.size()), out_flash(kHeads * kDim), out_split(kHeads * kDim),
        out_swz(kHeads * kDim);
    Dev<std::uint32_t> dpos(1);
    dk.put(hk);
    dv.put(hv);
    dq.put(hq);
    db.put(hb);
    ds.put(hs);
    dpos.put({c.position});

    const cc::AttentionDims dims{kHeads, kKv, kDim, kScale, c.capacity};
    const cc::PreparedBf16Flash prepared{db.p, ds.p, kExtent, c.window, c.wrapped, false};
    const std::size_t ws_bytes = cc::decode_split_workspace_bytes(dims, c.window);
    Dev<unsigned char> ws(ws_bytes);
    check(cc::decode_split_workspace_init(ws.p, dims, c.window, nullptr), "ws init");
    const auto* kb = reinterpret_cast<const __nv_bfloat16*>(dk.p);
    const auto* vb = reinterpret_cast<const __nv_bfloat16*>(dv.p);
    out_flash.poison();
    out_split.poison();
    out_swz.poison();
    const auto flash = [&] {
        return cc::attention_prefill_prepared_bf16(dims, prepared, dq.p, kb, vb, c.position, 1, out_flash.p, nullptr,
                                                   cc::FlashPipe::pad2, dpos.p);
    };
    const auto flash_swz = [&] {
        return cc::attention_prefill_prepared_bf16(dims, prepared, dq.p, kb, vb, c.position, 1, out_swz.p, nullptr,
                                                   cc::FlashPipe::swz3, dpos.p);
    };
    const auto split = [&] {
        return cc::attention_decode_prepared_split_bf16(dims, prepared, dq.p, kb, vb, dpos.p, out_split.p, ws.p, ws_bytes,
                                                        nullptr);
    };
    check(flash(), "flash");
    check(flash_swz(), "flash swz3");
    check(split(), "split 1");
    check(cudaDeviceSynchronize(), "sync");
    auto first = out_split.get();
    out_split.poison();  // the second launch must overwrite every element, not inherit the first launch's output
    if (!g_inject_stale) check(split(), "split 2");
    check(cudaDeviceSynchronize(), "sync");
    auto second = out_split.get();
    const bool det = std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0;
    auto fl = out_flash.get();
    auto fz = out_swz.get();
    const bool pipes_equal = std::memcmp(fl.data(), fz.data(), fl.size() * sizeof(float)) == 0;
    if (g_inject_nonfinite)
        for (auto* v : {&first, &second, &fl, &fz}) (*v)[0] = std::numeric_limits<float>::infinity();

    // double reference
    const unsigned lo = c.window && c.position >= c.window ? c.position - c.window + 1U : 0U;
    double err_split = 0, err_flash = 0, scale_ref = 0;
    for (unsigned h = 0; h < kHeads; ++h) {
        const unsigned kvh = h / (kHeads / kKv);
        std::vector<double> sc(c.position - lo + 1U);
        double mx = hs[h];
        for (unsigned key = lo; key <= c.position; ++key) {
            const unsigned slot = c.wrapped ? key % c.capacity : key;
            const std::uint16_t* kr = hk.data() + (static_cast<std::size_t>(slot) * kKv + kvh) * kDim;
            double dot = 0;
            for (unsigned i = 0; i < kDim; ++i) dot += static_cast<double>(hq[h * kDim + i]) * bf2f(kr[i]);
            const unsigned dist = c.position - key;
            double s = dot * kScale + (dist < kExtent ? hb[h * kExtent + dist] : 0.0);
            sc[key - lo] = s;
            mx = std::max(mx, s);
        }
        double den = std::exp(hs[h] - mx);
        std::vector<double> o(kDim, 0.0);
        for (unsigned key = lo; key <= c.position; ++key) {
            const unsigned slot = c.wrapped ? key % c.capacity : key;
            const std::uint16_t* vr = hv.data() + (static_cast<std::size_t>(slot) * kKv + kvh) * kDim;
            const double p = std::exp(sc[key - lo] - mx);
            den += p;
            for (unsigned i = 0; i < kDim; ++i) o[i] += p * bf2f(vr[i]);
        }
        for (unsigned i = 0; i < kDim; ++i) {
            const double ref = o[i] / den;
            scale_ref = std::max(scale_ref, std::fabs(ref));
            raise_max(err_split, std::fabs(first[h * kDim + i] - ref));
            raise_max(err_flash, std::fabs(fl[h * kDim + i] - ref));
        }
    }
    // FP32 split: bound 1e-4 of the output scale; it must not be worse than the BF16-operand flash.
    const double bound = 1e-4 * std::max(scale_ref, 1e-3);
    const std::size_t miss1 = unwritten(fl, fl.size()) + unwritten(first, first.size()) + unwritten(second, second.size());
    std::size_t bad = nonfinite(fl, fl.size()) + nonfinite(fz, fz.size()) + nonfinite(first, first.size()) +
                      nonfinite(second, second.size());
    bool ok = det && pipes_equal && unwritten(fz, fz.size()) == 0 && miss1 == 0 && bad == 0 && std::isfinite(err_flash) &&
              std::isfinite(err_split) && err_split <= bound && err_split <= std::max(err_flash, bound);
    const auto plan = strix::attn_split::decode_split_plan(c.position, c.window);
    // FNV-1a of output bits: compares two builds of the kernels case by case (schedule changes must keep them)
    const auto fnv = [](const std::vector<float>& v) {
        unsigned long long bits = 1469598103934665603ULL;
        for (const float f : v) {
            std::uint32_t u = 0;
            std::memcpy(&u, &f, sizeof u);
            for (unsigned b = 0; b < 4U; ++b) bits = (bits ^ ((u >> (8U * b)) & 0xffU)) * 1099511628211ULL;
        }
        return bits;
    };
    // The same flash at host positions over blocks of 16 / 17 / 128 / 129 rows ending at this case's row (a prompt's tiles):
    // per-row relative bias and sinks; 16 | 17 straddle the first 16-row warp slice, 128 | 129 the 128-row query tile.
    // A block the cache cannot hold (wrapped ring shorter than window + rows - 1) is skipped in both builds alike.
    char rows_text[200] = {0};
    std::size_t rows_text_len = 0;
    double err_rows = 0, scale_rows = 0;
    std::size_t miss_rows = 0;
    bool rows_det = true;
    for (const unsigned rows : kBlockRows) {
        unsigned long long rows_bits = 0ULL;
        const bool fits = block_fits(c, rows);
        if (fits) {
            const unsigned pos0 = block_pos0(c, rows);
            const std::size_t live = static_cast<std::size_t>(rows) * kHeads * kDim;
            std::vector<float> hqr(live), hbr(static_cast<std::size_t>(rows) * kHeads * kExtent),
                hsr(static_cast<std::size_t>(rows) * kHeads);
            for (auto& x : hqr) x = uni(rng) * 2.0f;
            for (auto& x : hbr) x = uni(rng);
            for (auto& x : hsr) x = uni(rng) * 2.0f;
            Dev<float> dqr(hqr.size()), dbr(hbr.size()), dsr(hsr.size()), outr(live);
            dqr.put(hqr);
            dbr.put(hbr);
            dsr.put(hsr);
            outr.poison();
            const cc::PreparedBf16Flash prepared_rows{dbr.p, dsr.p, kExtent, c.window, c.wrapped, false};
            const auto launch_rows = [&](bool repeat) {
                if (!(rows == 17U && (g_inject_unwritten || (repeat && g_inject_stale_rows))))
                    check(cc::attention_prefill_prepared_bf16(dims, prepared_rows, dqr.p, kb, vb, pos0, rows, outr.p,
                                                             nullptr, cc::FlashPipe::pad2, nullptr), "flash rows");
                check(cudaDeviceSynchronize(), "sync rows");
            };
            launch_rows(false);
            const auto out = outr.get();
            outr.poison();
            launch_rows(true);
            const auto repeat = outr.get();
            miss_rows += unwritten(out, live) + unwritten(repeat, live);
            bad += nonfinite(out, live) + nonfinite(repeat, live);
            rows_det = rows_det && std::memcmp(out.data(), repeat.data(), live * sizeof(float)) == 0;
            rows_bits = fnv(out);
            // double reference on the rows next to each boundary and on the last row (this case's own position)
            const unsigned sample[] = {15U, 16U, 127U, 128U, rows - 1U};
            for (unsigned si = 0; si < 5U; ++si) {
                const unsigned r = sample[si];
                if (r >= rows || (si == 4U && (r == 15U || r == 16U || r == 127U || r == 128U))) continue;
                const unsigned p = pos0 + r;
                const unsigned rlo = c.window && p >= c.window ? p - c.window + 1U : 0U;
                for (unsigned h = 0; h < kHeads; ++h) {
                    const unsigned kvh = h / (kHeads / kKv);
                    const float* qr = hqr.data() + (static_cast<std::size_t>(r) * kHeads + h) * kDim;
                    const float* br = hbr.data() + (static_cast<std::size_t>(r) * kHeads + h) * kExtent;
                    const double sink = hsr[static_cast<std::size_t>(r) * kHeads + h];
                    std::vector<double> sc(p - rlo + 1U);
                    double mx = sink;
                    for (unsigned key = rlo; key <= p; ++key) {
                        const unsigned slot = c.wrapped ? key % c.capacity : key;
                        const std::uint16_t* kr = hk.data() + (static_cast<std::size_t>(slot) * kKv + kvh) * kDim;
                        double dot = 0;
                        for (unsigned i = 0; i < kDim; ++i) dot += static_cast<double>(qr[i]) * bf2f(kr[i]);
                        const unsigned dist = p - key;
                        sc[key - rlo] = dot * kScale + (dist < kExtent ? br[dist] : 0.0);
                        mx = std::max(mx, sc[key - rlo]);
                    }
                    double den = std::exp(sink - mx);
                    std::vector<double> o(kDim, 0.0);
                    for (unsigned key = rlo; key <= p; ++key) {
                        const unsigned slot = c.wrapped ? key % c.capacity : key;
                        const std::uint16_t* vr = hv.data() + (static_cast<std::size_t>(slot) * kKv + kvh) * kDim;
                        const double pk = std::exp(sc[key - rlo] - mx);
                        den += pk;
                        for (unsigned i = 0; i < kDim; ++i) o[i] += pk * bf2f(vr[i]);
                    }
                    const float* got = out.data() + (static_cast<std::size_t>(r) * kHeads + h) * kDim;
                    for (unsigned i = 0; i < kDim; ++i) {
                        const double ref = o[i] / den;
                        scale_rows = std::max(scale_rows, std::fabs(ref));
                        raise_max(err_rows, std::fabs(got[i] - ref));
                    }
                }
            }
        }
        const int w = std::snprintf(rows_text + rows_text_len, sizeof rows_text - rows_text_len, "%s%u:%016llx%s",
                                    rows_text_len ? "," : "", rows, rows_bits, fits ? "" : "-skip");
        if (w > 0) rows_text_len = std::min(sizeof rows_text - 1U, rows_text_len + static_cast<std::size_t>(w));
    }
    // BF16 Q and P operands: a few 1e-3 of the output scale; 2e-2 (or 8x the one-row flash error) only rejects wrong rows.
    const double bound_rows = std::max(2e-2 * std::max(scale_rows, 1e-3), 8.0 * err_flash);
    ok = ok && rows_det && miss_rows == 0 && bad == 0 && std::isfinite(err_rows) && err_rows <= bound_rows;
    if (!ok) g_fail = true;
    std::printf("SPLIT %s pos=%u window=%u wrapped=%d splits=%u chunk=%u err_split=%.3e err_flash=%.3e ref_scale=%.3e det=%s "
                "bits=%016llx flash_bits=%016llx swz3=%s err_rows=%.3e unwritten=%zu nonfinite=%zu rows=%s %s\n",
                c.name, c.position, c.window, c.wrapped ? 1 : 0, plan.splits, plan.chunk, err_split, err_flash, scale_ref,
                det ? "EQUAL" : "DIFF", fnv(first), fnv(fl), pipes_equal ? "EQUAL" : "DIFF", err_rows, miss1 + miss_rows, bad, rows_text, ok ? "PASS" : "FAIL");
    if (check_only) return;

    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    float us[3] = {0, 0, 0};
    const auto launch = [&](int arm) { return arm == 0 ? flash() : arm == 1 ? split() : flash_swz(); };
    for (int arm = 0; arm < 3; ++arm) {
        for (int w = 0; w < 10; ++w) check(launch(arm), "warm");
        check(cudaEventRecord(a), "rec");
        for (int i = 0; i < iters; ++i) check(launch(arm), "run");
        check(cudaEventRecord(b), "rec");
        check(cudaEventSynchronize(b), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, a, b), "elapsed");
        us[arm] = 1000.0f * ms / static_cast<float>(iters);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    // unique K/V bytes of the walk (each KV head once; the 3 query heads of a group re-read it from L2)
    const unsigned walked = c.position - lo + 1U;
    const double kv_bytes = 2.0 * walked * kKv * kDim * sizeof(std::uint16_t);
    const auto gbs = [&](float t) { return t > 0 ? kv_bytes / (1000.0 * t) : 0.0; };
    std::printf("TIME %s pos=%u flash_us=%.2f split_us=%.2f ratio=%.2f swz3_us=%.2f swz3/pad2=%.3f kv_MB=%.2f "
                "GBs_pad2=%.0f GBs_swz3=%.0f\n",
                c.name, c.position, us[0], us[1], us[1] > 0 ? us[0] / us[1] : 0.0f, us[2], us[0] > 0 ? us[2] / us[0] : 0.0f,
                kv_bytes / 1e6, gbs(us[0]), gbs(us[2]));
}

}  // namespace

// Plain prompt flash (the unprepared form, Qwen's prefill): the pad2 body with and without the tail split, whose
// tail warps are the inactive-warp path of the body. Output digests compare two builds case by case; every element
// must be written and finite.
void run_plain() {
    constexpr unsigned kD = cc::kPrefillFlashHeadDim;
    std::mt19937 rng(777U);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    const std::size_t cache_elems = std::size_t{kPlainCapacity} * kKv * kD;
    std::vector<std::uint16_t> hk(cache_elems), hv(cache_elems);
    for (auto& x : hk) x = f2bf(uni(rng) * 2.0f);
    for (auto& x : hv) x = f2bf(uni(rng));
    std::vector<float> hq(std::size_t{kPlainMaxRows} * kHeads * kD);
    for (auto& x : hq) x = uni(rng) * 2.0f;
    Dev<std::uint16_t> dk(cache_elems), dv(cache_elems);
    Dev<float> dq(hq.size()), out(hq.size());
    dk.put(hk);
    dv.put(hv);
    dq.put(hq);
    const cc::AttentionDims dims = plain_dims();
    const auto* kb = reinterpret_cast<const __nv_bfloat16*>(dk.p);
    const auto* vb = reinterpret_cast<const __nv_bfloat16*>(dv.p);
    for (const auto& c : kPlainCases) {
        for (const auto tail : {cc::FlashTail::off, cc::FlashTail::split}) {
            out.poison();
            const cc::GroupedView qv{c.rows, kHeads, kD};
            check(cc::attention_prefill_flash(dims, dq.p, qv, nullptr, {}, kb, vb, c.pos0, c.rows, out.p, nullptr,
                                              cc::FlashPipe::pad2, tail), "plain flash");
            check(cudaDeviceSynchronize(), "sync plain");
            const auto o = out.get();
            const std::size_t live = std::size_t{c.rows} * kHeads * kD, miss = unwritten(o, live), bad = nonfinite(o, live);
            unsigned long long bits = 1469598103934665603ULL;
            for (std::size_t i = 0; i < live; ++i) {
                std::uint32_t u = 0;
                std::memcpy(&u, &o[i], sizeof u);
                for (unsigned b = 0; b < 4U; ++b) bits = (bits ^ ((u >> (8U * b)) & 0xffU)) * 1099511628211ULL;
            }
            const bool ok = miss == 0 && bad == 0;
            if (!ok) g_fail = true;
            std::printf("PLAIN pos0=%u rows=%u tail=%s bits=%016llx unwritten=%zu nonfinite=%zu %s\n", c.pos0, c.rows,
                        tail == cc::FlashTail::split ? "split" : "off", bits, miss, bad, ok ? "PASS" : "FAIL");
        }
    }
}

int main(int argc, char** argv) {
    int iters = 200;
    bool check_only = false, args_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--check-only")) check_only = true;
        else if (!std::strcmp(argv[i], "--check-args")) args_only = true;
        else if (!std::strcmp(argv[i], "--inject-unwritten")) g_inject_unwritten = true;
        else if (!std::strcmp(argv[i], "--inject-stale")) g_inject_stale = true;
        else if (!std::strcmp(argv[i], "--inject-stale-rows")) g_inject_stale_rows = true;
        else if (!std::strcmp(argv[i], "--inject-nonfinite")) g_inject_nonfinite = true;
        else {
            std::printf("unknown option %s\n", argv[i]);
            return 2;
        }
    }
    unsigned launches = 0;
    const unsigned refused = check_args(launches);  // host only: no CUDA call before this line
    std::printf("ARGS %s launches=%u refused=%u\n", refused == 0 ? "ok" : "FAIL", launches, refused);
    if (refused != 0) return 3;
    if (args_only) return 0;
    for (const auto& c : kCases) run_case(c, iters, check_only);
    run_plain();
    std::printf("RESULT %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
