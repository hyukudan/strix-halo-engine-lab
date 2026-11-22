// No-model check + timing of attention_decode_prepared_split_bf16 against the prepared flash step path (rows = 1, device
// position) on the Inkling 2 Mini decode shape (24 query heads, 8 KV heads, d128, relative bias extent 1024, sinks).
// Per case: both outputs vs a double reference of the same contract (FP32 Q, unrounded P), split determinism over two
// launches (also proves the ticket reset), then microseconds per launch (cudaEvent, `--iters`). Prints RESULT PASS|FAIL.
// Usage: cuda_core_decode_split_bench [--iters N] [--check-only]
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
#include <random>
#include <vector>

namespace cc = strix::cuda_core;

namespace {

constexpr unsigned kHeads = 24, kKv = 8, kDim = 128, kExtent = 1024;
constexpr float kScale = 0.08838834764831845f;
bool g_fail = false;

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
    Dev<float> dq(hq.size()), db(hb.size()), ds(hs.size()), out_flash(kHeads * kDim), out_split(kHeads * kDim);
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
    const auto flash = [&] {
        return cc::attention_prefill_prepared_bf16(dims, prepared, dq.p, kb, vb, c.position, 1, out_flash.p, nullptr,
                                                   cc::FlashPipe::pad2, dpos.p);
    };
    const auto split = [&] {
        return cc::attention_decode_prepared_split_bf16(dims, prepared, dq.p, kb, vb, dpos.p, out_split.p, ws.p, ws_bytes,
                                                        nullptr);
    };
    check(flash(), "flash");
    check(split(), "split 1");
    check(cudaDeviceSynchronize(), "sync");
    const auto first = out_split.get();
    check(split(), "split 2");
    check(cudaDeviceSynchronize(), "sync");
    const auto second = out_split.get();
    const bool det = std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0;
    const auto fl = out_flash.get();

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
            err_split = std::max(err_split, std::fabs(first[h * kDim + i] - ref));
            err_flash = std::max(err_flash, std::fabs(fl[h * kDim + i] - ref));
        }
    }
    // FP32 split: bound 1e-4 of the output scale; it must not be worse than the BF16-operand flash.
    const double bound = 1e-4 * std::max(scale_ref, 1e-3);
    const bool ok = det && err_split <= bound && err_split <= std::max(err_flash, bound);
    if (!ok) g_fail = true;
    const auto plan = strix::attn_split::decode_split_plan(c.position, c.window);
    // FNV-1a of the split output bits: compares two builds of the kernel case by case (schedule changes must keep them)
    unsigned long long bits = 1469598103934665603ULL;
    for (const float f : first) {
        std::uint32_t u = 0;
        std::memcpy(&u, &f, sizeof u);
        for (unsigned b = 0; b < 4U; ++b) bits = (bits ^ ((u >> (8U * b)) & 0xffU)) * 1099511628211ULL;
    }
    std::printf("SPLIT %s pos=%u window=%u wrapped=%d splits=%u chunk=%u err_split=%.3e err_flash=%.3e ref_scale=%.3e det=%s "
                "bits=%016llx %s\n",
                c.name, c.position, c.window, c.wrapped ? 1 : 0, plan.splits, plan.chunk, err_split, err_flash, scale_ref,
                det ? "EQUAL" : "DIFF", bits, ok ? "PASS" : "FAIL");
    if (check_only) return;

    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    float us[2] = {0, 0};
    for (int arm = 0; arm < 2; ++arm) {
        for (int w = 0; w < 10; ++w) check(arm ? split() : flash(), "warm");
        check(cudaEventRecord(a), "rec");
        for (int i = 0; i < iters; ++i) check(arm ? split() : flash(), "run");
        check(cudaEventRecord(b), "rec");
        check(cudaEventSynchronize(b), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, a, b), "elapsed");
        us[arm] = 1000.0f * ms / static_cast<float>(iters);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::printf("TIME %s pos=%u flash_us=%.2f split_us=%.2f ratio=%.2f\n", c.name, c.position, us[0], us[1],
                us[1] > 0 ? us[0] / us[1] : 0.0f);
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 200;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--check-only")) check_only = true;
    }
    // sliding visits (window 1024, wrapped ring) and full visits (prefix, unwrapped) at 0.1K / 1K / 2.5K / 8K / 16K
    const Case cases[] = {
        {"sliding", 1280, 1024, 37, true},   {"sliding", 1280, 1024, 1023, true}, {"sliding", 1280, 1024, 2520, true},
        {"sliding", 1280, 1024, 8191, true}, {"full", 16384, 0, 37, false},      {"full", 16384, 0, 100, false},
        {"full", 16384, 0, 255, false},      {"full", 16384, 0, 256, false},     {"full", 16384, 0, 511, false},
        {"full", 16384, 0, 1023, false},     {"full", 16384, 0, 2520, false},    {"full", 16384, 0, 8191, false},
        {"full", 16384, 0, 16383, false},
        // position-canonical plan: aligned sliding row (4 chunks), short window straddling a unit (one pass from first),
        // past 16K keys (512-key aligned chunks)
        {"sliding", 1280, 1024, 2559, true}, {"short", 256, 200, 3000, true}, {"full", 20480, 0, 20000, false},
    };
    for (const auto& c : cases) run_case(c, iters, check_only);
    std::printf("RESULT %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
