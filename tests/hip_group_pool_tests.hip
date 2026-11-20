// GPU test of the shared group-indexer device cores against the host oracle (src/attn/group_indexer_ref.cpp), synthetic
// fixtures only (no weights):
//   pool   attn_core/group_pool.hpp pool_groups_kernel, MODEL_B GatedSoftmaxApe   == pool_group_gated_ape (bits)
//   scan   attn_core/group_score_scan.hpp rg1 / rg2 / rk1 / rk2, MODEL_B view (bf16 keys, 16 heads, InHeadWeight) and a
//          GLM-shaped view (fp32 keys, 32 heads, InsideRelu), same / separate slots   == score_group -> score_key (bits)
//          + top-byte histogram
//   select attn_core/group_select_kernel.hpp (select_row SegmentPolicy<4>) fed by the scan's keys + hist0, MODEL_B view (P = t / 4, k = 511,
//          own-group tail), incl. forced ties at the cut   == select_groups + expand_tokens (token lists)
// Argument "rate": also time the scan and the select at 2,048 and 8,192 groups (8K / 32K tokens).
#pragma clang fp contract(off)  // declared arithmetic: the oracle's bits
#include "attn_core/group_pool.hpp"
#include "attn_core/group_score_scan.hpp"
#include "attn_core/select.hpp"
#include "attn_core/group_select_kernel.hpp"
#include "strix/attn/group_indexer.hpp"
#include "strix/ds41_moe.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gi = strix::attn::group_indexer;
// Gated-pool AlwaysOwn indexer shape exercised here (ratio 4, 16 heads x 128, 512 groups); a model binding keeps its
// own Shape in its model package.
constexpr gi::Shape kGatedApe{4U, 16U, 128U, 512U, gi::Pooling::GatedSoftmaxApe, gi::OwnGroup::AlwaysOwn,
                              gi::ScoreScale::InHeadWeight};
namespace gp = strix::attn::group_pool;
namespace cp = strix::attn_core::group_pool;
namespace gs = strix::attn_core::group_scan;
namespace sel = strix::attn_core::select;

namespace {
int g_fail = 0;
#define HIPCHECK(x)                                                                              \
    do {                                                                                         \
        const hipError_t e_ = (x);                                                               \
        if (e_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

std::uint32_t bits(float x) { return __builtin_bit_cast(std::uint32_t, x); }
std::uint16_t bf16_of(float x) { return static_cast<std::uint16_t>(bits(gi::bf16_round(x)) >> 16U); }
float exp_ds41(float x) { return strix::ds41::declared_exp(x); }

template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    HIPCHECK(hipMalloc(&p, h.size() * sizeof(T)));
    HIPCHECK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
template <class T> std::vector<T> host(const T* p, std::size_t n) {
    std::vector<T> h(n);
    HIPCHECK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

// ------------------------------------------------------------------ pool
struct ModelBPoolView {
    static constexpr unsigned kRatio = 4, kDim = 128;
    const std::uint16_t* tk;  // [groups * 4][128] bf16 token keys
    gp::F32Ape w;
    float* out;  // [groups][128]
    unsigned groups;
    __device__ bool live(unsigned b) const { return b < groups; }
    __device__ unsigned group(unsigned b) const { return b; }
    __device__ float key(unsigned g, unsigned j, unsigned i) const {
        return gp::bf16_value(tk[(static_cast<std::size_t>(g) * kRatio + j) * kDim + i]);
    }
    __device__ const gp::F32Ape& weights() const { return w; }
    __device__ float exp(float x) const { return strix::ds41::declared_exp(x); }
    __device__ void store(unsigned g, unsigned lane, const float* o) const { out[static_cast<std::size_t>(g) * kDim + lane] = o[lane]; }
};

void test_pool(std::mt19937& rng) {
    constexpr unsigned R = 4, D = 128, G = 64;
    std::normal_distribution<float> n01(0.0F, 1.0F);
    std::vector<std::uint16_t> tk(G * R * D);
    std::vector<float> wv(D * D), wg(D * D), ape(R * D), norm(D);
    for (auto& x : tk) x = bf16_of(n01(rng));
    for (float& x : wv) x = n01(rng) * 0.09F;
    for (float& x : wg) x = n01(rng) * 0.5F;
    for (float& x : ape) x = n01(rng);
    for (float& x : norm) x = 1.0F + 0.1F * n01(rng);
    float *dwv = dev(wv), *dwg = dev(wg), *dape = dev(ape), *dnorm = dev(norm), *dout = nullptr;
    std::uint16_t* dtk = dev(tk);
    HIPCHECK(hipMalloc(&dout, G * D * sizeof(float)));
    const ModelBPoolView v{dtk, gp::F32Ape{gi::GatedApeWeights{dwv, dwg, dape, dnorm, 1e-5F}, D}, dout, G};
    cp::pool_groups_kernel<ModelBPoolView><<<G, D>>>(v);
    HIPCHECK(hipGetLastError());
    const std::vector<float> got = host(dout, G * D);
    const gi::GatedApeWeights hw{wv.data(), wg.data(), ape.data(), norm.data(), 1e-5F};
    unsigned same = 0;
    std::vector<float> keys(R * D), want(D);
    for (unsigned g = 0; g < G; ++g) {
        for (unsigned e = 0; e < R * D; ++e) keys[e] = gp::bf16_value(tk[g * R * D + e]);
        gi::pool_group_gated_ape(kGatedApe, keys.data(), hw, exp_ds41, want.data());
        for (unsigned o = 0; o < D; ++o) same += bits(got[g * D + o]) == bits(want[o]) ? 1U : 0U;
    }
    std::printf("pool modelb GatedSoftmaxApe: %u / %u bit-identical to the oracle\n", same, G * D);
    CHECK(same == G * D);
    for (void* p : {static_cast<void*>(dwv), static_cast<void*>(dwg), static_cast<void*>(dape), static_cast<void*>(dnorm),
                    static_cast<void*>(dout), static_cast<void*>(dtk)})
        HIPCHECK(hipFree(p));
}

// ------------------------------------------------------------------ scan
template <class K, unsigned H, bool Inside> struct TestScan {
    struct Rows {
        const K* keys[2];
        unsigned pools[2];
        unsigned rows;
    };
    using Key = K;
    static constexpr unsigned kHeads = H, kDim = 128;
    static constexpr float kInvSqrt128 = 0.08838834764831845F;
    __host__ __device__ static float weight_scale() { return Inside ? 0.17677669529663687F : 0.022097087F; }
    __device__ static unsigned rows(const Rows& d) { return d.rows; }
    __device__ static const K* keys(const Rows& d, unsigned r) { return d.keys[r]; }
    __device__ static unsigned pools(const Rows& d, unsigned r) { return d.pools[r]; }
    __device__ static float weight(float raw) { return raw * weight_scale(); }
    __device__ static float scaled(float dot) {
        if constexpr (Inside) return dot * kInvSqrt128;
        else return dot;
    }
    __device__ static std::uint32_t key_of(float s) { return gi::score_key(s); }
};
using ModelBScan = TestScan<std::uint16_t, 16, false>;
using GlmScanT = TestScan<float, 32, true>;

template <class V> float key_value(const typename V::Key& k) {
    if constexpr (sizeof(typename V::Key) == 4) return k;
    else return gp::bf16_value(k);
}

// Fixture of up to two rows over one or two key slots; launches variant `var` (1 rg1, 2 rg2, 3 rk2, 5 rk1).
template <class V> struct ScanFixture {
    unsigned P_max;
    std::vector<typename V::Key> keys[2];
    std::vector<float> q, wraw;  // [2][H][128], [2][H]
    typename V::Key* dkeys[2] = {nullptr, nullptr};
    float *dq = nullptr, *dw = nullptr;
    std::uint32_t *dout = nullptr, *dhist = nullptr;
    ScanFixture(std::mt19937& rng, unsigned pmax, bool bf16_q) : P_max(pmax) {
        std::normal_distribution<float> n01(0.0F, 1.0F);
        for (auto& k : keys) {
            k.resize(static_cast<std::size_t>(pmax) * V::kDim);
            for (auto& x : k) {
                if constexpr (sizeof(typename V::Key) == 4) x = n01(rng);
                else x = bf16_of(n01(rng));
            }
        }
        q.resize(2U * V::kHeads * V::kDim);
        for (float& x : q) x = bf16_q ? gi::bf16_round(n01(rng)) : n01(rng);
        wraw.resize(2U * V::kHeads);
        for (float& x : wraw) x = n01(rng);
        for (unsigned s = 0; s < 2; ++s) dkeys[s] = dev(keys[s]);
        dq = dev(q);
        dw = dev(wraw);
        HIPCHECK(hipMalloc(&dout, 2U * pmax * sizeof(std::uint32_t)));
        HIPCHECK(hipMalloc(&dhist, 2U * 256U * sizeof(std::uint32_t)));
    }
    ~ScanFixture() {
        for (void* p : {static_cast<void*>(dkeys[0]), static_cast<void*>(dkeys[1]), static_cast<void*>(dq), static_cast<void*>(dw),
                        static_cast<void*>(dout), static_cast<void*>(dhist)})
            (void)hipFree(p);
    }
    typename V::Rows rows(unsigned n, bool same_slot, unsigned p0, unsigned p1) const {
        return typename V::Rows{{dkeys[0], same_slot ? dkeys[0] : dkeys[1]}, {p0, p1}, n};
    }
    void launch(int var, const typename V::Rows& d, unsigned p_bound) const {
        HIPCHECK(hipMemsetAsync(dhist, 0, 2U * 256U * sizeof(std::uint32_t)));
        const unsigned tiles = (p_bound + gs::kThreads - 1) / gs::kThreads;
        const unsigned rk_tiles = (p_bound + 2 * gs::kThreads - 1) / (2 * gs::kThreads);
        if (var == 1) gs::score_kernel<V><<<dim3(tiles, d.rows), gs::kThreads>>>(d, dq, dw, dout, P_max, dhist);
        else if (var == 2) gs::score_rg_kernel<V, 2><<<dim3(tiles, (d.rows + 1) / 2), gs::kThreads>>>(d, dq, dw, dout, P_max, dhist);
        else if (var == 3) gs::score_rk_kernel<V, 2><<<dim3(rk_tiles, (d.rows + 1) / 2), gs::kThreads>>>(d, dq, dw, dout, P_max, dhist);
        else gs::score_rk_kernel<V, 1><<<dim3(rk_tiles, d.rows), gs::kThreads>>>(d, dq, dw, dout, P_max, dhist);
        HIPCHECK(hipGetLastError());
    }
    // Host oracle: score_group with the contract shape of the view.
    std::uint32_t want(unsigned r, unsigned slot, unsigned p) const {
        const gi::Shape s = V::kHeads == 16 ? kGatedApe : gi::kGlmDsa;
        float k[128], w[32];
        for (unsigned i = 0; i < V::kDim; ++i) k[i] = key_value<V>(keys[slot][static_cast<std::size_t>(p) * V::kDim + i]);
        for (unsigned h = 0; h < V::kHeads; ++h) w[h] = wraw[r * V::kHeads + h] * V::weight_scale();
        return gi::score_key(gi::score_group(s, q.data() + static_cast<std::size_t>(r) * V::kHeads * V::kDim, k, w, V::kInvSqrt128));
    }
};

template <class V> void test_scan(std::mt19937& rng, const char* name, bool bf16_q) {
    ScanFixture<V> f(rng, 1003, bf16_q);
    for (int var : {1, 2, 3, 5}) {
        for (bool same : {true, false}) {
            const unsigned p0 = 1003, p1 = 937;
            const auto d = f.rows(2, same, p0, p1);
            f.launch(var, d, p0);
            const auto got = host(f.dout, 2U * f.P_max);
            const auto hist = host(f.dhist, 512);
            unsigned bad = 0, hist_bad = 0;
            for (unsigned r = 0; r < 2; ++r) {
                const unsigned P = r == 0 ? p0 : p1;
                std::vector<unsigned> h(256, 0);
                for (unsigned p = 0; p < P; ++p) {
                    const std::uint32_t w = f.want(r, (same || r == 0) ? 0U : 1U, p);
                    bad += got[r * f.P_max + p] == w ? 0U : 1U;
                    ++h[w >> 24U];
                }
                for (unsigned e = 0; e < 256; ++e) hist_bad += hist[r * 256 + e] == h[e] ? 0U : 1U;
            }
            std::printf("scan %s var %d %s slots: key mismatches %u, histogram mismatches %u\n", name, var, same ? "same" : "separate",
                        bad, hist_bad);
            CHECK(bad == 0U && hist_bad == 0U);
        }
    }
}

// ------------------------------------------------------------------ select (MODEL_B binding of the shared kernel)
using ModelBSelectView = strix::attn_core::group_select::PositionView<kGatedApe, 4096>;

void test_select(std::mt19937& rng, bool all_tied) {
    ScanFixture<ModelBScan> f(rng, 2600, true);
    // Forced ties: 40 groups share one key row (equal scores); all_tied: every group equal -> the cut is all ties and
    // the declared rule must keep the 511 lowest group indices.
    for (unsigned g = all_tied ? 1U : 600U; g < (all_tied ? 2600U : 640U); ++g)
        std::memcpy(&f.keys[0][static_cast<std::size_t>(g) * 128], &f.keys[0][static_cast<std::size_t>(all_tied ? 0U : 599U) * 128],
                    128 * sizeof(std::uint16_t));
    HIPCHECK(hipMemcpy(f.dkeys[0], f.keys[0].data(), f.keys[0].size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    std::uint32_t *dtok = nullptr, *dn = nullptr;
    HIPCHECK(hipMalloc(&dtok, 4096 * sizeof(std::uint32_t)));
    HIPCHECK(hipMalloc(&dn, sizeof(std::uint32_t)));
    unsigned* dtpos = nullptr;
    HIPCHECK(hipMalloc(&dtpos, sizeof(unsigned)));
    unsigned bad_rows = 0, rows = 0;
    for (std::uint32_t t : {0U, 5U, 2043U, 2047U, 2048U, 2049U, 2052U, 3001U, 6000U, 10399U, 10400U}) {
        const unsigned P = gi::visible_groups(kGatedApe, t);
        f.launch(1, f.rows(1, true, P, 0), P > 0 ? P : 1);
        HIPCHECK(hipMemcpy(dtpos, &t, sizeof(unsigned), hipMemcpyHostToDevice));
        strix::attn_core::group_select::enqueue_select<ModelBSelectView>(strix::attn_core::group_select::PositionRows{dtpos}, 1U, f.dout,
                                                                      P > 0 ? P : 1U, f.dhist, dtok, dn, nullptr);
        HIPCHECK(hipGetLastError());
        const unsigned n = host(dn, 1)[0];
        std::vector<std::uint32_t> got = host(dtok, n < 4096 ? n : 4096);
        std::vector<float> scores(P);
        for (unsigned p = 0; p < P; ++p) {
            const std::uint32_t key = f.want(0, 0, p);
            scores[p] = __builtin_bit_cast(float, (key & 0x80000000U) != 0U ? (key & 0x7FFFFFFFU) : ~key);
        }
        const std::vector<std::uint32_t> want = gi::expand_tokens(kGatedApe, gi::select_groups(kGatedApe, scores.data(), P), t);
        const bool ok = got == want;
        bad_rows += ok ? 0U : 1U;
        ++rows;
        if (!ok) std::printf("select t=%u: device %u tokens, oracle %zu tokens\n", t, n, want.size());
    }
    std::printf("select modelb (P = t/4, k = 511, own tail, %s): %u / %u rows equal to the oracle\n",
                all_tied ? "all groups tied" : "40 tied groups", rows - bad_rows, rows);
    CHECK(bad_rows == 0U);
    (void)hipFree(dtok);
    (void)hipFree(dn);
    (void)hipFree(dtpos);
}

// ------------------------------------------------------------------ rate (timing only, synthetic, <= 8,192 groups)
template <class V> void rate_scan(std::mt19937& rng, const char* name, int var, unsigned P, unsigned nrows, bool same) {
    ScanFixture<V> f(rng, P, true);
    const auto d = f.rows(nrows, same, P, P);
    hipEvent_t a, b;
    HIPCHECK(hipEventCreate(&a));
    HIPCHECK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) f.launch(var, d, P);
    constexpr int kIters = 50;
    HIPCHECK(hipEventRecord(a));
    for (int i = 0; i < kIters; ++i) f.launch(var, d, P);
    HIPCHECK(hipEventRecord(b));
    HIPCHECK(hipEventSynchronize(b));
    float ms = 0.0F;
    HIPCHECK(hipEventElapsedTime(&ms, a, b));
    const double us = 1000.0 * ms / kIters;  // includes the 2 KiB histogram memset per launch
    const double bytes = static_cast<double>(P) * V::kDim * sizeof(typename V::Key) * (same ? 1 : nrows);
    std::printf("rate scan %s var %d P=%u rows=%u %s: %.1f us/launch, %.1f GB/s key stream\n", name, var, P, nrows,
                same ? "same-slot" : "separate", us, bytes / (us * 1e3));
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
}

void rate_select(std::mt19937& rng, unsigned P) {
    ScanFixture<ModelBScan> f(rng, P, true);
    const std::uint32_t t = P * 4U + 1U;
    std::uint32_t *dtok = nullptr, *dn = nullptr;
    HIPCHECK(hipMalloc(&dtok, 4096 * sizeof(std::uint32_t)));
    HIPCHECK(hipMalloc(&dn, sizeof(std::uint32_t)));
    f.launch(1, f.rows(1, true, P, 0), P);
    unsigned* dtpos = nullptr;
    HIPCHECK(hipMalloc(&dtpos, sizeof(unsigned)));
    HIPCHECK(hipMemcpy(dtpos, &t, sizeof(unsigned), hipMemcpyHostToDevice));
    const auto run_select = [&] {
        strix::attn_core::group_select::enqueue_select<ModelBSelectView>(strix::attn_core::group_select::PositionRows{dtpos}, 1U, f.dout, P,
                                                                      f.dhist, dtok, dn, nullptr);
    };
    hipEvent_t a, b;
    HIPCHECK(hipEventCreate(&a));
    HIPCHECK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) run_select();
    constexpr int kIters = 50;
    HIPCHECK(hipEventRecord(a));
    for (int i = 0; i < kIters; ++i) run_select();
    HIPCHECK(hipEventRecord(b));
    HIPCHECK(hipEventSynchronize(b));
    float ms = 0.0F;
    HIPCHECK(hipEventElapsedTime(&ms, a, b));
    std::printf("rate select modelb P=%u k=511: %.1f us/launch, %.2f ns/group\n", P, 1000.0 * ms / kIters, 1e6 * ms / kIters / P);
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    (void)hipFree(dtok);
    (void)hipFree(dn);
    (void)hipFree(dtpos);
}
}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(20261118U);
    test_pool(rng);
    test_scan<ModelBScan>(rng, "modelb(bf16,16h,InHeadWeight)", true);
    test_scan<GlmScanT>(rng, "glm-shape(fp32,32h,InsideRelu)", false);
    test_select(rng, false);
    test_select(rng, true);
    if (argc > 1 && std::strcmp(argv[1], "rate") == 0) {
        for (unsigned P : {2048U, 8192U}) {
            rate_scan<ModelBScan>(rng, "modelb", 1, P, 1, true);
            rate_scan<ModelBScan>(rng, "modelb", 5, P, 1, true);
            rate_scan<GlmScanT>(rng, "glm-shape", 1, P, 1, true);
            rate_select(rng, P);
        }
        rate_scan<GlmScanT>(rng, "glm-shape", 1, 8192, 2, false);  // cross-check: receipt R2 rg1 32K = 62.6 us
        rate_scan<ModelBScan>(rng, "modelb", 3, 8192, 2, true);
    }
    if (g_fail != 0) {
        std::printf("hip_group_pool_tests: %d FAIL\n", g_fail);
        return 1;
    }
    std::printf("hip_group_pool_tests: PASS\n");
    return 0;
}
