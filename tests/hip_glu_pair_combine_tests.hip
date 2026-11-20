// Decode FFN fusions (Hy4-flash binder) on synthetic Q8_0 / FP32 data at the Hy4 shapes, no model:
//  1. identity: the GLU projection group (ops/glu/q8_rows_glu_pair.hpp: ONE q8_rows launch with the paired store) ==
//     the incumbent one-member q8_rows launches with the bf16 Linear store + glu::swiglu_kernel, bitwise (plain member
//     output and act), at the Hy4 MoE layer (latent down 1536 + shared gate / up 1536, width 3072), the Hy4 dense layer
//     (gate / up 13824) and small edge geometries (partial last output group, no plain member); an odd plain count throws.
//  2. identity: moe_core::enqueue_combine_rounded_eo (bf16_steps, 16 x 1536) == glu::launch_epilogue<bf16>(eo) +
//     enqueue_combine, bitwise, with NaN / inf / overflowing / denormal / -0 values in eo.
//  3. timing (printed, not asserted): per layer, incumbent vs fused, ABAB x 3; weights rotate over copies larger than the
//     32 MB last-level cache.
#include "ops/glu/q8_rows_glu_pair.hpp"
#include "ops/glu/glu_device.hpp"
#include "moe_core/route_wave_kernel.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
#define CHECK(x)                                                                                        \
    do {                                                                                                \
        const hipError_t e_ = (x);                                                                      \
        if (e_ != hipSuccess) {                                                                         \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                               \
        }                                                                                               \
    } while (0)

namespace glu = strix::ops::glu;
using Pair = glu::Q8RowsGluPair<glu::DeclaredExp, glu::Round::bf16>;

// The incumbent one-member site: a Q8_0 Linear's bf16 stored output (the Hy4 adapter's Q8Rows with the store epilogue).
struct LinearSite {
    using Input = float;
    static constexpr unsigned rows = 1U;
    const void* input[1];
    strix::q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* out;
    __device__ static float coefficient(unsigned scale, std::int8_t code) {
        return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(scale))) * static_cast<float>(code);
    }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) { return lower + upper; }
    __device__ void store(std::uint32_t o, unsigned, float value) const { out[o] = glu::epilogue<glu::Round::bf16>(value, nullptr, 0U); }
};

std::uint32_t rng = 0x9E3779B9U;
std::uint32_t next_u32() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next_u32() % 1000000U) / 1000000.0F; }
std::uint32_t bits_of(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}
float from_bits(std::uint32_t b) {
    float v;
    std::memcpy(&v, &b, 4);
    return v;
}
float bf16_value(float v) { return from_bits(bits_of(v) & 0xffff0000U); }
std::vector<unsigned char> q8_rows_host(std::uint64_t rows, std::uint32_t width) {
    std::vector<unsigned char> w(rows * (width / 32U) * 34U);
    for (std::size_t at = 0; at < w.size(); at += 34U) {
        const _Float16 h = static_cast<_Float16>(0.001F + 0.02F * static_cast<float>(next_u32() % 1000U) / 1000.0F);
        std::memcpy(&w[at], &h, 2);
        for (unsigned i = 0; i < 32U; ++i) w[at + 2U + i] = static_cast<unsigned char>(static_cast<int>(next_u32() % 255U) - 127);
    }
    return w;
}
template <class T> T* to_device(const std::vector<T>& h) {
    T* d = nullptr;
    CHECK(hipMalloc(&d, h.size() * sizeof(T)));
    CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <class T> std::vector<T> to_host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}
template <class T> T* zeros(std::size_t n) {
    T* d = nullptr;
    CHECK(hipMalloc(&d, n * sizeof(T)));
    CHECK(hipMemset(d, 0, n * sizeof(T)));
    return d;
}
std::size_t mismatches(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) n += bits_of(a[i]) != bits_of(b[i]) ? 1U : 0U;
    return n;
}
template <class F> double time_us(F&& f, unsigned reps, hipStream_t s) {
    f(0U);
    CHECK(hipStreamSynchronize(s));
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    CHECK(hipEventRecord(a, s));
    for (unsigned i = 0; i < reps; ++i) f(i);
    CHECK(hipEventRecord(b, s));
    CHECK(hipEventSynchronize(b));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b));
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    return 1000.0 * static_cast<double>(ms) / reps;
}

struct Geometry {
    std::uint32_t width, plain, ff;
    const char* name;
};
// One weight set: [plain rows][gate rows][up rows] in one allocation (each member its own row range).
struct Set {
    unsigned char* w;
    const unsigned char* plain(const Geometry&) const { return w; }
    const unsigned char* gate(const Geometry& g) const { return w + std::size_t{g.plain} * (g.width / 32U) * 34U; }
    const unsigned char* up(const Geometry& g) const { return gate(g) + std::size_t{g.ff} * (g.width / 32U) * 34U; }
};
void incumbent(const Set& w, const Geometry& g, const float* x, float* z, float* gb, float* ub, float* act, hipStream_t s) {
    const auto linear = [&](const unsigned char* weights, std::uint32_t outputs, float* y) {
        LinearSite site{};
        site.input[0] = x;
        site.segments = strix::q8_rows::segments({{weights, outputs}});
        site.width = g.width;
        site.outputs = outputs;
        site.out = y;
        strix::q8_rows::launch(site, false, s);
    };
    if (g.plain) linear(w.plain(g), g.plain, z);
    linear(w.gate(g), g.ff, gb);
    linear(w.up(g), g.ff, ub);
    glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(gb, ub, g.ff, glu::no_clamp, act, s);
}
void fused(const Set& w, const Geometry& g, const float* x, float* z, float* act, hipStream_t s) {
    const auto site = glu::q8_rows_glu_pair<glu::DeclaredExp, glu::Round::bf16>(x, g.width, g.plain ? w.plain(g) : nullptr, g.plain, z,
                                                                               w.gate(g), w.up(g), g.ff, act, glu::no_clamp);
    strix::q8_rows::launch(site, false, s);
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

// Identity on copy 0; with reps, ABAB timing over `copies` weight sets.
void run_group(const Geometry& g, unsigned copies, unsigned reps, hipStream_t s) {
    const std::uint64_t rows = std::uint64_t{g.plain} + 2U * g.ff;
    std::vector<Set> sets(copies);
    for (auto& set : sets) {
        const auto host = q8_rows_host(rows, g.width);
        set.w = to_device(host);
    }
    std::vector<float> xh(g.width);
    for (auto& v : xh) v = bf16_value(uniform(-3.0F, 3.0F));  // xn is a bf16-valued norm output
    float* x = to_device(xh);
    const std::size_t zn = g.plain ? g.plain : 1U;
    float *z0 = zeros<float>(zn), *z1 = zeros<float>(zn), *gb = zeros<float>(g.ff), *ub = zeros<float>(g.ff);
    float *a0 = zeros<float>(g.ff), *a1 = zeros<float>(g.ff);
    incumbent(sets[0], g, x, z0, gb, ub, a0, s);
    fused(sets[0], g, x, z1, a1, s);
    CHECK(hipStreamSynchronize(s));
    CHECK(hipGetLastError());
    const auto zh0 = to_host(z0, zn), zh1 = to_host(z1, zn), ah0 = to_host(a0, g.ff), ah1 = to_host(a1, g.ff);
    const std::size_t zbad = g.plain ? mismatches(zh0, zh1) : 0U, abad = mismatches(ah0, ah1);
    std::size_t nonzero = 0;
    for (float v : ah1) nonzero += (v != 0.0F && std::isfinite(v)) ? 1U : 0U;
    std::printf("identity %s: plain %zu/%u act %zu/%u mismatches, act finite nonzero %zu -> %s\n", g.name, zbad, g.plain, abad, g.ff,
                nonzero, zbad == 0U && abad == 0U ? "PASS" : "FAIL");
    expect(zbad == 0U && abad == 0U, g.name);
    expect(nonzero * 2U > g.ff, "fused act mostly nonzero (sanity)");
    if (reps) {
        for (unsigned pair = 0; pair < 3U; ++pair) {
            const double ti = time_us([&](unsigned i) { incumbent(sets[i % copies], g, x, z0, gb, ub, a0, s); }, reps, s);
            const double tf = time_us([&](unsigned i) { fused(sets[i % copies], g, x, z1, a1, s); }, reps, s);
            const double gb_set = static_cast<double>(rows * (g.width / 32U) * 34U) / 1e9;
            std::printf("timing %s pair %u: incumbent %.2f us (%u launches) fused %.2f us (1 launch) delta %+.2f us; fused %.1f GB/s\n",
                        g.name, pair, ti, g.plain ? 4U : 3U, tf, tf - ti, gb_set / (tf * 1e-6));
        }
    }
    for (auto& set : sets) CHECK(hipFree(set.w));
    for (float* p : {x, z0, z1, gb, ub, a0, a1}) CHECK(hipFree(p));
}

void run_combine(unsigned reps, hipStream_t s) {
    constexpr unsigned K = 16U, L = 1536U;
    std::vector<float> eo(std::size_t{K} * L), w(K);
    for (auto& v : eo) v = uniform(-2.0F, 2.0F) * (next_u32() % 7U == 0U ? 1000.0F : 1.0F);
    const std::uint32_t specials[] = {0x7fc00001U, 0x7f800000U, 0xff800000U, 0x7f7fffffU, 0x00000001U, 0x80000000U, 0x3f808000U,
                                      0x3f818000U, 0xff7fffffU};
    for (unsigned i = 0; i < sizeof specials / sizeof specials[0]; ++i) eo[(i * 977U) % eo.size()] = from_bits(specials[i]);
    for (auto& v : w) v = uniform(0.0F, 0.2F);
    float *e0 = to_device(eo), *e1 = to_device(eo), *wd = to_device(w), *y0 = zeros<float>(L), *y1 = zeros<float>(L);
    glu::launch_epilogue<glu::Round::bf16>(e0, nullptr, L, std::size_t{K} * L, e0, s);
    strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, K, L>(wd, e0, nullptr, y0, 1U, s);
    strix::moe_core::enqueue_combine_rounded_eo<strix::moe::CombineForm::bf16_steps, K, L>(wd, e1, nullptr, y1, 1U, s);
    CHECK(hipStreamSynchronize(s));
    CHECK(hipGetLastError());
    const std::size_t bad = mismatches(to_host(y0, L), to_host(y1, L));
    std::printf("identity combine 16x1536 rounded eo: %zu/%u mismatches -> %s\n", bad, L, bad == 0U ? "PASS" : "FAIL");
    expect(bad == 0U, "combine rounded eo");
    if (reps) {
        CHECK(hipMemcpy(e0, eo.data(), eo.size() * 4U, hipMemcpyHostToDevice));
        for (unsigned pair = 0; pair < 3U; ++pair) {
            const double ti = time_us(
                [&](unsigned) {
                    glu::launch_epilogue<glu::Round::bf16>(e1, nullptr, L, std::size_t{K} * L, e0, s);
                    strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, K, L>(wd, e0, nullptr, y0, 1U, s);
                },
                reps, s);
            const double tf = time_us(
                [&](unsigned) { strix::moe_core::enqueue_combine_rounded_eo<strix::moe::CombineForm::bf16_steps, K, L>(wd, e1, nullptr, y1, 1U, s); },
                reps, s);
            std::printf("timing combine pair %u: incumbent %.2f us (2 launches) fused %.2f us (1 launch) delta %+.2f us\n", pair, ti, tf, tf - ti);
        }
    }
    for (float* p : {e0, e1, wd, y0, y1}) CHECK(hipFree(p));
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    run_group({64U, 2U, 6U, "edge plain 2 + pair 6 (partial group)"}, 1U, 0U, s);
    run_group({32U, 0U, 1U, "edge pair 1 (no plain)"}, 1U, 0U, s);
    run_group({96U, 4U, 5U, "edge plain 4 + pair 5"}, 1U, 0U, s);
    run_group({3072U, 1536U, 1536U, "hy4-moe latent+shared gate/up"}, bench ? 5U : 1U, bench ? 400U : 0U, s);
    run_group({3072U, 0U, 13824U, "hy4-dense gate/up"}, 1U, bench ? 60U : 0U, s);
    run_combine(bench ? 2000U : 0U, s);
    bool threw = false;
    try {
        float* d = zeros<float>(8);
        auto site = glu::q8_rows_glu_pair<glu::DeclaredExp, glu::Round::bf16>(d, 32U, d, 3U, d, d, d, 2U, d, glu::no_clamp);
        strix::q8_rows::launch(site, false, s);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "odd pair_first throws");
    std::printf("odd plain count rejected: %s\n", threw ? "PASS" : "FAIL");
    CHECK(hipStreamDestroy(s));
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
