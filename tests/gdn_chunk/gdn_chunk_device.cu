// Device fixture of the FAST chunked GDN prefill (gdn-chunk16-f32-v1) against the serial gdn_prefill_layer (GPU ticket only).
//   gdn_chunk_device            : rows 1, 17, 300, 4096 (Qwen dims 16 key / 48 value heads): output and new-state relative RMS
//                                 chunk vs serial <= 1e-4, chunk run twice bitwise (deterministic). Prints GDN CHECK lines.
//   gdn_chunk_device --bench N  : whole gdn_prefill_layer (conv + delta + conv commit) serial vs chunk, and gdn_chunk_delta alone,
//                                 median of 7 timed runs at N rows. Prints GDN BENCH lines.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/gdn_chunk.hpp"

namespace cc = strix::cuda_core;
#define CKD(x)                                                                                    \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                         \
        }                                                                                         \
    } while (0)

struct Buf {
    float* p = nullptr;
    std::size_t n = 0;
    explicit Buf(std::size_t n_) : n(n_) { CKD(cudaMalloc(&p, n * sizeof(float))); }
    ~Buf() { cudaFree(p); }
    void up(const std::vector<float>& h) { CKD(cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice)); }
    [[nodiscard]] std::vector<float> down() const {
        std::vector<float> h(n);
        CKD(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }
};
static std::vector<float> rnd(std::size_t n, std::mt19937& g, float mean, float sd) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (float& x : v) x = d(g);
    return v;
}
static double rel(const std::vector<float>& a, const std::vector<float>& b, std::size_t off, std::size_t n) {
    double num = 0, den = 0;
    for (std::size_t i = off; i < off + n; ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1));
}

struct Fixture {
    cc::GdnDims d;
    std::uint32_t rows;
    std::vector<float> h_conv, h_delta;
    Buf conv_w, ssm_a, dt_bias, gain, conv, delta, raw, qkvc, ar, br, z, out, ws;
    std::uint32_t* parity = nullptr;
    Fixture(std::uint32_t r, unsigned seed)
        : rows(r), conv_w(std::size_t{d.conv_channels()} * 4), ssm_a(d.value_heads), dt_bias(d.value_heads), gain(d.value_dim),
          conv(std::size_t{d.conv_channels()} * 3), delta(2 * std::size_t{d.value_heads} * 128 * 128),
          raw(std::size_t{r} * d.conv_channels()), qkvc(std::size_t{r} * d.conv_channels()), ar(std::size_t{r} * d.value_heads),
          br(std::size_t{r} * d.value_heads), z(std::size_t{r} * d.v_width()), out(std::size_t{r} * d.v_width()),
          ws(cc::gdn_chunk_workspace_floats(d, r)) {
        std::mt19937 g(seed);
        conv_w.up(rnd(conv_w.n, g, 0.f, 0.5f));
        std::vector<float> sa(d.value_heads);
        std::uniform_real_distribution<float> u(-3.f, 0.5f);
        for (float& x : sa) x = -std::exp(u(g));
        ssm_a.up(sa);
        dt_bias.up(rnd(dt_bias.n, g, 0.f, 0.5f));
        gain.up(rnd(gain.n, g, 1.f, 0.1f));
        h_conv = rnd(conv.n, g, 0.f, 1.f);
        h_delta = rnd(delta.n, g, 0.f, 0.3f);
        raw.up(rnd(raw.n, g, 0.f, 1.f));
        ar.up(rnd(ar.n, g, 0.f, 1.f));
        br.up(rnd(br.n, g, 0.f, 2.f));
        z.up(rnd(z.n, g, 0.f, 1.f));
        CKD(cudaMalloc(&parity, sizeof(std::uint32_t)));
    }
    ~Fixture() { cudaFree(parity); }
    void reset() {
        conv.up(h_conv);
        delta.up(h_delta);
        CKD(cudaMemset(parity, 0, sizeof(std::uint32_t)));
        CKD(cudaMemset(out.p, 0, out.n * sizeof(float)));
    }
    cudaError_t layer(bool chunk, std::uint8_t scan = 0) {
        const cc::GdnLayer L{conv_w.p, ssm_a.p, dt_bias.p, gain.p};
        return cc::gdn_prefill_layer(d, L, cc::GdnState{conv.p, delta.p}, cc::GdnWindow{raw.p, qkvc.p, ar.p, br.p}, z.p, out.p, rows,
                                     1e-6f, parity, chunk ? cc::GdnChunkWork{ws.p, ws.n, false, scan} : cc::GdnChunkWork{}, nullptr);
    }
    cudaError_t delta_only(std::uint8_t scan = 0) {
        const cc::GdnLayer L{conv_w.p, ssm_a.p, dt_bias.p, gain.p};
        return cc::gdn_chunk_delta(d, L, cc::GdnState{conv.p, delta.p}, cc::GdnWindow{raw.p, qkvc.p, ar.p, br.p}, z.p, out.p, rows,
                                   1e-6f, parity, cc::GdnChunkWork{ws.p, ws.n, false, scan}, nullptr);
    }
};

// scan 0 = f32 (gdn-chunk16-f32-v1, bound 1e-4), 1 = tf32 (gdn-chunk16-tf32-v1, TF32 operands: bound 3e-3)
static int check(std::uint32_t rows, std::uint8_t scan) {
    Fixture f(rows, 1234u + rows);
    const std::size_t side = std::size_t{f.d.value_heads} * 128 * 128;
    f.reset();
    CKD(f.layer(false));
    CKD(cudaDeviceSynchronize());
    const auto o_s = f.out.down(), s_s = f.delta.down();
    f.reset();
    CKD(f.layer(true, scan));
    CKD(cudaDeviceSynchronize());
    const auto o_c = f.out.down(), s_c = f.delta.down();
    f.reset();
    CKD(f.layer(true, scan));
    CKD(cudaDeviceSynchronize());
    const auto o_c2 = f.out.down(), s_c2 = f.delta.down();
    const bool det = std::memcmp(o_c.data(), o_c2.data(), o_c.size() * 4) == 0 && std::memcmp(s_c.data(), s_c2.data(), s_c.size() * 4) == 0;
    const bool committed_untouched = std::memcmp(s_c.data(), f.h_delta.data(), side * 4) == 0;
    const double eo = rel(o_c, o_s, 0, o_s.size()), es = rel(s_c, s_s, side, side);
    const double bound = scan ? 3e-3 : 1e-4;
    const bool ok = det && committed_untouched && eo <= bound && es <= bound;
    std::printf("GDN CHECK scan=%s rows=%u out_rel=%.3e state_rel=%.3e deterministic=%d committed-side-untouched=%d (bound %.0e) %s\n",
                scan ? "tf32" : "f32", rows, eo, es, det ? 1 : 0, committed_untouched ? 1 : 0, bound, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

template <class F>
static float time_ms(F&& fn, Fixture& f) {
    cudaEvent_t a, b;
    CKD(cudaEventCreate(&a));
    CKD(cudaEventCreate(&b));
    std::vector<float> ms;
    for (int it = 0; it < 9; ++it) {
        f.reset();
        CKD(cudaDeviceSynchronize());
        CKD(cudaEventRecord(a));
        CKD(fn());
        CKD(cudaEventRecord(b));
        CKD(cudaEventSynchronize(b));
        float x = 0;
        CKD(cudaEventElapsedTime(&x, a, b));
        if (it >= 2) ms.push_back(x);  // 2 warm-up runs
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp p{};
    CKD(cudaGetDevice(&dev));
    CKD(cudaGetDeviceProperties(&p, dev));
    std::printf("device %s sm_%d%d SMs %d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    if (argc >= 3 && std::strcmp(argv[1], "--bench") == 0) {
        const auto rows = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
        Fixture f(rows, 7u);
        const float s = time_ms([&] { return f.layer(false); }, f);
        const float c = time_ms([&] { return f.layer(true); }, f);
        CKD(f.layer(false));  // fills qkv_c (conv output) for the delta-only timing
        const float dc = time_ms([&] { return f.delta_only(); }, f);
        const float dt = time_ms([&] { return f.delta_only(1); }, f);
        const float ct = time_ms([&] { return f.layer(true, 1); }, f);
        std::printf("GDN BENCH rows=%u layer_serial_ms=%.4f layer_chunk_ms=%.4f delta_chunk_ms=%.4f speedup=%.2fx (median of 7)\n", rows,
                    s, c, dc, s / c);
        std::printf("GDN BENCH TF32 rows=%u layer_chunk_tc_ms=%.4f delta_chunk_tc_ms=%.4f delta f32 %.4f -> tf32 %.4f ms (saved %.4f)\n", rows,
                    ct, dt, dc, dt, dc - dt);
        return 0;
    }
    int fails = 0;
    for (std::uint8_t scan : {std::uint8_t{0}, std::uint8_t{1}})
        for (std::uint32_t r : {1u, 17u, 300u, 4096u}) fails += check(r, scan);
    std::printf("gdn_chunk_device: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
