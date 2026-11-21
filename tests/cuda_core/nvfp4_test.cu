// NVFP4 (TURBO weights): converter vs host RTN reference, GEMV vs host dequant reference, rows 1..32 bitwise row
// invariance, and a bandwidth microbench on the 27B shapes. Run: cuda_core_nvfp4_test [--bench-only|--test-only].
// Cross-kernel weight read-ahead (bitwise + cold chain timing): cuda_core_nvfp4_test --readahead [rows] [window us] [reps].
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/nvfp4.hpp"
#include "pdl_harness.cuh"

namespace cc = strix::cuda_core;
#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);          \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

static int g_fail = 0;
static void expect(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}
static float h2f(float v) { return __half2float(__float2half_rn(v)); }
static const float kV[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static void test_shape(std::uint32_t out, std::uint32_t in, std::mt19937& rng) {
    std::printf("-- shape %u x %u\n", out, in);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> w(static_cast<std::size_t>(out) * in);
    for (auto& v : w) v = nd(rng);
    for (int i = 0; i < 40; ++i) w[rng() % w.size()] *= 30.f;  // outliers
    for (std::uint32_t k = 0; k < 16; ++k) w[k] = 0.f;          // exact-zero group
    for (std::uint32_t k = 16; k < 32; ++k) w[k] *= 1e-6f;      // tiny group (scale underflow)
    float* d_w;
    CK(cudaMalloc(&d_w, w.size() * 4));
    CK(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    void* d_t;
    CK(cudaMalloc(&d_t, cc::nvfp4_bytes(out, in)));
    float gs = 0;
    CK(cc::nvfp4_quantize_f32(d_w, out, in, d_t, &gs, 0));
    std::vector<unsigned char> tiles(cc::nvfp4_bytes(out, in));
    CK(cudaMemcpy(tiles.data(), d_t, tiles.size(), cudaMemcpyDeviceToHost));
    float amax = 0;
    for (float v : w) amax = std::max(amax, std::fabs(v));
    expect(gs == amax / (6.0f * 448.0f), "global scale = amax/(6*448)");
    std::size_t mism = 0, scmis = 0;
    double se = 0, sn = 0;
    float maxsc = 0;
    std::vector<float> deq(w.size());
    for (std::uint32_t o = 0; o < out; ++o)
        for (std::uint32_t G = 0; G < in / 16; ++G) {
            unsigned char codes[16];
            const unsigned char sb = cc::nvfp4_host_quantize_group(&w[static_cast<std::size_t>(o) * in + G * 16], gs, codes);
            const std::uint32_t m = o >> 4, r = o & 15, h = r >> 3, g = r & 7, c = G >> 3, j = G & 7;
            const unsigned char* rec = tiles.data() + (static_cast<std::size_t>(m) * (in >> 7) + c) * cc::kNvfp4RecordBytes;
            if (rec[1024 + g * 16 + h * 8 + j] != sb) ++scmis;
            const float sd = cc::nvfp4_host_e4m3(sb);
            maxsc = std::max(maxsc, sd);
            for (int i = 0; i < 16; ++i) {
                const float dv = cc::nvfp4_host_dequant(tiles.data(), in, o, G * 16 + i);
                const float rv = ((codes[i] & 8) ? -1.f : 1.f) * kV[codes[i] & 7] * sd;
                if (dv != rv) ++mism;
                deq[static_cast<std::size_t>(o) * in + G * 16 + i] = dv * gs;
            }
        }
    for (std::size_t i = 0; i < w.size(); ++i) {
        se += double(deq[i] - w[i]) * (deq[i] - w[i]);
        sn += double(w[i]) * w[i];
    }
    std::printf("   scale bytes != host RTN: %zu, dequant != host RTN: %zu, max scale %.1f, rel RMS err %.4f\n", scmis, mism, maxsc,
                std::sqrt(se / sn));
    expect(scmis == 0 && mism == 0, "converter == host RTN reference (scales and values)");

    const std::uint32_t R = 32;
    std::vector<float> x(static_cast<std::size_t>(R) * in);
    std::normal_distribution<float> xd(0.f, 1.f);
    for (auto& v : x) v = xd(rng);
    float *d_x, *d_y, *d_res;
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMalloc(&d_y, std::size_t(R) * out * 4));
    CK(cudaMalloc(&d_res, std::size_t(R) * out * 4));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    std::vector<float> res(std::size_t(R) * out);
    for (auto& v : res) v = xd(rng);
    CK(cudaMemcpy(d_res, res.data(), res.size() * 4, cudaMemcpyHostToDevice));
    cc::DenseWeight W;
    W.data = d_t;
    W.format = cc::WeightFormat::nvfp4;
    W.out = out;
    W.in = in;
    W.gscale = gs;
    std::vector<float> dq(static_cast<std::size_t>(out) * in);
    for (std::uint32_t o = 0; o < out; ++o)
        for (std::uint32_t k = 0; k < in; ++k) dq[std::size_t(o) * in + k] = cc::nvfp4_host_dequant(tiles.data(), in, o, k);
    std::vector<double> ref(std::size_t(R) * out);
    for (std::uint32_t r = 0; r < R; ++r) {
        std::vector<float> xr(in);
        for (std::uint32_t k = 0; k < in; ++k) xr[k] = h2f(x[std::size_t(r) * in + k]);
        for (std::uint32_t o = 0; o < out; ++o) {
            double s = 0;
            for (std::uint32_t k = 0; k < in; ++k) s += double(dq[std::size_t(o) * in + k]) * xr[k];
            ref[std::size_t(r) * out + o] = s * gs;
        }
    }
    std::vector<float> y32(std::size_t(R) * out), y(std::size_t(R) * out);
    CK(cc::nvfp4_dense_rows(W, d_x, d_y, R, nullptr, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y32.data(), d_y, y32.size() * 4, cudaMemcpyDeviceToHost));
    double maxabs = 0, refmax = 0;
    for (double v : ref) refmax = std::max(refmax, std::fabs(v));
    for (std::size_t i = 0; i < ref.size(); ++i) maxabs = std::max(maxabs, std::fabs(y32[i] - ref[i]));
    std::printf("   rows 32: max|y-ref|=%.3e (ref max %.3e) rel %.2e\n", maxabs, refmax, maxabs / refmax);
    expect(maxabs / refmax < 1e-5, "GEMV rows 32 within 1e-5 of the host double reference (f16 activations)");
    for (std::uint32_t Rr : {1u, 2u, 5u, 8u, 9u, 16u, 17u, 24u, 25u}) {
        CK(cc::nvfp4_dense_rows(W, d_x, d_y, Rr, nullptr, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y.data(), d_y, std::size_t(Rr) * out * 4, cudaMemcpyDeviceToHost));
        char b[96];
        std::snprintf(b, sizeof b, "rows %u bitwise == rows 32 (same rows)", Rr);
        expect(std::memcmp(y.data(), y32.data(), std::size_t(Rr) * out * 4) == 0, b);
    }
    {
        bool same = true;
        for (std::uint32_t r : {0u, 7u, 31u}) {
            CK(cc::nvfp4_dense_rows(W, d_x + std::size_t(r) * in, d_y, 1, nullptr, 0));
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(y.data(), d_y, std::size_t(out) * 4, cudaMemcpyDeviceToHost));
            same = same && std::memcmp(y.data(), y32.data() + std::size_t(r) * out, std::size_t(out) * 4) == 0;
        }
        expect(same, "single-row calls bitwise == rows of the 32-row batch");
    }
    {
        CK(cc::nvfp4_dense_rows(W, d_x, d_y, 3, d_res, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y.data(), d_y, std::size_t(3) * out * 4, cudaMemcpyDeviceToHost));
        bool same = true;
        for (std::size_t i = 0; i < std::size_t(3) * out; ++i) same = same && (y[i] == res[i] + y32[i]);
        expect(same, "residual = fl(residual + (W x)) bitwise");
    }
    {   // multi-job launch == separate calls, bitwise
        float* d_y2;
        CK(cudaMalloc(&d_y2, std::size_t(R) * out * 4));
        const cc::Nvfp4Job jobs[2] = {{W, d_y, nullptr}, {W, d_y2, d_res}};
        CK(cc::nvfp4_dense_rows_multi(jobs, 2, d_x, 5, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> ya(std::size_t(5) * out), yb(std::size_t(5) * out);
        CK(cudaMemcpy(ya.data(), d_y, ya.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yb.data(), d_y2, yb.size() * 4, cudaMemcpyDeviceToHost));
        bool same = std::memcmp(ya.data(), y32.data(), ya.size() * 4) == 0;
        for (std::size_t i = 0; i < yb.size(); ++i) same = same && yb[i] == res[i] + y32[i];
        expect(same, "multi-job launch bitwise == separate calls (plain + residual)");
        cudaFree(d_y2);
    }
    {
        cc::DenseWeight U = W;  // up == gate weights: y = silu(g) * g
        CK(cc::nvfp4_gate_up_silu_rows(W, U, d_x, d_y, 4, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y.data(), d_y, std::size_t(4) * out * 4, cudaMemcpyDeviceToHost));
        std::size_t bad = 0;
        for (std::size_t i = 0; i < std::size_t(4) * out; ++i) {
            const float g = y32[i];
            const float e = g / (1.0f + std::exp(-g)) * g;
            if (std::fabs(y[i] - e) > 1e-5f * (std::fabs(e) + 1e-3f)) ++bad;
        }
        expect(bad == 0, "fused gate/up silu close to host epilogue");
    }
    cudaFree(d_w); cudaFree(d_t); cudaFree(d_x); cudaFree(d_y); cudaFree(d_res);
}

__global__ void read_kernel(const uint4* p, std::size_t n, unsigned* sink) {
    unsigned a = 0;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
        uint4 v;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
        a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x12345678u) *sink = a;
}

static void read_peak() {
    const std::size_t bytes = std::size_t(1) << 30;
    uint4* p;
    unsigned* sink;
    CK(cudaMalloc(&p, bytes));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemset(p, 1, bytes));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int bpsm : {2, 4, 8}) {
        read_kernel<<<188 * bpsm, 256>>>(p, bytes / 16, sink);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 5; ++i) read_kernel<<<188 * bpsm, 256>>>(p, bytes / 16, sink);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("read-only 1 GiB, %d blocks/SM x 256 thr: %.0f GB/s\n", bpsm, 5.0 * bytes / (ms * 1e-3) / 1e9);
    }
    cudaFree(p);
    cudaFree(sink);
}

static void bench() {
    read_peak();
    struct S {
        const char* name;
        std::uint32_t out, in;
        bool fused;
    };
    const S shapes[] = {{"qkv 5120->12288", 12288, 5120, false},        {"kv 5120->1024", 1024, 5120, false},
                        {"o 6144->5120", 5120, 6144, false},            {"gdn_qkv 5120->10240", 10240, 5120, false},
                        {"z 5120->6144", 6144, 5120, false},            {"gate+up 5120->17408 x2", 17408, 5120, true},
                        {"down 17408->5120", 5120, 17408, false},       {"lm_head 5120->248320", 248320, 5120, false}};
    const std::uint32_t rows_list[] = {1, 2, 4, 8, 16, 32};
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    float* d_x;
    CK(cudaMalloc(&d_x, std::size_t(32) * 17408 * 4));
    {
        std::vector<float> hx(std::size_t(32) * 17408);
        std::mt19937 r(5);
        std::normal_distribution<float> d(0, 1);
        for (auto& v : hx) v = d(r);
        CK(cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    }
    float* d_y;
    CK(cudaMalloc(&d_y, std::size_t(32) * 248320 * 4));
    std::printf("%-24s %4s", "shape (GB/s over weight bytes)", "ks");
    for (auto r : rows_list) std::printf("   R=%-2u us  GB/s", r);
    std::printf("\n");
    for (const auto& s : shapes) {
        const std::size_t per = cc::nvfp4_bytes(s.out, s.in) * (s.fused ? 2 : 1);
        const std::size_t sets = std::max<std::size_t>(2, (std::size_t(1500) << 20) / per + 1);
        unsigned char* buf;
        CK(cudaMalloc(&buf, per * sets));
        std::vector<unsigned char> h(std::min<std::size_t>(per, 64u << 20));
        std::mt19937 rg(7);
        for (auto& b : h) b = static_cast<unsigned char>(rg());
        for (std::size_t o = 0; o < per * sets; o += h.size())
            CK(cudaMemcpy(buf + o, h.data(), std::min(h.size(), per * sets - o), cudaMemcpyHostToDevice));
        std::printf("%-24s %4u", s.name, cc::nvfp4_split_k(s.out, s.in));
        for (auto r : rows_list) {
            auto run_s = [&](std::size_t set, cudaStream_t st) {
                cc::DenseWeight W;
                W.format = cc::WeightFormat::nvfp4;
                W.out = s.out;
                W.in = s.in;
                W.gscale = 0.001f;
                W.data = buf + set * per;
                if (s.fused) {
                    cc::DenseWeight U = W;
                    U.data = buf + set * per + per / 2;
                    CK(cc::nvfp4_gate_up_silu_rows(W, U, d_x, d_y, r, st));
                } else {
                    CK(cc::nvfp4_dense_rows(W, d_x, d_y, r, nullptr, st));
                }
            };
            auto run = [&](std::size_t set) { run_s(set, 0); };
            for (std::size_t i = 0; i < sets; ++i) run(i);
            CK(cudaDeviceSynchronize());
            // graph of `sets` calls (each on its own weight copy) removes host launch gaps: device-side time per call
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            cudaStream_t cs;
            CK(cudaStreamCreate(&cs));
            run_s(0, cs);  // workspaces are per (device, stream): size this stream's one outside the capture
            CK(cudaStreamSynchronize(cs));
            CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
            for (std::size_t i = 0; i < sets; ++i) run_s(i, cs);
            CK(cudaStreamEndCapture(cs, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            CK(cudaGraphLaunch(ge, cs));
            CK(cudaStreamSynchronize(cs));
            CK(cudaEventRecord(e0, cs));
            for (int i = 0; i < 4; ++i) CK(cudaGraphLaunch(ge, cs));
            CK(cudaEventRecord(e1, cs));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(gr));
            cc::nvfp4_release_workspace(cs);
            CK(cudaStreamDestroy(cs));
            const double us = ms * 1e3 / (4.0 * sets);
            std::printf("  %7.1f %5.0f", us, per / (us * 1e-6) / 1e9);
        }
        std::printf("   [%.1f MB]\n", per / 1e6);
        cudaFree(buf);
    }
}

static cc::DenseWeight make_weight(std::uint32_t out, std::uint32_t in, std::mt19937& rng, std::vector<void*>& owned) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> w(static_cast<std::size_t>(out) * in);
    for (auto& v : w) v = nd(rng);
    float* d_w;
    CK(cudaMalloc(&d_w, w.size() * 4));
    CK(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    void* d_t;
    CK(cudaMalloc(&d_t, cc::nvfp4_bytes(out, in)));
    float gs = 0;
    CK(cc::nvfp4_quantize_f32(d_w, out, in, d_t, &gs, 0));
    CK(cudaFree(d_w));
    owned.push_back(d_t);
    cc::DenseWeight W;
    W.data = d_t;
    W.format = cc::WeightFormat::nvfp4;
    W.out = out;
    W.in = in;
    W.gscale = gs;
    return W;
}

// PDL chain test (proven harness, pdl_harness.cuh). Part A (run_probes, once) shows on this device that the harness SEES a
// hazard: a seeded pre-wait read / pre-wait write is caught in every repetition in which the successor really started early
// (%globaltimer witness), the serial form never overlaps, a release before the last write is never stale. Part B: the real
// chain norm -> A -> B(+x) -> twin -> gate/up -> norm -> A -> B(+b) -> gate/up (every fp4_gemv template; split-K and ks 1)
// with PDL (stream and graph) and a poison gate before each call in turn must equal the serial form bit for bit; the same
// gate in front of the seeded twin (pre-wait read, pre-wait write) must differ, with the twin seen starting inside the poison
// window. Inputs are uploaded from pinned memory on the test stream; every comparison follows cudaStreamSynchronize.
static bool g_test_pdl = false;

static void test_pdl_chain(std::uint32_t split_target, std::mt19937& rng) {
    static bool probes_done = false;
    if (!probes_done) {
        probes_done = true;
        const int f = pdlh::run_probes(16);
        expect(f == 0, "pdl probes: seeded hazards caught whenever overlapped, graph overlaps every rep, serial clean");
    }
    constexpr std::uint32_t kH = 5120, kI = 6144, kF = 1024;
    constexpr int kCalls = 9, kTwinAt = 3, kSeedReps = 3;
    std::vector<void*> owned;
    const cc::DenseWeight A = make_weight(kI, kH, rng, owned), B = make_weight(kH, kI, rng, owned);
    const cc::DenseWeight G = make_weight(kF, kH, rng, owned), U = make_weight(kF, kH, rng, owned);
    std::vector<float> gain_h(kH);
    std::uniform_real_distribution<float> ug(0.5f, 1.5f);
    for (auto& v : gain_h) v = ug(rng);
    float* gain;
    CK(cudaMalloc(&gain, kH * 4));
    CK(cudaMemcpy(gain, gain_h.data(), kH * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());  // weights + gain (legacy stream) are complete before any work on the test stream
    cc::nvfp4_set_split_target(split_target);
    std::printf("-- pdl chain: split target %u, ks A %u B %u gate/up %u\n", split_target, cc::nvfp4_split_k(kI, kH),
                cc::nvfp4_split_k(kH, kI), cc::nvfp4_split_k(kF, kH));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    unsigned long long *stamps, *hs;  // [0] gate latest release, [1] gate earliest restore, [2] twin earliest entry
    CK(cudaMalloc(&stamps, 3 * 8));
    CK(cudaMallocHost(&hs, 3 * 8));
    auto set_pdl = [](bool on) {
        cc::set_programmatic_launch(on);
        g_test_pdl = on;
    };
    for (std::uint32_t rows : {1u, 8u, 16u, 24u, 32u}) {  // every fp4_gemv template (<1,4> <2,4> <3,2> <4,2>)
        const std::size_t nh = static_cast<std::size_t>(rows) * kH, ni = static_cast<std::size_t>(rows) * kI,
                          nf = static_cast<std::size_t>(rows) * kF;
        // out buffers in snapshot order; dx is input-only but is poisoned by the gate like every live buffer.
        const char* names[6] = {"dn", "da", "db", "db2", "dc", "dt"};
        const std::size_t sz[6] = {nh, ni, nh, nh, nf, nh};
        const std::uint32_t cols[6] = {kH, kI, kH, kH, kF, kH};
        std::size_t lo[7] = {0};
        for (int b = 0; b < 6; ++b) lo[b + 1] = lo[b] + sz[b];
        float* dx;
        float* d[6];
        CK(cudaMalloc(&dx, nh * 4));
        for (int b = 0; b < 6; ++b) CK(cudaMalloc(&d[b], sz[b] * 4));
        float *dn = d[0], *da = d[1], *db = d[2], *db2 = d[3], *dc = d[4], *dt = d[5];
        pdlh::Live live{};
        live.p[0] = dx;
        live.n[0] = nh;
        for (int b = 0; b < 6; ++b) {
            live.p[b + 1] = d[b];
            live.n[b + 1] = sz[b];
        }
        live.count = 7;
        float *shadow, *hx, *ho;
        CK(cudaMalloc(&shadow, (nh + lo[6]) * 4));
        CK(cudaMallocHost(&hx, nh * 4));
        CK(cudaMallocHost(&ho, lo[6] * 4));
        std::uniform_real_distribution<float> ud(-1.f, 1.f);
        for (std::size_t i = 0; i < nh; ++i) hx[i] = ud(rng);
        CK(cudaMemcpyAsync(dx, hx, nh * 4, cudaMemcpyHostToDevice, s));
        auto poison = [&]() {  // a skipped write, or a read before the producer wrote, cannot pass
            for (int b = 0; b < 6; ++b) CK(cudaMemsetAsync(d[b], 0xFF, sz[b] * 4, s));
        };
        poison();
        CK(cudaStreamSynchronize(s));
        const cc::GroupedView v{rows, 1, kH, 0, 0};
        auto chain = [&](int gate_at, int twin_mode) {
            auto gate = [&](int k) {
                if (gate_at == k)
                    CK(pdlh::launch(g_test_pdl, dim3(64), dim3(256), s, pdlh::poison_gate, live, shadow, stamps, 50000ull));
            };
            gate(0);
            CK(cc::rmsnorm(dx, v, gain, dn, v, 1e-6f, s));
            gate(1);
            CK(cc::nvfp4_dense_rows(A, dn, da, rows, nullptr, s));
            gate(2);
            CK(cc::nvfp4_dense_rows(B, da, db, rows, dx, s));
            gate(3);
            CK(pdlh::launch(g_test_pdl, dim3(static_cast<unsigned>((nh + 255) / 256)), dim3(256), s, pdlh::twin_copy,
                            static_cast<const float*>(db), dt, static_cast<unsigned>(nh), twin_mode, stamps + 2));
            gate(4);
            CK(cc::nvfp4_gate_up_silu_rows(G, U, db, dc, rows, s));
            gate(5);
            CK(cc::rmsnorm(db, v, gain, dn, v, 1e-6f, s));
            gate(6);
            CK(cc::nvfp4_dense_rows(A, dn, da, rows, nullptr, s));
            gate(7);
            CK(cc::nvfp4_dense_rows(B, da, db2, rows, db, s));
            gate(8);
            CK(cc::nvfp4_gate_up_silu_rows(G, U, db2, dc, rows, s));
        };
        // One run (stream or one graph replay), outputs and stamps read after cudaStreamSynchronize; then re-poison.
        // twin_in: the twin started inside the gate's poison window (only meaningful when the gate sits before it).
        bool twin_in = false;
        auto run = [&](int gate_at, int twin_mode, bool graph) {
            CK(cudaMemsetAsync(stamps, 0, 8, s));
            CK(cudaMemsetAsync(stamps + 1, 0xFF, 16, s));
            if (graph) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                chain(gate_at, twin_mode);
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, 0));
                CK(cudaGraphLaunch(ge, s));
                CK(cudaStreamSynchronize(s));
                CK(cudaGraphExecDestroy(ge));
                CK(cudaGraphDestroy(g));
            } else {
                chain(gate_at, twin_mode);
            }
            for (int b = 0; b < 6; ++b) CK(cudaMemcpyAsync(ho + lo[b], d[b], sz[b] * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(hs, stamps, 3 * 8, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            twin_in = hs[0] != 0 && hs[2] < hs[1];
            std::vector<float> o(ho, ho + lo[6]);
            poison();
            return o;
        };
        auto same = [&](const std::vector<float>& p, const std::vector<float>& q, const char* side, int rep, bool loud) {
            if (std::memcmp(p.data(), q.data(), p.size() * 4) == 0) return true;
            for (int b = 0; loud && b < 6; ++b) {
                std::size_t ndiff = 0, first = 0;
                for (std::size_t i = lo[b]; i < lo[b + 1]; ++i)
                    if (std::memcmp(&p[i], &q[i], 4) != 0 && ndiff++ == 0) first = i - lo[b];
                if (ndiff == 0) continue;
                std::uint32_t got = 0, want = 0;
                std::memcpy(&got, &p[lo[b] + first], 4);
                std::memcpy(&want, &q[lo[b] + first], 4);
                std::printf("   diff rows %u %s %d %s: %zu of %zu differ, first [row %zu col %zu] serial %08x got %08x\n", rows,
                            side, rep, names[b], ndiff, sz[b], first / cols[b], first % cols[b], want, got);
            }
            return false;
        };
        char what[240];
        set_pdl(false);
        const std::vector<float> ref = run(-1, 0, false);
        const bool ref_ok = same(run(4, 0, false), ref, "serial-gate", 4, true);  // second, independent serial reference
        const std::vector<float> ref2 = run(-1, 2, false);                          // seeded twin mode 2 serial: dt = 3 + db
        std::snprintf(what, sizeof what, "pdl chain rows %u split target %u: serial reference stable (x2)", rows, split_target);
        expect(ref_ok, what);
        set_pdl(true);
        bool ok = true;
        for (bool graph : {false, true}) {
            const char* form = graph ? "graph" : "stream";
            for (int rep = 0; rep < (graph ? 3 : 4); ++rep) ok = same(run(-1, 0, graph), ref, form, rep, true) && ok;
            for (int k = 0; k < kCalls; ++k) ok = same(run(k, 0, graph), ref, graph ? "graph-gate" : "stream-gate", k, true) && ok;
        }
        std::snprintf(what, sizeof what,
                      "pdl chain rows %u split target %u: stream x4 + gate x%d == serial, graph x3 + gate x%d == serial", rows,
                      split_target, kCalls, kCalls);
        expect(ok, what);
        // Seeded twin behind the gate: must differ from its serial form in every repetition in which it started early.
        for (bool graph : {false, true}) {
            for (int mode : {1, 2}) {
                int caught = 0, early = 0;
                for (int rep = 0; rep < kSeedReps; ++rep) {
                    const bool differs = !same(run(kTwinAt, mode, graph), mode == 2 ? ref2 : ref, "seeded", rep, false);
                    caught += differs ? 1 : 0;
                    early += twin_in ? 1 : 0;
                }
                std::snprintf(what, sizeof what,
                              "pdl chain rows %u split target %u: %s seeded twin %s caught %d/%d, started in poison window %d/%d",
                              rows, split_target, graph ? "graph" : "stream",
                              mode == 1 ? "pre-wait read" : "pre-wait write", caught, kSeedReps, early, kSeedReps);
                expect(caught >= early && (!graph || (caught == kSeedReps && early == kSeedReps)), what);
            }
        }
        set_pdl(false);
        CK(cudaStreamSynchronize(s));  // the poison memsets of the last run are still queued on s
        CK(cudaFree(dx));
        for (float* p : d) CK(cudaFree(p));
        CK(cudaFree(shadow));
        CK(cudaFreeHost(hx));
        CK(cudaFreeHost(ho));
    }
    CK(cudaStreamSynchronize(s));
    cc::nvfp4_release_workspace(s);
    CK(cudaStreamDestroy(s));
    CK(cudaFree(stamps));
    CK(cudaFreeHost(hs));
    CK(cudaFree(gain));
    for (void* p : owned) CK(cudaFree(p));
    cc::nvfp4_set_split_target(0);  // back to the default target
}

// --ncu-probe R: the verify out proj (5120 x 6144, ks 48, + residual) three times, then the hidden-size RMSNorm
// (n 5120, R rows) three times, on random data; for a one-instance profiler run (skip 2 launches of the kernel, count 1).
static void ncu_probe(std::uint32_t rows) {
    std::mt19937 rng(77);
    std::vector<void*> owned;
    const cc::DenseWeight B = make_weight(5120, 6144, rng, owned);
    std::vector<float> h(static_cast<std::size_t>(rows) * 6144);
    for (auto& v : h) v = std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
    float *dx, *dr, *dy, *dg;
    CK(cudaMalloc(&dx, h.size() * 4));
    CK(cudaMemcpy(dx, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dr, static_cast<std::size_t>(rows) * 5120 * 4));
    CK(cudaMemcpy(dr, h.data(), static_cast<std::size_t>(rows) * 5120 * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, static_cast<std::size_t>(rows) * 5120 * 4));
    CK(cudaMalloc(&dg, 5120 * 4));
    CK(cudaMemcpy(dg, h.data(), 5120 * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 3; ++i) CK(cc::nvfp4_dense_rows(B, dx, dy, rows, dr, 0));
    const cc::GroupedView v{rows, 1, 5120, 0, 0};
    for (int i = 0; i < 3; ++i) CK(cc::rmsnorm(dy, v, dg, dr, v, 1e-6f, 0));
    CK(cudaDeviceSynchronize());
    std::printf("ncu probe rows %u: out proj ks %u x3, rmsnorm 5120 x3 done\n", rows, cc::nvfp4_split_k(5120, 6144));
    for (float* q : {dx, dr, dy, dg}) CK(cudaFree(q));
    for (void* q : owned) CK(cudaFree(q));
}

// Cross-kernel weight read-ahead (nvfp4_set_readahead). S = 3 sets of the verify chain out 6144->5120 -> window ->
// gate+up 5120->17408 (x2) -> window -> down 17408->5120 (+resid), 168 MB of weights a set (> 3x L2 over the sets, so a
// set's weights are cold at its turn), captured as ONE graph; the window is a one-block spin of `win_ns` (no memory, as
// the reduce / norm / recurrent / attention windows). Off vs on: every output bitwise equal (a prefetch has no
// architectural effect), and the on capture really holds one read-ahead node per GEMV after the first (non-vacuous).
// Timing: graph replays, us per chain, cold weights.
static void clone_weight(const cc::DenseWeight& src, cc::DenseWeight& dst, std::vector<void*>& owned) {
    dst = src;
    void* p = nullptr;
    CK(cudaMalloc(&p, cc::nvfp4_bytes(src.out, src.in)));
    CK(cudaMemcpy(p, src.data, cc::nvfp4_bytes(src.out, src.in), cudaMemcpyDeviceToDevice));
    owned.push_back(p);
    dst.data = p;
}

static void test_readahead(std::uint32_t rows, unsigned long long win_ns, int reps, bool pdl) {
    constexpr int S = 3;
    std::mt19937 rng(777);
    std::vector<void*> owned;
    struct Set {
        cc::DenseWeight o, g, u, d;
        float *h, *hid, *y;
    };
    Set base{};
    base.o = make_weight(5120, 6144, rng, owned);
    base.g = make_weight(17408, 5120, rng, owned);
    base.u = make_weight(17408, 5120, rng, owned);
    base.d = make_weight(5120, 17408, rng, owned);
    std::vector<Set> sets(S);
    for (int i = 0; i < S; ++i) {
        if (i == 0) {
            sets[0] = base;
        } else {
            clone_weight(base.o, sets[i].o, owned);
            clone_weight(base.g, sets[i].g, owned);
            clone_weight(base.u, sets[i].u, owned);
            clone_weight(base.d, sets[i].d, owned);
        }
        CK(cudaMalloc(&sets[i].h, static_cast<std::size_t>(rows) * 5120 * 4));
        CK(cudaMalloc(&sets[i].hid, static_cast<std::size_t>(rows) * 17408 * 4));
        CK(cudaMalloc(&sets[i].y, static_cast<std::size_t>(rows) * 5120 * 4));
    }
    std::vector<float> hx(static_cast<std::size_t>(rows) * 6144);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : hx) v = nd(rng);
    float* dx = nullptr;
    CK(cudaMalloc(&dx, hx.size() * 4));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cc::set_programmatic_launch(pdl);
    auto chain = [&]() {
        for (const Set& t : sets) {
            CK(cc::nvfp4_dense_rows(t.o, dx, t.h, rows, nullptr, s));
            pdlh::probe_holder<<<1, 32, 0, s>>>(win_ns);
            CK(cc::nvfp4_gate_up_silu_rows(t.g, t.u, t.h, t.hid, rows, s));
            pdlh::probe_holder<<<1, 32, 0, s>>>(win_ns);
            CK(cc::nvfp4_dense_rows(t.d, t.hid, t.y, rows, t.h, s));
        }
    };
    // One eager pass before any capture, as the engine does (it reserves its split-K workspace first): the first GEMV on a
    // stream allocates that stream's workspace, which a capturing stream refuses ("operation not permitted when stream is
    // capturing"). The eager pass captures nothing, so it leaves no read-ahead site behind.
    cc::nvfp4_set_readahead(0);
    chain();
    CK(cudaStreamSynchronize(s));
    struct Arm {
        std::size_t nodes = 0;
        std::uint64_t bytes = 0;
        double us = 0;
        std::vector<float> out;
    } arm[2];
    for (int m = 0; m < 2; ++m) {
        cc::nvfp4_set_readahead(m);
        (void)cc::nvfp4_take_readahead_bytes();
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        chain();
        CK(cudaStreamEndCapture(s, &g));
        arm[m].bytes = cc::nvfp4_take_readahead_bytes();
        CK(cudaGraphGetNodes(g, nullptr, &arm[m].nodes));
        cudaGraphExec_t ge;
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (const Set& t : sets)
            for (float* q : {t.h, t.hid, t.y}) CK(cudaMemsetAsync(q, 0xFF, rows * 4, s));  // poison the first row
        CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        for (const Set& t : sets) {
            const std::size_t n0 = arm[m].out.size();
            arm[m].out.resize(n0 + static_cast<std::size_t>(rows) * (5120 + 17408 + 5120));
            float* o = arm[m].out.data() + n0;
            CK(cudaMemcpy(o, t.h, static_cast<std::size_t>(rows) * 5120 * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(o + rows * 5120, t.hid, static_cast<std::size_t>(rows) * 17408 * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(o + rows * (5120 + 17408), t.y, static_cast<std::size_t>(rows) * 5120 * 4, cudaMemcpyDeviceToHost));
        }
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        CK(cudaGraphLaunch(ge, s));  // warm the instantiation
        CK(cudaEventRecord(e0, s));
        for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        arm[m].us = 1000.0 * ms / (reps * S);
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
    }
    // Thread scope: global on, this thread's scope off -> the capture adds no side node; the scope's end restores it.
    std::size_t scoped_nodes = 0, restored_nodes = 0;
    std::uint64_t scoped_bytes = 1;
    for (int phase = 0; phase < 2; ++phase) {
        cc::nvfp4_set_readahead(1);
        (void)cc::nvfp4_take_readahead_bytes();
        cudaGraph_t g;
        {
            const cc::Nvfp4ReadaheadScope off(phase == 0 ? 0 : -1);
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            chain();
            CK(cudaStreamEndCapture(s, &g));
        }
        if (phase == 0) scoped_bytes = cc::nvfp4_take_readahead_bytes();
        CK(cudaGraphGetNodes(g, nullptr, phase == 0 ? &scoped_nodes : &restored_nodes));
        CK(cudaGraphDestroy(g));
    }
    expect(scoped_bytes == 0 && scoped_nodes == arm[0].nodes && restored_nodes == arm[1].nodes,
           "read-ahead: a thread scope of 0 captures no side node; the global mode applies again after it");
    cc::nvfp4_set_readahead(-1);
    cc::set_programmatic_launch(true);
    const bool same = arm[0].out.size() == arm[1].out.size() &&
                      std::memcmp(arm[0].out.data(), arm[1].out.data(), arm[0].out.size() * 4) == 0;
    // Consumers above the stream-bound rows get no side node (strix/decode_weight_sweep.hpp readahead_rows_ok).
    const bool eligible = rows <= cc::kNvfp4ReadaheadMaxRows;
    const bool added = eligible ? arm[0].bytes == 0 && arm[1].bytes > 0 && arm[1].nodes == arm[0].nodes + (3 * S - 1)
                                : arm[0].bytes == 0 && arm[1].bytes == 0 && arm[1].nodes == arm[0].nodes;
    expect(same, "read-ahead: graph outputs bitwise equal off vs on");
    expect(added, eligible ? "read-ahead: one side node per GEMV after the first (on), none (off)"
                           : "read-ahead: no side node for consumers above the stream-bound rows");
    std::printf("RA rows %u win %llu us pdl %d: nodes off %zu on %zu, last head %.1f MB | bitwise %s | off %.2f us on %.2f us "
                "per chain (%+.1f %%)\n",
                rows, win_ns / 1000, pdl ? 1 : 0, arm[0].nodes, arm[1].nodes, arm[1].bytes / 1048576.0, same ? "yes" : "NO",
                arm[0].us, arm[1].us, 100.0 * (arm[1].us - arm[0].us) / arm[0].us);
    CK(cudaStreamDestroy(s));
    CK(cudaFree(dx));
    for (const Set& t : sets)
        for (float* q : {t.h, t.hid, t.y}) CK(cudaFree(q));
    for (void* q : owned) CK(cudaFree(q));
}

int main(int argc, char** argv) {
    const std::string a = argc > 1 ? argv[1] : "";
    if (a == "--ncu-probe") {
        CK(cudaSetDevice(0));
        ncu_probe(argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 8u);
        return 0;
    }
    if (a == "--readahead") {  // [rows] [window us] [reps]: both PDL forms, bitwise + timing
        CK(cudaSetDevice(0));
        const std::uint32_t rows = argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 8u;
        const unsigned long long win = argc > 3 ? 1000ull * static_cast<unsigned long long>(std::atoi(argv[3])) : 20000ull;
        const int reps = argc > 4 ? std::atoi(argv[4]) : 20;
        test_readahead(rows, win, reps, false);
        test_readahead(rows, win, reps, true);
        std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    // argv[2]: a split target (EXACT policy) or "fast" (the FAST split policy of strix/split_k_policy.hpp for device 0)
    const bool fast_split = argc > 2 && std::string(argv[2]) == "fast";
    if (argc > 2 && !fast_split) cc::nvfp4_set_split_target(static_cast<std::uint32_t>(std::atoi(argv[2])));
    CK(cudaSetDevice(0));
    if (fast_split) CK(cc::nvfp4_set_split_profile(true));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device 0: %s cc %d.%d SMs %d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    std::printf("split policy: %s\n", cc::nvfp4_split_identity());
    if (a != "--bench-only") {
        std::mt19937 rng(1234);
        test_shape(256, 512, rng);
        test_shape(48, 5120, rng);
        test_shape(1024, 6144, rng);
        test_shape(5120, 17408, rng);
        if (fast_split) {  // shapes with a FAST row: host reference + rows 1..32 invariance at the FAST split factor
            std::printf("fast split: ks 5120x6144 %u, 6144x5120 %u\n", cc::nvfp4_split_k(5120, 6144), cc::nvfp4_split_k(6144, 5120));
            test_shape(5120, 6144, rng);
            test_shape(6144, 5120, rng);
        }
        if (argc <= 2) {  // the chain sets its own split targets (argv[2] overrides the target for the shapes/bench)
            std::mt19937 rng2(4321);
            test_pdl_chain(1000, rng2);
            test_pdl_chain(1, rng2);
            test_readahead(8, 20000ull, 3, false);
        }
        std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    }
    if (a != "--test-only") bench();
    return g_fail ? 1 : 0;
}
