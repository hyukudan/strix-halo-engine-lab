// NVFP4 (TURBO weights): converter vs host RTN reference, GEMV vs host dequant reference, rows 1..32 bitwise row
// invariance, and a bandwidth microbench on the 27B shapes. Run: cuda_core_nvfp4_test [--bench-only|--test-only].
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

// Programmatic dependent launch + L2 prefetch on fp4_gemv / fp4_reduce are scheduling only: a dependent chain
// A(x) -> B(a) + x -> gate/up(b) -> A(b) -> B(a) + b -> gate/up(b2) run with PDL on (stream, then a CUDA graph replayed
// three times) must equal the serialised form (PDL off) bit for bit; rows 1/8/16/24/32, ks 40/48/40 and ks 1 (split target 1),
// plain and with a spin kernel forcing the pre-wait overlap at each of the six calls. Why this is complete: common.cuh.
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

// Test-only PDL helpers. g_test_pdl mirrors cc::set_programmatic_launch for the test kernels below.
static bool g_test_pdl = false;
__device__ __forceinline__ void t_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void t_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void t_spin(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); while (t - t0 < ns);
}
// Releases its successor at entry, waits for its predecessor (keeps the chain transitive), then spins: every kernel behind
// it that may run code before its own wait runs that code now, while nothing after this point has been written yet.
__global__ void pdl_spin_kernel(unsigned long long ns) {
    t_trigger();
    t_wait();
    t_spin(ns);
}
// Negative control: the writer releases at entry and writes after a spin; reader_early reads BEFORE its wait (contract
// violation), reader_late after it.
__global__ void hz_writer(float* buf, unsigned long long ns) {
    t_trigger();
    t_wait();
    t_spin(ns);
    buf[0] = 1.f;
}
__global__ void hz_reader(const float* buf, float* out, int early) {
    float v = early ? *static_cast<const volatile float*>(buf) : 0.f;
    t_trigger();
    t_wait();
    out[0] = early ? v : buf[0];
}
template <class... K, class... A>
static cudaError_t launch_test(void (*k)(K...), cudaStream_t s, A... a) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(32);
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = g_test_pdl ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, k, static_cast<K>(a)...);
}

// The stress below can only prove something if a pre-wait read of a predecessor's output IS caught: with PDL the early reader
// must see the old value (0), the late reader and the serial form the new one (1).
static void pdl_hazard_control() {
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float *buf, *out;
    CK(cudaMalloc(&buf, 4));
    CK(cudaMalloc(&out, 4));
    auto run = [&](bool pdl, int early) {
        g_test_pdl = pdl;
        CK(cudaMemsetAsync(buf, 0, 4, s));
        CK(cudaMemsetAsync(out, 0xFF, 4, s));
        CK(launch_test(hz_writer, s, buf, 200000ull));
        CK(launch_test(hz_reader, s, static_cast<const float*>(buf), out, early));
        float v = -1.f;
        CK(cudaMemcpyAsync(&v, out, 4, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        g_test_pdl = false;
        return v;
    };
    const float serial_early = run(false, 1), pdl_early = run(true, 1), pdl_late = run(true, 0);
    std::printf("-- pdl hazard control: serial early %g, pdl early %g, pdl late %g\n", static_cast<double>(serial_early),
                static_cast<double>(pdl_early), static_cast<double>(pdl_late));
    expect(serial_early == 1.f && pdl_early == 0.f && pdl_late == 1.f,
           "pdl hazard control: a read before the wait is caught (pdl early 0), after the wait is not (1)");
    CK(cudaFree(buf));
    CK(cudaFree(out));
    CK(cudaStreamDestroy(s));
}

static void test_pdl_chain(std::uint32_t split_target, std::mt19937& rng) {
    static bool control_done = false;
    if (!control_done) {
        control_done = true;
        pdl_hazard_control();
    }
    constexpr std::uint32_t kH = 5120, kI = 6144, kF = 1024;
    std::vector<void*> owned;
    const cc::DenseWeight A = make_weight(kI, kH, rng, owned), B = make_weight(kH, kI, rng, owned);
    const cc::DenseWeight G = make_weight(kF, kH, rng, owned), U = make_weight(kF, kH, rng, owned);
    cc::nvfp4_set_split_target(split_target);
    std::printf("-- pdl chain: split target %u, ks A %u B %u gate/up %u\n", split_target, cc::nvfp4_split_k(kI, kH),
                cc::nvfp4_split_k(kH, kI), cc::nvfp4_split_k(kF, kH));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (std::uint32_t rows : {1u, 8u, 16u, 24u, 32u}) {  // every fp4_gemv template (<1,4> <2,4> <3,2> <4,2>)
        std::uniform_real_distribution<float> ud(-1.f, 1.f);
        std::vector<float> x(static_cast<std::size_t>(rows) * kH);
        for (auto& v : x) v = ud(rng);
        const std::size_t na = static_cast<std::size_t>(rows) * kI, nb = static_cast<std::size_t>(rows) * kH,
                          nc = static_cast<std::size_t>(rows) * kF;
        float *dx, *da, *db, *db2, *dc;
        CK(cudaMalloc(&dx, nb * 4));
        CK(cudaMalloc(&da, na * 4));
        CK(cudaMalloc(&db, nb * 4));
        CK(cudaMalloc(&db2, nb * 4));
        CK(cudaMalloc(&dc, nc * 4));
        // Everything on s and synchronised: s is NonBlocking, so a legacy-stream cudaMemcpy from pageable memory is NOT
        // ordered before work on s (its DMA may still be in flight when it returns). That let the serial reference read dx
        // before it landed (RTX chains 10:40 / 11:10: serial NaN or different in every element, PDL runs stable).
        CK(cudaMemcpyAsync(dx, x.data(), nb * 4, cudaMemcpyHostToDevice, s));
        auto poison = [&]() {  // a skipped write, or a read before the producer wrote, cannot pass
            CK(cudaMemsetAsync(da, 0xFF, na * 4, s));
            CK(cudaMemsetAsync(db, 0xFF, nb * 4, s));
            CK(cudaMemsetAsync(db2, 0xFF, nb * 4, s));
            CK(cudaMemsetAsync(dc, 0xFF, nc * 4, s));
        };
        poison();
        CK(cudaStreamSynchronize(s));
        // spin_at k (0..5): a PDL spin kernel before call k, so every kernel from k on that may run before its wait does so
        // while k's inputs are still being produced / poisoned (forced overlap, not left to timing). -1: plain chain.
        auto chain = [&](int spin_at) {
            auto spin = [&](int here) {
                if (spin_at == here) CK(launch_test(pdl_spin_kernel, s, 50000ull));
            };
            spin(0);
            CK(cc::nvfp4_dense_rows(A, dx, da, rows, nullptr, s));
            spin(1);
            CK(cc::nvfp4_dense_rows(B, da, db, rows, dx, s));
            spin(2);
            CK(cc::nvfp4_gate_up_silu_rows(G, U, db, dc, rows, s));
            spin(3);
            CK(cc::nvfp4_dense_rows(A, db, da, rows, nullptr, s));
            spin(4);
            CK(cc::nvfp4_dense_rows(B, da, db2, rows, db, s));
            spin(5);
            CK(cc::nvfp4_gate_up_silu_rows(G, U, db2, dc, rows, s));
        };
        auto snap = [&]() {
            std::vector<float> o(na + 2 * nb + nc);
            CK(cudaMemcpyAsync(o.data(), da, na * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(o.data() + na, db, nb * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(o.data() + na + nb, db2, nb * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(o.data() + na + 2 * nb, dc, nc * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            poison();
            return o;
        };
        // Compares one run with the serial reference; on a difference prints, per output buffer (da = K1/K4, db = K2,
        // db2 = K5, dc = K3r/K6r), the number of differing elements and the first one (row, column, bits).
        auto same = [&](const std::vector<float>& p, const std::vector<float>& q, const char* side, int rep) {
            if (p.size() == q.size() && std::memcmp(p.data(), q.data(), p.size() * 4) == 0) return true;
            const char* names[4] = {"da", "db", "db2", "dc"};
            const std::size_t lo[5] = {0, na, na + nb, na + 2 * nb, na + 2 * nb + nc};
            const std::uint32_t cols[4] = {kI, kH, kH, kF};
            for (int b = 0; b < 4; ++b) {
                std::size_t ndiff = 0, first = 0;
                for (std::size_t i = lo[b]; i < lo[b + 1]; ++i)
                    if (std::memcmp(&p[i], &q[i], 4) != 0 && ndiff++ == 0) first = i - lo[b];
                if (ndiff == 0) {
                    std::printf("   diff rows %u %s rep %d %s: equal\n", rows, side, rep, names[b]);
                    continue;
                }
                std::uint32_t got = 0, want = 0;
                std::memcpy(&got, &p[lo[b] + first], 4);
                std::memcpy(&want, &q[lo[b] + first], 4);
                std::printf("   diff rows %u %s rep %d %s: %zu of %zu differ, first [row %zu col %zu] serial %08x got %08x\n",
                            rows, side, rep, names[b], ndiff, lo[b + 1] - lo[b], first / cols[b], first % cols[b], want, got);
            }
            return false;
        };
        auto set_pdl = [](bool on) {
            cc::set_programmatic_launch(on);
            g_test_pdl = on;
        };
        set_pdl(false);
        chain(-1);
        const std::vector<float> ref = snap();
        chain(2);  // the serial form with a spin in the middle: a second, independent reference
        const bool ref_ok = same(snap(), ref, "serial2", 0);
        char what[200];
        std::snprintf(what, sizeof what, "pdl chain rows %u split target %u: serial reference stable (x2)", rows, split_target);
        expect(ref_ok, what);
        set_pdl(true);
        bool stream_ok = true;
        for (int rep = 0; rep < 4; ++rep) {
            chain(-1);
            stream_ok = same(snap(), ref, "stream", rep) && stream_ok;
        }
        for (int at = 0; at < 6; ++at) {  // forced overlap: spin before each call in turn
            chain(at);
            stream_ok = same(snap(), ref, "stream-spin", at) && stream_ok;
        }
        bool graph_ok = true;
        for (int at = -1; at < 6; ++at) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            chain(at);
            CK(cudaStreamEndCapture(s, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            for (int rep = 0; rep < (at < 0 ? 3 : 1); ++rep) {
                CK(cudaGraphLaunch(ge, s));
                graph_ok = same(snap(), ref, at < 0 ? "graph" : "graph-spin", at < 0 ? rep : at) && graph_ok;
            }
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(g));
        }
        std::snprintf(what, sizeof what,
                      "pdl chain rows %u split target %u: stream x4 + spin x6 == serial, graph x3 + spin x6 == serial", rows,
                      split_target);
        expect(stream_ok && graph_ok, what);
        CK(cudaStreamSynchronize(s));  // the poison memsets of the last snap() are still queued on s
        for (float* p : {dx, da, db, db2, dc}) CK(cudaFree(p));
    }
    CK(cudaStreamSynchronize(s));
    cc::nvfp4_release_workspace(s);
    CK(cudaStreamDestroy(s));
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

int main(int argc, char** argv) {
    const std::string a = argc > 1 ? argv[1] : "";
    if (a == "--ncu-probe") {
        CK(cudaSetDevice(0));
        ncu_probe(argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 8u);
        return 0;
    }
    if (argc > 2) cc::nvfp4_set_split_target(static_cast<std::uint32_t>(std::atoi(argv[2])));
    CK(cudaSetDevice(0));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device 0: %s cc %d.%d SMs %d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    if (a != "--bench-only") {
        std::mt19937 rng(1234);
        test_shape(256, 512, rng);
        test_shape(48, 5120, rng);
        test_shape(1024, 6144, rng);
        test_shape(5120, 17408, rng);
        if (argc <= 2) {  // the chain sets its own split targets (argv[2] overrides the target for the shapes/bench)
            std::mt19937 rng2(4321);
            test_pdl_chain(1000, rng2);
            test_pdl_chain(1, rng2);
        }
        std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    }
    if (a != "--test-only") bench();
    return g_fail ? 1 : 0;
}
