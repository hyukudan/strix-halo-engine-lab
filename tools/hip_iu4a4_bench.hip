// Timing of the shared A4W4 class (core/iu4a4_dense.hpp, FAST, dense-iu4a4-rot-v1) against the EXACT IU4 g64 W4A16
// production entries of the FP16 tall core (iu4a16 enqueue prod16 / prod8, x16 tiles + enqueue_x16), same process,
// same shapes. Values are random (timing only; the device fixture tests/hip_iu4a4_dense_tests.hip checks the bits).
//   hip-iu4a4-bench <rows> in:n [in:n ...]
//   hip-iu4a4-bench rate      (register-only rate loops, see rate_kernel)
// Per shape: one line per arm (us, TOPS), then "ratio" = best EXACT entry / (quantise + gemm).
#include "core/f16_dense_iu4.hpp"
#include "core/iu4a4_dense.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace iu = strix::core::f16_dense_tall::iu4a16;
namespace tall = strix::core::f16_dense_tall;
namespace ad = strix::core::iu4a4_dense;
#define CK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

template <class F> double time_us(F&& launch, int reps = 20) {
    for (int i = 0; i < 3; ++i) launch();
    CK(hipDeviceSynchronize());
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a, nullptr));
    for (int i = 0; i < reps; ++i) launch();
    CK(hipEventRecord(b, nullptr));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipGetLastError());
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return 1000.0 * ms / reps;
}
// rate mode (no model): register-only loops on random or constant operand bits, 8 tiles (4 W x 2 x) per wave as the
// GEMM. Kind 0: iu4 WMMA chain only (64 WMMA per iteration). Kind 1: the GEMM's per-block contract arithmetic (C
// restarted per A group, mul / mad_i24 per group, cvt + fma per block with u / steps from an L1-resident table): the
// GEMM with its fragment loads removed. asm "+v" keeps the operands opaque per iteration (no hoisting, no instructions).
template <int Kind> __global__ __launch_bounds__(256) void rate_kernel(int iters, const int* __restrict__ frag, const unsigned* __restrict__ utab,
                                                                   const float* __restrict__ stab, float* out) {
    using I2 = int __attribute__((ext_vector_type(2)));
    using I8 = int __attribute__((ext_vector_type(8)));
    const unsigned t = threadIdx.x, lane = t & 31U;
    I2 wa[4], xa[2], wb[4], xb[2];
    for (unsigned i = 0; i < 4U; ++i) wa[i] = I2{frag[(i * 256U + t) * 2U], frag[(i * 256U + t) * 2U + 1U]}, wb[i] = I2{frag[((i + 4U) * 256U + t) * 2U], frag[((i + 4U) * 256U + t) * 2U + 1U]};
    for (unsigned j = 0; j < 2U; ++j) xa[j] = I2{frag[((j + 8U) * 256U + t) * 2U], frag[((j + 8U) * 256U + t) * 2U + 1U]}, xb[j] = I2{frag[((j + 10U) * 256U + t) * 2U], frag[((j + 10U) * 256U + t) * 2U + 1U]};
    const I8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    I8 c[4][2];
    float acc[4][2][8];
    for (unsigned i = 0; i < 4U; ++i)
        for (unsigned j = 0; j < 2U; ++j) {
            c[i][j] = zero;
            for (unsigned e = 0; e < 8U; ++e) acc[i][j][e] = 0.0F;
        }
    const auto opaque = [&]() __attribute__((always_inline)) {  // per A group: no CSE of equal groups, no hoisting
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) asm volatile("" : "+v"(wa[i]), "+v"(wb[i]));
#pragma unroll
        for (unsigned j = 0; j < 2U; ++j) asm volatile("" : "+v"(xa[j]), "+v"(xb[j]));
    };
    for (int it = 0; it < iters; ++it) {
        if constexpr (Kind == 0) {
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                opaque();
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 2U; ++j) {
                        c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wa[i], true, xa[j], c[i][j], false);
                        c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wb[i], true, xb[j], c[i][j], false);
                    }
            }
        } else {
            const unsigned row = unsigned(it) & 63U;
            const unsigned ub[2] = {utab[row * 64U + lane], utab[row * 64U + 32U + lane]};
            I8 tt[4][2];
#pragma unroll
            for (unsigned g = 0; g < 4U; ++g) {
                opaque();
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 2U; ++j) {
                        c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wa[i], true, xa[j], zero, false);
                        c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wb[i], true, xb[j], c[i][j], false);
                    }
#pragma unroll
                for (unsigned j = 0; j < 2U; ++j) {
                    const int ug = static_cast<int>((ub[j] >> (8U * g)) & 255U);
#pragma unroll
                    for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) tt[i][j][e] = g == 0U ? __mul24(ug, c[i][j][e]) : __mul24(ug, c[i][j][e]) + tt[i][j][e];
                }
            }
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                const float* s = stab + (row * 4U + i) * 256U + (lane >> 4U) * 8U;
#pragma unroll
                for (unsigned j = 0; j < 2U; ++j)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) acc[i][j][e] = __builtin_fmaf(static_cast<float>(tt[i][j][e]), s[e], acc[i][j][e]);
            }
        }
    }
    float r = 0.0F;
    for (unsigned i = 0; i < 4U; ++i)
        for (unsigned j = 0; j < 2U; ++j)
            for (unsigned e = 0; e < 8U; ++e) r += Kind == 0 ? float(c[i][j][e]) : acc[i][j][e];  // every output live (no DCE)
    if (r == 1234.5F) out[t] = r;
}

int rate_main() {
    int* frag = nullptr;
    unsigned* utab = nullptr;
    float *stab = nullptr, *out = nullptr;
    const std::size_t nfrag = 12U * 256U * 2U, nu = 64U * 64U, ns = 64U * 4U * 256U;
    CK(hipMalloc(&frag, nfrag * 4U));
    CK(hipMalloc(&utab, nu * 4U));
    CK(hipMalloc(&stab, ns * 4U));
    CK(hipMalloc(&out, 256U * 4U));
    std::mt19937 g(11U);
    const auto fill = [&](bool rnd) {
        std::vector<int> hf(nfrag);
        std::vector<unsigned> hu(nu);
        std::vector<float> hs(ns);
        std::uniform_real_distribution<float> d(0.5F, 2.0F);
        for (auto& e : hf) e = rnd ? int(g()) : 0x11111111;
        for (auto& e : hu) e = rnd ? (g() % 255U + 1U) | (g() % 255U + 1U) << 8U | (g() % 255U + 1U) << 16U | (g() % 255U + 1U) << 24U : 0x01010101U;
        for (auto& e : hs) e = rnd ? d(g) : 1.0F;
        CK(hipMemcpy(frag, hf.data(), nfrag * 4U, hipMemcpyHostToDevice));
        CK(hipMemcpy(utab, hu.data(), nu * 4U, hipMemcpyHostToDevice));
        CK(hipMemcpy(stab, hs.data(), ns * 4U, hipMemcpyHostToDevice));
    };
    const unsigned ctas = 320U;
    const int iters = 1500;
    const double tops_per = double(ctas) * 8.0 * iters * 64.0 * 8192.0 / 1e12;  // ops of one launch, in T
    for (int round = 0; round < 2; ++round)
        for (int kind = 0; kind < 2; ++kind)
            for (int rnd = 1; rnd >= 0; --rnd) {
                fill(rnd != 0);
                const double us = time_us([&] {
                    if (kind == 0) hipLaunchKernelGGL(rate_kernel<0>, dim3(ctas), dim3(256), 0, nullptr, iters, frag, utab, stab, out);
                    else hipLaunchKernelGGL(rate_kernel<1>, dim3(ctas), dim3(256), 0, nullptr, iters, frag, utab, stab, out);
                }, 5);
                std::printf("rate round %d kind %d (%s) data %-5s %9.1f us %6.2f TOPS\n", round, kind, kind == 0 ? "wmma-only " : "contract  ",
                            rnd ? "rand" : "const", us, tops_per / us * 1e6);
            }
    for (void* p : {static_cast<void*>(frag), static_cast<void*>(utab), static_cast<void*>(stab), static_cast<void*>(out)}) CK(hipFree(p));
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "rate") return rate_main();
    if (argc < 3) {
        std::printf("usage: hip-iu4a4-bench <rows> in:n [in:n ...]\n");
        return 2;
    }
    const unsigned rows = unsigned(std::atoi(argv[1]));
    std::vector<std::pair<unsigned, unsigned>> shapes;
    unsigned max_in = 0, max_n = 0;
    for (int i = 2; i < argc; ++i) {
        unsigned in = 0, n = 0;
        if (std::sscanf(argv[i], "%u:%u", &in, &n) != 2 || !ad::supported(in, n)) {
            std::printf("bad shape '%s'\n", argv[i]);
            return 2;
        }
        shapes.emplace_back(in, n);
        max_in = std::max(max_in, in);
        max_n = std::max(max_n, n);
    }
    const std::size_t max_w = std::size_t(max_n) * max_in / 2U + std::size_t(max_n) * (max_in / 64U) * 2U;
    std::uint8_t *w = nullptr, *w4 = nullptr, *codes = nullptr;
    float *x = nullptr, *y = nullptr, *step = nullptr, *base = nullptr;
    _Float16* xt = nullptr;
    std::uint32_t* u = nullptr;
    CK(hipMalloc(&w, max_w));
    CK(hipMalloc(&w4, ad::weight_code_bytes(max_in, max_n)));
    CK(hipMalloc(&step, ad::weight_step_bytes(max_in, max_n)));
    CK(hipMalloc(&x, std::size_t(rows) * max_in * 4U));
    CK(hipMalloc(&y, std::size_t(rows) * max_n * 4U));
    CK(hipMalloc(&xt, std::size_t(rows) * max_in * 2U));
    CK(hipMalloc(&codes, ad::act_code_bytes(rows, max_in)));
    CK(hipMalloc(&u, ad::act_u_bytes(rows, max_in)));
    CK(hipMalloc(&base, ad::act_base_bytes(rows)));
    {
        std::mt19937 g(7U);
        std::vector<std::uint8_t> hw(max_w);
        for (auto& e : hw) e = std::uint8_t(g());
        CK(hipMemcpy(w, hw.data(), max_w, hipMemcpyHostToDevice));
        std::vector<float> hx(std::size_t(rows) * max_in);
        std::uniform_real_distribution<float> d(-1.0F, 1.0F);
        for (auto& e : hx) e = d(g);
        CK(hipMemcpy(x, hx.data(), hx.size() * 4U, hipMemcpyHostToDevice));
    }
    {
        int occ = 0;
        CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, reinterpret_cast<const void*>(&ad::detail::gemm_kernel<ad::OutCols, tall::RawStore>), 256, 0));
        hipDeviceProp_t prop;
        CK(hipGetDeviceProperties(&prop, 0));
        std::printf("gemm occupancy %d CTAs per CU (LDS per CU %zu B, per CTA max %zu B)\n", occ, std::size_t(prop.maxSharedMemoryPerMultiProcessor),
                    std::size_t(prop.sharedMemPerBlock));
    }
    for (const auto& [in, n] : shapes) {
        const std::size_t packed = std::size_t(n) * in / 2U, sbytes = std::size_t(n) * (in / 64U) * 2U;
        {
            std::vector<std::uint16_t> hs(sbytes / 2U, 0x2c00U);
            CK(hipMemcpy(w + packed, hs.data(), sbytes, hipMemcpyHostToDevice));
        }
        const auto* scale = reinterpret_cast<const std::uint16_t*>(w + packed);
        const double gop = 2.0 * rows * double(in) * n / 1e3;
        auto pr = [&](const char* arm, double us) {
            std::printf("bench in %5u n %5u rows %5u %-14s %9.1f us %6.2f TOPS\n", in, n, rows, arm, us, gop / us / 1e3);
            return us;
        };
        iu::View v{w, scale, x, y, in, n, n, rows, n, 0U, 0U};
        const double p16 = pr("exact-prod16", time_us([&] { iu::enqueue(v, nullptr, 16U); }));
        const double p8 = pr("exact-prod8", time_us([&] { iu::enqueue(v, nullptr, 8U); }));
        const double px = pr("exact-x16", time_us([&] {
            tall::enqueue_x16_tiles(x, xt, rows, in, nullptr);
            iu::enqueue_x16(v, xt, nullptr);
        }));
        const double best = std::min({p16, p8, px});
        // GEMM-only yardstick: the x16 entry without its tile pass (tiles made once) and the tile pass alone; the A4W4
        // counterparts are a4w4-gemm and a4w4-quantise (each producer pass is what an engine fuses upstream).
        tall::enqueue_x16_tiles(x, xt, rows, in, nullptr);
        const double pxg = pr("exact-x16-gemm", time_us([&] { iu::enqueue_x16(v, xt, nullptr); }));
        const double pxt = time_us([&] { tall::enqueue_x16_tiles(x, xt, rows, in, nullptr); });
        std::printf("bench in %5u n %5u rows %5u %-14s %9.1f us\n", in, n, rows, "exact-x16-tiles", pxt);
        const double bestg = std::min({p16, p8, pxg});
        const double rq = time_us([&] { ad::enqueue_requantise(w, scale, in, n, w4, step, nullptr); }, 1);
        std::printf("bench in %5u n %5u rows %5u %-14s %9.1f us (load time, once per matrix)\n", in, n, rows, "a4w4-requant", rq);
        const ad::Act act{codes, u, base, rows, in};
        const ad::Weights wt{w4, step, in, n};
        const ad::OutCols oc{y, n, n / 16U};
        const double q = time_us([&] { ad::enqueue_quantise(x, in, act, nullptr); });
        std::printf("bench in %5u n %5u rows %5u %-14s %9.1f us\n", in, n, rows, "a4w4-quantise", q);
        const double g = pr("a4w4-gemm", time_us([&] { ad::enqueue_gemm(wt, act, oc, nullptr); }));
        const double qg = pr("a4w4-quant+gemm", time_us([&] {
            ad::enqueue_quantise(x, in, act, nullptr);
            ad::enqueue_gemm(wt, act, oc, nullptr);
        }));
        std::printf("ratio in %5u n %5u rows %5u best-exact %.1f us / a4w4 %.1f us = %.3fx\n", in, n, rows, best, qg, best / qg);
        std::printf("ratio-gemm in %5u n %5u rows %5u best-exact-gemm %.1f us / a4w4-gemm %.1f us = %.3fx\n", in, n, rows, bestg, g, bestg / g);
    }
    for (void* p : {static_cast<void*>(w), static_cast<void*>(w4), static_cast<void*>(step), static_cast<void*>(x), static_cast<void*>(y),
                    static_cast<void*>(xt), static_cast<void*>(codes), static_cast<void*>(u), static_cast<void*>(base)})
        CK(hipFree(p));
    return 0;
}
