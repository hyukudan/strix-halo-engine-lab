// Timing of the shared A4W4 class (core/iu4a4_dense.hpp, FAST, dense-iu4a4-rot-v1) against the EXACT IU4 g64 W4A16
// production entries of the FP16 tall core (iu4a16 enqueue prod16 / prod8, x16 tiles + enqueue_x16), same process,
// same shapes. Values are random (timing only; the device fixture tests/hip_iu4a4_dense_tests.hip checks the bits).
//   hip-iu4a4-bench <rows> in:n [in:n ...]
// Per shape: one line per arm (us, TOPS), then "ratio" = best EXACT entry / (quantise + gemm).
#include "core/f16_dense_iu4.hpp"
#include "core/iu4a4_dense.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
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
}  // namespace

int main(int argc, char** argv) {
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
        const double rq = time_us([&] { ad::enqueue_requantise(w, scale, in, n, w4, step, nullptr); }, 1);
        std::printf("bench in %5u n %5u rows %5u %-14s %9.1f us (load time, once per matrix)\n", in, n, rows, "a4w4-requant", rq);
        const ad::Act act{codes, u, base, rows, in};
        const ad::Weights wt{w4, step, in, n};
        const ad::OutCols oc{y, n, n / 16U};
        const double q = time_us([&] { ad::enqueue_quantise(x, in, act, nullptr); });
        std::printf("bench in %5u n %5u rows %5u %-14s %9.1f us\n", in, n, rows, "a4w4-quantise", q);
        pr("a4w4-gemm", time_us([&] { ad::enqueue_gemm(wt, act, oc, nullptr); }));
        const double qg = pr("a4w4-quant+gemm", time_us([&] {
            ad::enqueue_quantise(x, in, act, nullptr);
            ad::enqueue_gemm(wt, act, oc, nullptr);
        }));
        std::printf("ratio in %5u n %5u rows %5u best-exact %.1f us / a4w4 %.1f us = %.3fx\n", in, n, rows, best, qg, best / qg);
    }
    for (void* p : {static_cast<void*>(w), static_cast<void*>(w4), static_cast<void*>(step), static_cast<void*>(x), static_cast<void*>(y),
                    static_cast<void*>(xt), static_cast<void*>(codes), static_cast<void*>(u), static_cast<void*>(base)})
        CK(hipFree(p));
    return 0;
}
