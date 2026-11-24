// Device self-test of the shared hi/lo panel launch (core/f16_dense_tall_hilo.hpp): panel addressing, the zero-padded
// final 8-unit iteration (units not a multiple of 8) and partial output tiles, blocked and unblocked merge, against an
// independent FP64 model of y = x . w over the stored panels. Weight, activation and output buffers are allocated
// exactly (no slack), so a read or write outside the prepared layout is an out-of-bounds access. Needs a GPU.
// The WMMA order is not modelled, so the bound is a few FP32 ulps of sum |x w| plus the split residual (2^-20 |x|).
// Exit 0 = PASS (prints "f16 hilo panels: 0 failures").
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/f16_dense_tall_hilo.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace hilo = strix::core::f16_dense_tall::hilo;

namespace {

constexpr float kScale = 4096.0F;
constexpr float kUnscale = 1.0F / 4096.0F;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

struct Rng {
    std::uint64_t s = 0x9E3779B97F4A7C15ULL;
    float next() {
        s ^= s << 13U; s ^= s >> 7U; s ^= s << 17U;
        return static_cast<float>((s >> 11U) % 2000001U) / 1000000.0F - 1.0F;
    }
};

// Returns the number of failing elements (and the stored-value failure word through `flag`).
template <unsigned Block> unsigned run_case(unsigned rows, unsigned outputs, unsigned units, unsigned* flag) {
    const unsigned stride = 16U * units, tiles = (outputs + 15U) / 16U;
    Rng rng;
    std::vector<float> xs(std::size_t{rows} * stride), ws(std::size_t{outputs} * stride);
    for (auto& v : xs) v = rng.next();
    for (auto& v : ws) v = static_cast<float>(static_cast<_Float16>(rng.next()));
    // Panels: weights [tiles][units][16 columns][16 k] (columns >= outputs are +0), x hi then lo [units][rows][16].
    std::vector<_Float16> wp(std::size_t{tiles} * units * 256U, static_cast<_Float16>(0.0F));
    std::vector<_Float16> xp(std::size_t{2U} * rows * stride);
    for (unsigned o = 0; o < outputs; ++o)
        for (unsigned k = 0; k < stride; ++k)
            wp[((std::size_t{o / 16U} * units + k / 16U) * 16U + o % 16U) * 16U + k % 16U] = static_cast<_Float16>(ws[std::size_t{o} * stride + k]);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned k = 0; k < stride; ++k) {
            const float x = xs[std::size_t{r} * stride + k];
            const _Float16 hi = static_cast<_Float16>(x);
            const _Float16 lo = static_cast<_Float16>((x - static_cast<float>(hi)) * kScale);
            const std::size_t at = (std::size_t{k / 16U} * rows + r) * 16U + k % 16U;
            xp[at] = hi;
            xp[std::size_t{rows} * stride + at] = lo;
        }
    _Float16 *dw = nullptr, *dx = nullptr;
    float* dy = nullptr;
    unsigned* df = nullptr;
    ok(hipMalloc(&dw, wp.size() * sizeof(_Float16)), "malloc w");
    ok(hipMalloc(&dx, xp.size() * sizeof(_Float16)), "malloc x");
    ok(hipMalloc(&dy, std::size_t{rows} * outputs * sizeof(float)), "malloc y");
    ok(hipMalloc(&df, sizeof(unsigned)), "malloc flag");
    ok(hipMemcpy(dw, wp.data(), wp.size() * sizeof(_Float16), hipMemcpyHostToDevice), "copy w");
    ok(hipMemcpy(dx, xp.data(), xp.size() * sizeof(_Float16), hipMemcpyHostToDevice), "copy x");
    ok(hipMemset(dy, 0xFF, std::size_t{rows} * outputs * sizeof(float)), "fill y");  // NaN: an unwritten element fails
    ok(hipMemset(df, 0, sizeof(unsigned)), "flag");
    hilo::launch_panels<Block, kUnscale>(dw, dx, rows, outputs, stride, dy, df, nullptr);
    ok(hipGetLastError(), "launch");
    ok(hipDeviceSynchronize(), "sync");
    std::vector<float> y(std::size_t{rows} * outputs);
    ok(hipMemcpy(y.data(), dy, y.size() * sizeof(float), hipMemcpyDeviceToHost), "copy y");
    ok(hipMemcpy(flag, df, sizeof(unsigned), hipMemcpyDeviceToHost), "copy flag");
    ok(hipFree(dw), "free w"); ok(hipFree(dx), "free x"); ok(hipFree(dy), "free y"); ok(hipFree(df), "free flag");
    unsigned bad = 0;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned o = 0; o < outputs; ++o) {
            double ref = 0.0, mag = 0.0;
            for (unsigned k = 0; k < stride; ++k) {
                const double p = static_cast<double>(xs[std::size_t{r} * stride + k]) * ws[std::size_t{o} * stride + k];
                ref += p;
                mag += std::fabs(p);
            }
            const double got = y[std::size_t{r} * outputs + o];
            if (!(std::fabs(got - ref) <= 1e-5 * mag + 1e-6)) {
                if (bad++ < 3U) std::printf("  MISMATCH r %u o %u got %.8g want %.8g\n", r, o, got, ref);
            }
        }
    return bad;
}

}  // namespace

int main() {
    unsigned failures = 0;
    try {
        // units: partial iteration (3, 11, 17) and whole (8, 16); outputs: one output, partial tile, two/four tiles with a
        // partial last tile; rows: one row, a partial CTA, a full CTA.
        for (const unsigned units : {3U, 8U, 11U, 17U, 16U})
            for (const unsigned outputs : {1U, 5U, 16U, 21U, 64U, 70U})
                for (const unsigned rows : {1U, 7U, 130U}) {
                    unsigned fa = 0, fb = 0;
                    const unsigned ba = run_case<0>(rows, outputs, units, &fa);
                    const unsigned bb = run_case<8>(rows, outputs, units, &fb);
                    const bool pass = ba == 0 && bb == 0 && fa == 0 && fb == 0;
                    std::printf("CASE units %2u outputs %3u rows %3u unblocked bad %u blocked bad %u failure %u/%u %s\n", units, outputs, rows, ba, bb, fa, fb,
                                pass ? "PASS" : "FAIL");
                    failures += pass ? 0U : 1U;
                }
    } catch (const std::exception& e) {
        std::printf("ERROR %s\n", e.what());
        return 2;
    }
    std::printf("f16 hilo panels: %u failures\n", failures);
    return failures ? 1 : 0;
}
