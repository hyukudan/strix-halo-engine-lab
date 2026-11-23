// Hi/lo split of the shared FP16 tall GEMM (f16_dense_tall ColsHiLo, f16_hilo_math.hpp): host checks.
// 1. merge() rounds twice (scaled product, then the add), never as one FMA.
// 2. The core's schedule (8-unit iterations, merge_after at the end of an iteration, final merge) gives the same bits
//    as the per-K16 engine form it replaces (chains restart when (k0 + 16) % block == 0 or k0 + 16 >= stride), for
//    blocked and unblocked products, signed zeros, subnormal low parts and large operands. The WMMA step is modelled
//    by a fixed FP32 order; both schedules feed it identical operands, so any order the hardware uses gives the same
//    equality.
#include "core/f16_hilo_math.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
namespace hl = strix::core::f16_hilo;

int failures = 0;
void check(bool ok, const char* what, unsigned a = 0, unsigned b = 0) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s (%08x vs %08x)\n", what, a, b);
    }
}

// FP32 -> FP16 (RNE, overflow to inf, NaN kept) -> FP32.
float fp16(float x) {
    const std::uint32_t u = std::bit_cast<std::uint32_t>(x), sign = u & 0x80000000U, a = u & 0x7fffffffU;
    if (a >= 0x7f800000U) return x;
    const auto make = [&](float v) { return sign ? -v : v; };
    const double m = std::bit_cast<float>(a);
    if (m >= 65520.0) return make(std::numeric_limits<float>::infinity());
    const int e = m == 0.0 ? -24 : std::max(std::ilogb(m), -14);
    const double q = std::ldexp(1.0, e - 10);  // FP16 quantum at this binade (subnormal quantum 2^-24)
    const double r = std::nearbyint(m / q) * q;  // round-to-nearest-even (default mode)
    return make(static_cast<float>(r));
}

struct Rng {
    std::uint64_t s;
    std::uint32_t next() {
        s ^= s << 13U; s ^= s >> 7U; s ^= s << 17U;
        return static_cast<std::uint32_t>(s >> 11U);
    }
    float uniform() { return static_cast<float>(next() % 2000001U) / 1000000.0F - 1.0F; }
};

// WMMA model: c + products of one 16-wide k unit in a fixed order (FP16 products are exact in FP32).
float unit_step(float c, const float* a, const float* b) {
    float s = c;
    for (unsigned k = 0; k < 16U; ++k) s = s + a[k] * b[k];
    return s;
}

constexpr float kScale = 4096.0F, kUnscale = 1.0F / 4096.0F;

// The engine form being replaced, transcribed per output (k0 over stride in steps of 16, block in inputs).
float engine_form(const std::vector<float>& hi, const std::vector<float>& lo, const std::vector<float>& w,
                  unsigned stride, bool blocked, unsigned block) {
    float h = 0.0F, l = 0.0F, s = 0.0F;
    for (unsigned k0 = 0; k0 < stride; k0 += 16U) {
        h = unit_step(h, &hi[k0], &w[k0]);
        l = unit_step(l, &lo[k0], &w[k0]);
        if (blocked && ((k0 + 16U) % block == 0U || k0 + 16U >= stride)) {
            volatile float scaled = l * kUnscale;
            const float part = h + scaled;
            s = s + part;
            h = 0.0F;
            l = 0.0F;
        }
    }
    if (blocked) return s;
    volatile float scaled = l * kUnscale;
    return h + scaled;
}

// The shared core's schedule: iterations of 8 units, block check after each, final merge (f16_dense_tall::core).
float core_form(const std::vector<float>& hi, const std::vector<float>& lo, const std::vector<float>& w, unsigned units,
                unsigned block_units) {
    float acc = 0.0F, low = 0.0F, sum = 0.0F;
    for (unsigned base = 0; base < units; base += 8U) {
        for (unsigned u = base; u < base + 8U; ++u) {
            acc = unit_step(acc, &hi[16U * u], &w[16U * u]);
            low = unit_step(low, &lo[16U * u], &w[16U * u]);
        }
        if (block_units != 0U && hl::merge_after(base + 7U, units, block_units)) {
            sum = sum + hl::merge(acc, low, kUnscale);
            acc = 0.0F;
            low = 0.0F;
        }
    }
    return block_units != 0U ? sum : hl::merge(acc, low, kUnscale);
}

void merge_rounds_twice() {
    Rng r{0x9e3779b97f4a7c15ULL};
    for (unsigned i = 0; i < 200000U; ++i) {
        // Low partial sums near the subnormal range make l * unscale inexact, where an FMA would differ.
        const float h = std::ldexp(r.uniform(), -static_cast<int>(r.next() % 150U));
        const float l = std::ldexp(r.uniform(), -static_cast<int>(r.next() % 150U));
        const float scaled = static_cast<float>(static_cast<double>(l) * static_cast<double>(kUnscale));
        const float want = static_cast<float>(static_cast<double>(h) + static_cast<double>(scaled));
        const float got = hl::merge(h, l, kUnscale);
        check(std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(want), "merge two roundings",
              std::bit_cast<std::uint32_t>(got), std::bit_cast<std::uint32_t>(want));
    }
}

void merge_points() {
    check(!hl::merge_after(6, 16, 8) && hl::merge_after(7, 16, 8) && hl::merge_after(15, 16, 8), "block 8");
    check(!hl::merge_after(7, 32, 16) && hl::merge_after(15, 32, 16) && hl::merge_after(31, 32, 16), "block 16");
    check(!hl::merge_after(7, 16, 0) && hl::merge_after(15, 16, 0), "unblocked");
    check(hl::merge_after(23, 24, 16), "last partial block");
}

void schedules_agree() {
    Rng r{12345U};
    const unsigned unit_counts[] = {8U, 16U, 24U, 64U, 168U};
    for (unsigned units : unit_counts) {
        const unsigned stride = 16U * units;
        for (unsigned trial = 0; trial < 64U; ++trial) {
            std::vector<float> x(stride), hi(stride), lo(stride), w(stride);
            const unsigned kind = trial % 4U;
            for (unsigned k = 0; k < stride; ++k) {
                float v = r.uniform();
                if (kind == 1U) v = std::ldexp(v, 12);                           // large: hi near the FP16 top
                if (kind == 2U) v = std::ldexp(v, -20);                          // tiny: subnormal operands
                if (kind == 3U) v = (k % 3U) != 0U ? -0.0F : v;                        // signed zeros
                x[k] = v;
                w[k] = fp16((kind == 3U && (k % 5U) != 0U) ? -0.0F : r.uniform());
                hi[k] = fp16(v);
                lo[k] = fp16(hl::low_operand(v, hi[k], kScale));
            }
            for (unsigned block_units : {0U, 8U, 16U}) {
                const float want = engine_form(hi, lo, w, stride, block_units != 0U, 16U * (block_units ? block_units : 1U));
                const float got = core_form(hi, lo, w, units, block_units);
                check(std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(want), "schedule bits",
                      std::bit_cast<std::uint32_t>(got), std::bit_cast<std::uint32_t>(want));
            }
        }
    }
    // All-negative-zero operands: blocked sums start at +0 (+0 + -0 = +0), unblocked keeps the chains' sign.
    std::vector<float> z(128U, -0.0F), w(128U, 1.0F);
    check(std::bit_cast<std::uint32_t>(core_form(z, z, w, 8U, 8U)) == std::bit_cast<std::uint32_t>(engine_form(z, z, w, 128U, true, 128U)),
          "blocked zero sign");
}

// Stored units not a multiple of 8: the core's final iteration reads +0 operands past the stored units (zero padding),
// and a blocked product still closes its last block at the last stored unit; equal to the per-K16 engine form.
void partial_final_iteration() {
    Rng r{4242U};
    for (unsigned units : {1U, 3U, 7U, 9U, 11U, 17U, 23U}) {
        const unsigned stride = 16U * units, padded = (units + 7U) / 8U * 8U;
        for (unsigned trial = 0; trial < 32U; ++trial) {
            std::vector<float> hi(16U * padded, 0.0F), lo(16U * padded, 0.0F), w(16U * padded, 0.0F);
            std::vector<float> hs(stride), ls(stride), ws(stride);
            for (unsigned k = 0; k < stride; ++k) {
                const float v = (trial % 2U) ? std::ldexp(r.uniform(), -18) : r.uniform();
                hi[k] = hs[k] = fp16(v);
                lo[k] = ls[k] = fp16(hl::low_operand(v, hi[k], kScale));
                w[k] = ws[k] = fp16(r.uniform());
            }
            for (unsigned block_units : {0U, 8U, 16U}) {
                const float want = engine_form(hs, ls, ws, stride, block_units != 0U, 16U * (block_units ? block_units : 1U));
                const float got = core_form(hi, lo, w, units, block_units);
                check(std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(want), "partial final iteration",
                      std::bit_cast<std::uint32_t>(got), std::bit_cast<std::uint32_t>(want));
            }
        }
    }
}

void split_reconstructs() {
    Rng r{777U};
    for (unsigned i = 0; i < 100000U; ++i) {
        const float x = std::ldexp(r.uniform(), static_cast<int>(r.next() % 24U) - 12);
        const float hi = fp16(x), lo = fp16(hl::low_operand(x, hi, kScale));
        const double err = std::fabs(static_cast<double>(x) - (static_cast<double>(hi) + static_cast<double>(lo) / kScale));
        check(err <= std::ldexp(std::fabs(static_cast<double>(x)), -20) + std::ldexp(1.0, -36), "hi + lo / scale");
    }
}
}  // namespace

int main() {
    merge_rounds_twice();
    merge_points();
    schedules_agree();
    partial_final_iteration();
    split_reconstructs();
    if (failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("f16_hilo host tests: PASS");
    return EXIT_SUCCESS;
}
