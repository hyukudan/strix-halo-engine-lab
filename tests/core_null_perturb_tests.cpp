// strix::core::fp16_ulp_up_bits (src/core/null_perturb.hpp): a gate null must survive the consumer's FP16 input rounding.
// Invariant: for every finite x in the FP16 range, round-to-FP16(fp16_ulp_up(x)) is exactly the next FP16 value above
// round-to-FP16(x) or differs from it (never equal), while x + one f32 ulp rounds to the same FP16 value (the absorbed
// null that made the GLM prefill null bitwise equal to its reference).
#include "core/null_perturb.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

namespace {

int failures = 0;

void check(bool ok, const char* what, float x) {
    if (!ok) {
        ++failures;
        if (failures < 20) std::printf("FAIL %s x=%.9g\n", what, static_cast<double>(x));
    }
}

std::uint32_t bits_of(float x) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

float float_of(std::uint32_t u) {
    float x = 0.0f;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

// Round to the nearest FP16 value (ties to even), returned as a double; the reference the staging kernels implement.
double round_fp16(float xf) {
    const double x = static_cast<double>(xf);
    if (x == 0.0) return 0.0;
    const double a = std::fabs(x);
    int e = 0;
    (void)std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)
    const int exp_unbiased = e - 1;
    const double spacing = std::ldexp(1.0, (exp_unbiased < -14 ? -14 : exp_unbiased) - 10);
    const double q = a / spacing;
    double r = std::floor(q);
    const double frac = q - r;
    if (frac > 0.5 || (frac == 0.5 && std::fmod(r, 2.0) != 0.0)) r += 1.0;
    return std::copysign(r * spacing, x);
}

void sweep(float x) {
    if (!std::isfinite(x) || std::fabs(x) > 60000.0f) return;
    check(static_cast<double>(strix::core::round_fp16_value(x)) == round_fp16(x), "round_fp16_value == reference", x);
    const float up = float_of(strix::core::fp16_ulp_up_bits(bits_of(x)));
    check(std::isfinite(up) && up > x, "step moves x up", x);
    check(round_fp16(up) != round_fp16(x), "one FP16 ulp survives FP16 rounding", x);
    const double gap = round_fp16(up) - round_fp16(x);
    const double a = std::fabs(static_cast<double>(x));
    int e = 0;
    (void)std::frexp(a, &e);
    const double ulp16 = std::ldexp(1.0, ((e - 1) < -14 ? -14 : (e - 1)) - 10);
    check(gap > 0.0 && gap <= 2.0 * ulp16, "rounded change is one or two FP16 spacings", x);
}

}  // namespace

int main() {
    // Exponent sweep, both signs, several mantissas each: FP16 normal range, the subnormal range, zero.
    const std::uint32_t mantissas[] = {0x000000U, 0x000001U, 0x0007ffU, 0x001000U, 0x2aaaaaU, 0x400000U, 0x7fe000U, 0x7fffffU};
    for (int e = -30; e <= 15; ++e)
        for (const std::uint32_t m : mantissas)
            for (const std::uint32_t s : {0U, 0x80000000U}) sweep(float_of(s | (static_cast<std::uint32_t>(e + 127) << 23) | m));
    sweep(0.0f);
    sweep(-0.0f);
    // Pseudo-random activations in [-64, 64] (residual-stream scale).
    std::uint32_t state = 12345U;
    for (int i = 0; i < 200000; ++i) {
        state = state * 1664525U + 1013904223U;
        const float x = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 128.0f;
        sweep(x);
    }
    // The defect this replaces: one f32 ulp is absorbed by FP16 rounding for typical activations.
    int absorbed = 0;
    for (int i = 0; i < 1000; ++i) {
        state = state * 1664525U + 1013904223U;
        const float x = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 8.0f;
        absorbed += round_fp16(float_of(bits_of(x) + 1U)) == round_fp16(x) ? 1 : 0;
    }
    check(absorbed > 990, "an f32 one-ulp null is absorbed by FP16 rounding (> 99 %)", 0.0f);
    // Non-finite values pass through.
    check(strix::core::fp16_ulp_up_bits(0x7f800000U) == 0x7f800000U, "inf unchanged", 0.0f);
    check(strix::core::fp16_ulp_up_bits(0x7fc00000U) == 0x7fc00000U, "nan unchanged", 0.0f);
    // Gate null site parser (STRIX_DENSE_NULL): fields, bounds, rejection.
    {
        const auto ns = strix::core::parse_null_site("0:17:5119", 64U, 5120U, "N");
        check(ns.layer == 0U && ns.pos == 17U && ns.elem == 5119U, "null site 0:17:5119", 0.0f);
        const auto big = strix::core::parse_null_site("63:4294967295:0", 64U, 5120U, "N");
        check(big.layer == 63U && big.pos == 4294967295ULL && big.elem == 0U, "null site bounds", 0.0f);
        int rejected = 0;
        for (const char* t : {"", "0", "0:1", "0:1:2:3", "64:0:0", "0:0:5120", "a:0:0", "0:-1:0", "0::0", ":0:0", "0:4294967296:0",
                              "0:0:1 ", "00000000000:0:0"}) {
            try {
                (void)strix::core::parse_null_site(t, 64U, 5120U, "N");
            } catch (const std::invalid_argument&) {
                ++rejected;
            }
        }
        check(rejected == 13, "null site rejects malformed or out-of-range text (13 cases)", static_cast<float>(rejected));
    }
    std::printf("null_perturb tests: %s (absorbed f32 one-ulp nulls %d / 1000)\n", failures == 0 ? "PASS" : "FAIL", absorbed);
    return failures == 0 ? 0 : 1;
}
