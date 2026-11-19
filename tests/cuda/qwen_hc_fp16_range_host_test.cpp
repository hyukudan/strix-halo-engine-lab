// Host test of the hc_mma FP16-range admission (include/strix/cuda/qwen_hc_fp16_range.hpp): admitted inside the range,
// refused with a reason at the activation bound and on non-finite weights; v2 operand shift: the six production weights that
// round to FP16 zero unshifted are carried (none dropped), the real pack range is admitted, a range above 2^40 is refused.
#include "strix/cuda/qwen_hc_fp16_range.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace hr = strix::cuda::qwen_hc::fp16_range;

// FP32 -> FP16 round to nearest even (the value, as double): __floats2half2_rn's rounding.
static double fp16_round(float xf) {
    const double x = xf, a = std::fabs(x);
    if (!std::isfinite(x)) return x;
    if (a >= 65520.0) return std::copysign(std::numeric_limits<double>::infinity(), x);
    int p = 0;
    (void)std::frexp(a, &p);                                  // a = m 2^p, m in [0.5, 1): ulp = 2^(p - 11) when normal
    const double q = a < 0x1p-14 ? 0x1p-24 : std::ldexp(1.0, p - 11);
    return std::nearbyint(x / q) * q;                         // default rounding mode: nearest, ties to even
}

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

int main() {
    std::vector<float> norm(10240, 1.5F), inject(40960, 0.01F);
    inject[7] = 0.0F;  // exact zeros are not an underflow
    {
        hr::Scan s;
        hr::scan_norm(s, norm.data(), norm.size());
        hr::scan_inject(s, inject.data(), inject.size());
        check(hr::refusal(s).empty(), "typical weights admitted: " + hr::refusal(s));
        check(s.norm_max_abs == 1.5F && s.inject_min_nonzero_abs == 0.01F, "scan extremes");
    }
    {  // sqrt(2560) * 161 = 8146 < 8188 admitted; 162 -> 8197 refused
        hr::Scan a, b;
        norm[3] = -161.0F;
        hr::scan_norm(a, norm.data(), norm.size());
        check(hr::refusal(a).empty(), "161 admitted: " + hr::refusal(a));
        norm[3] = -162.0F;
        hr::scan_norm(b, norm.data(), norm.size());
        check(hr::refusal(b).find("FP16 activation range") != std::string::npos, "162 refused: " + hr::refusal(b));
        norm[3] = 1.5F;
    }
    {  // v2: weights that round to FP16 zero unshifted are carried by the operand shift, not refused and not dropped.
       // The six values are the production pack's (blk.10/11/27/31/44 hc_*_inject; rtx_integ3c hcscan), its max is 1.32.
        const float six[6] = {-1.327135e-08F, 2.048910e-08F, -2.223533e-08F, -2.066372e-09F, 2.514571e-08F, 2.584420e-08F};
        const std::size_t at[6] = {675, 3 * 10240 + 3326, 10240 + 7478, 10240 + 6705, 8302, 3 * 10240 + 9664};
        std::vector<float> w = inject;
        for (int i = 0; i < 6; ++i) w[at[i]] = six[i];
        w[100] = 1.32F;
        w[200] = 0x1p-20F;  // an unshifted FP16 subnormal
        hr::Scan s;
        hr::scan_inject(s, w.data(), w.size());
        const int e = hr::inject_exp(s);
        check(hr::refusal(s).empty(), "pack-like weights admitted: " + hr::refusal(s));
        check(s.inject_underflow == 6 && s.inject_subnormal == 7, "6 unshifted zeros, 7 unshifted subnormals counted");
        check(e == 14 && std::ldexp(1.32, e) < 32768.0, "operand exponent 14 for max 1.32: " + std::to_string(e));
        std::size_t dropped_v1 = 0, dropped_v2 = 0, subnormal_v2 = 0, imprecise_v2 = 0;
        for (const float v : w) {
            if (v == 0.0F) continue;
            dropped_v1 += fp16_round(v) == 0.0 ? 1U : 0U;
            const double shifted = std::ldexp(static_cast<double>(v), e);   // exact in FP32 too (power of two)
            const double q = fp16_round(static_cast<float>(shifted));        // the kernel's FP16 B operand
            dropped_v2 += q == 0.0 ? 1U : 0U;
            if (std::fabs(shifted) < 0x1p-14) ++subnormal_v2;                // only the 2^-28.85 one (-> 2^-14.85)
            else if (std::fabs(std::ldexp(q, -e) - v) > std::ldexp(std::fabs(static_cast<double>(v)), -11)) ++imprecise_v2;
        }
        check(dropped_v1 == 6 && dropped_v2 == 0 && subnormal_v2 == 1 && imprecise_v2 == 0,
              "v1 drops 6, v2 drops 0, 1 subnormal, normals within 2^-11: " + std::to_string(dropped_v1) + " " +
                  std::to_string(dropped_v2) + " " + std::to_string(subnormal_v2) + " " + std::to_string(imprecise_v2));
        // One-hot activation on a dropped weight (x = sqrt(2560) * 2.117, the bound at blk.44 k=8302): v1's logit term is 0,
        // v2's is the weight's term to 2^-11 after the 2^-(e+2) unscale (the kernel's x * inj_unscale).
        const double x = fp16_round(static_cast<float>(std::sqrt(2560.0) * 2.117));
        const double v1 = x * fp16_round(six[4]) * 0.25;
        const double v2 = x * fp16_round(static_cast<float>(std::ldexp(six[4], e))) * std::ldexp(1.0, -(e + 2));
        const double ref = x * static_cast<double>(six[4]) * 0.25;
        check(v1 == 0.0 && std::fabs(v2 - ref) <= std::ldexp(std::fabs(ref), -11), "dropped term restored");
    }
    {  // all-denormal-scale inject tensors: the shift would be 2^e with e > 124 (2^e inf or 2^-(e+2) not normal): refused
        hr::Scan tiny, edge;
        std::vector<float> w = {0x1p-120F, 0x1p-121F};  // max < 2^-119: e = 134, ldexp(1.0F, 134) = inf
        hr::scan_inject(tiny, w.data(), w.size());
        check(hr::inject_exp(tiny) == 134 && !std::isfinite(std::ldexp(1.0F, hr::inject_exp(tiny))) &&
                  hr::refusal(tiny).find("outside the float scale range") != std::string::npos,
              "denormal-scale inject refused: " + hr::refusal(tiny));
        w = {0x1.8p-110F};  // max < 2^-109: e = 124, the largest admitted shift; both scales normal
        hr::scan_inject(edge, w.data(), w.size());
        const int e = hr::inject_exp(edge);
        check(e == 124 && hr::refusal(edge).empty() && std::isnormal(std::ldexp(1.0F, e)) && std::isnormal(std::ldexp(1.0F, -(e + 2))),
              "edge shift 2^124 admitted: " + hr::refusal(edge));
    }
    {  // the real pack's measured range (min 2.0664e-9 = 2^-28.85, max 1.32) is admitted; 2^-41 .. 1 is not representable
        hr::Scan s, r, big;
        std::vector<float> w = {2.0664e-09F, 1.32F, 0.25F};
        hr::scan_inject(s, w.data(), w.size());
        check(hr::refusal(s).empty() && hr::inject_exp(s) == 14 && fp16_round(static_cast<float>(std::ldexp(w[0], 14))) != 0.0,
              "pack range admitted: " + hr::refusal(s));
        w = {0x1p-41F, 1.0F};
        hr::scan_inject(r, w.data(), w.size());
        check(hr::refusal(r).find("exceeds FP16") != std::string::npos, "range 2^41 refused: " + hr::refusal(r));
        w = {70000.0F, 0.01F};  // above 65504: v1 made it inf; the shift is negative (-2) and keeps it finite
        hr::scan_inject(big, w.data(), w.size());
        const int e = hr::inject_exp(big);
        check(hr::refusal(big).empty() && e == -2 && std::isfinite(fp16_round(static_cast<float>(std::ldexp(70000.0, e)))),
              "large weights shifted down: " + std::to_string(e));
    }
    {  // e = 0 is v1 bit for bit: x * 2^-2 == x / 4 for normal, subnormal and signed values (both correctly rounded)
        const float xs[] = {1.0F, -3.0F, 0x1.fffffep+127F, 0x1p-126F, 0x1.8p-148F, 0x1p-149F, 12345.678F, -0x1.2345p-130F};
        bool same = true;
        for (const float v : xs) same = same && (v * std::ldexp(1.0F, -2)) == (v / 4.0F);
        check(same, "e = 0 unscale equals the v1 division");
    }
    {  // non-finite weights refuse the form
        hr::Scan s;
        norm[0] = std::numeric_limits<float>::quiet_NaN();
        inject[0] = std::numeric_limits<float>::infinity();
        hr::scan_norm(s, norm.data(), norm.size());
        hr::scan_inject(s, inject.data(), inject.size());
        check(s.nonfinite == 2 && hr::refusal(s).find("non-finite") != std::string::npos, "non-finite refused: " + hr::refusal(s));
    }
    if (failures == 0) std::printf("qwen_hc_fp16_range_host_test: OK\n");
    return failures == 0 ? 0 : 1;
}
