#pragma once
// Model admission of the FAST HC form hc_mma (qwen-cuda-hc-f16mma-v2, qwen_hc.cu "FAST forms"): the [P] GEMM converts the
// normalized activations to FP16 without saturation (__floats2half2_rn) and rounds the FP32 inject weights to FP16. Both are
// safe only if the model's weights keep the values inside the FP16 range, so the range argument is checked on the weights
// at load instead of being a comment:
//   - activations: a stream's RMS-normalized value is |x / rms| <= sqrt(2560), so |x * w| <= sqrt(2560) * max|norm w|.
//     Admit only below 65504 / 8 (three bits of headroom for the K-sum's partial products and the low path).
//   - inject weights: the FP16 operand is w * 2^e with one per-model exponent e (inject_exp) that puts max|w| * 2^e just below
//     2^15; the kernel scales the FP32 accumulator back by 2^-e (exact: a power of two). Rounding w itself would send every
//     |w| <= 2^-25 to FP16 zero (2^-24 is the smallest subnormal) and every |w| < 2^-14 to a subnormal (absolute error up to
//     2^-25); the production pack has 6 such zeros and 17,255 such subnormals among 3.9 M weights (dynamic range 2^29.3).
//     With the shift no non-zero weight is dropped and every |w| >= 2^(-14-e) keeps FP16's 2^-11 relative precision.
//     The form is refused only if a shift cannot represent the tensor: a non-zero weight still at or below 2^-25 after it
//     (dynamic range above ~2^40). Non-finite weights refuse it too.
// v1 rounded w unshifted (dropped the 6); v2 = the shift. e = 0 gives v1's arithmetic bit for bit (x * 0.25 == x / 4).
// Host-only (no CUDA types): the driver copies the norms / injects to the host once when hc_mma is requested.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>

namespace strix::cuda::qwen_hc::fp16_range {

inline constexpr float kHidden = 2560.0F;
inline constexpr float kFp16Max = 65504.0F;
inline constexpr float kActivationLimit = kFp16Max / 8.0F;  // bound on sqrt(kHidden) * max|norm w|
inline constexpr float kFp16ZeroBelow = 0x1p-25F;          // |w| <= this rounds to FP16 zero (round half to even)
inline constexpr int kInjectTopExp = 15;                   // shifted max|w| < 2^15 (half of FP16's 65504 range)
inline constexpr int kInjectExpMax = 124;                  // 2^e and 2^-(e+2) stay normal floats for |e| <= 124

struct Scan {
    float norm_max_abs = 0.0F;
    float inject_max_abs = 0.0F;
    float inject_min_nonzero_abs = std::numeric_limits<float>::infinity();
    std::size_t nonfinite = 0;
    std::size_t inject_underflow = 0;  // non-zero |w| <= 2^-25 unshifted (what v1 dropped; carried by the shift)
    std::size_t inject_subnormal = 0;  // non-zero |w| < 2^-14 unshifted (v1 rounded them to FP16 subnormals)
};

inline void scan_norm(Scan& s, const float* w, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(w[i])) { ++s.nonfinite; continue; }
        s.norm_max_abs = std::fmax(s.norm_max_abs, std::fabs(w[i]));
    }
}

inline void scan_inject(Scan& s, const float* w, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(w[i])) { ++s.nonfinite; continue; }
        const float a = std::fabs(w[i]);
        if (a == 0.0F) continue;
        s.inject_max_abs = std::fmax(s.inject_max_abs, a);
        s.inject_min_nonzero_abs = std::fmin(s.inject_min_nonzero_abs, a);
        if (a <= kFp16ZeroBelow) ++s.inject_underflow;
        if (a < 0x1p-14F) ++s.inject_subnormal;
    }
}

// The per-model operand exponent: the largest e with inject_max_abs * 2^e < 2^15 (0 when every inject weight is zero).
inline int inject_exp(const Scan& s) {
    if (!(s.inject_max_abs > 0.0F) || !std::isfinite(s.inject_max_abs)) return 0;
    int p = 0;
    (void)std::frexp(s.inject_max_abs, &p);  // max = m 2^p, m in [0.5, 1): max < 2^p
    return kInjectTopExp - p;
}

// Empty = admit hc_mma; otherwise the reason it is refused.
inline std::string refusal(const Scan& s) {
    std::string out;
    const auto add = [&](const std::string& r) { out += (out.empty() ? "" : "; ") + r; };
    if (s.nonfinite) add(std::to_string(s.nonfinite) + " non-finite HC norm/inject weights");
    const float bound = std::sqrt(kHidden) * s.norm_max_abs;
    if (!(bound < kActivationLimit))
        add("sqrt(2560) * max|norm w| = " + std::to_string(bound) + " >= 65504/8 (FP16 activation range)");
    const int e = inject_exp(s);
    // The kernel receives 2^e and 2^-(e+2) as floats (qwen_hc.cu launch): both must be normal floats, or the operand turns
    // into inf (NaN logits) or the unscale into 0 (the inject term vanishes). Only an all-denormal-scale tensor reaches this.
    if (e > kInjectExpMax || e < -kInjectExpMax) {
        add("inject operand shift 2^" + std::to_string(e) + " is outside the float scale range (|e| <= " +
            std::to_string(kInjectExpMax) + ")");
        return out;
    }
    if (std::isfinite(s.inject_min_nonzero_abs) && std::ldexp(s.inject_min_nonzero_abs, e) <= kFp16ZeroBelow) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "inject |w| range %.4g .. %.4g exceeds FP16 with the 2^%d shift (min rounds to zero)",
                      static_cast<double>(s.inject_min_nonzero_abs), static_cast<double>(s.inject_max_abs), e);
        add(buf);
    }
    return out;
}

}  // namespace strix::cuda::qwen_hc::fp16_range
