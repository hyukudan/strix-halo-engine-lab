#pragma once
// Model admission of the FAST HC form hc_mma (qwen-cuda-hc-f16mma-v1, qwen_hc.cu "FAST forms"): the [P] GEMM converts the
// normalized activations to FP16 without saturation (__floats2half2_rn) and rounds the FP32 inject weights to FP16. Both are
// safe only if the model's weights keep the values inside the FP16 range, so the range argument is checked on the weights
// at load instead of being a comment:
//   - activations: a stream's RMS-normalized value is |x / rms| <= sqrt(2560), so |x * w| <= sqrt(2560) * max|norm w|.
//     Admit only below 65504 / 8 (three bits of headroom for the K-sum's partial products and the low path).
//   - inject weights: a non-zero FP32 weight with |w| <= 2^-25 rounds to FP16 zero (2^-24 is the smallest subnormal), which
//     would silently drop that term; any such weight refuses the form. Non-finite weights refuse it too.
// Host-only (no CUDA types): the driver copies the norms / injects to the host once when hc_mma is requested.
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace strix::cuda::qwen_hc::fp16_range {

inline constexpr float kHidden = 2560.0F;
inline constexpr float kFp16Max = 65504.0F;
inline constexpr float kActivationLimit = kFp16Max / 8.0F;  // bound on sqrt(kHidden) * max|norm w|
inline constexpr float kFp16ZeroBelow = 0x1p-25F;          // |w| <= this rounds to FP16 zero (round half to even)

struct Scan {
    float norm_max_abs = 0.0F;
    float inject_min_nonzero_abs = std::numeric_limits<float>::infinity();
    std::size_t nonfinite = 0, inject_underflow = 0;
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
        s.inject_min_nonzero_abs = std::fmin(s.inject_min_nonzero_abs, a);
        if (a <= kFp16ZeroBelow) ++s.inject_underflow;
    }
}

// Empty = admit hc_mma; otherwise the reason it is refused.
inline std::string refusal(const Scan& s) {
    std::string out;
    const auto add = [&](const std::string& r) { out += (out.empty() ? "" : "; ") + r; };
    if (s.nonfinite) add(std::to_string(s.nonfinite) + " non-finite HC norm/inject weights");
    const float bound = std::sqrt(kHidden) * s.norm_max_abs;
    if (!(bound < kActivationLimit))
        add("sqrt(2560) * max|norm w| = " + std::to_string(bound) + " >= 65504/8 (FP16 activation range)");
    if (s.inject_underflow)
        add(std::to_string(s.inject_underflow) + " non-zero inject weights round to FP16 zero (|w| <= 2^-25)");
    return out;
}

}  // namespace strix::cuda::qwen_hc::fp16_range
