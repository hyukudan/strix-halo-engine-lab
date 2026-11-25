#pragma once
// Host arithmetic shared by the reference tower and the family projectors (no tensors, no device code): the row
// normalization and the clamped SwiGLU gate. Double accumulation, FP32 rows in and out, one fixed operation order, so
// every caller that uses them rounds the same way.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace strix::vision {

// y[r] = (x[r] - mean) / sqrt(var + eps) * w + b over `rows` rows of n values. center: LayerNorm (the mean is
// subtracted); otherwise RMSNorm (mean 0). b may be null (no bias). y may alias x.
inline void norm_rows(const float* x, std::size_t rows, std::size_t n, const float* w, const float* b, float eps, bool center, float* y) {
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * n;
        double mean = 0.0, sq = 0.0;
        if (center) {
            for (std::size_t i = 0; i < n; ++i) mean += static_cast<double>(xr[i]);
            mean /= static_cast<double>(n);
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(xr[i]) - mean;
            sq += d * d;
        }
        const double inv = 1.0 / std::sqrt(sq / static_cast<double>(n) + static_cast<double>(eps));
        for (std::size_t i = 0; i < n; ++i)
            y[r * n + i] = static_cast<float>((static_cast<double>(xr[i]) - mean) * inv * static_cast<double>(w[i]) +
                                              (b != nullptr ? static_cast<double>(b[i]) : 0.0));
    }
}

// silu(min(gate, limit)) * clamp(up, -limit, limit) in double, rounded to FP32; limit <= 0 means no clamp.
[[nodiscard]] inline float clamped_swiglu(float gate_in, float up_in, double limit) {
    double gate = static_cast<double>(gate_in), up = static_cast<double>(up_in);
    if (limit > 0.0) {
        gate = std::min(gate, limit);
        up = std::clamp(up, -limit, limit);
    }
    return static_cast<float>(gate / (1.0 + std::exp(-gate)) * up);
}

}  // namespace strix::vision
