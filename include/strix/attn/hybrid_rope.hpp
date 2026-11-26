#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <stdexcept>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_HYBRID_ROPE_HD __host__ __device__ inline
#else
#define STRIX_HYBRID_ROPE_HD inline
#endif

namespace strix::attn {

struct HybridYarn {
    double original_context, factor, beta_fast, beta_slow;
};

// Callers bind the domain, base and pair count. Evaluate in double and round
// once, retaining the hybrid attention reference's operation order.
inline void hybrid_rope_row(std::uint64_t position, double base, const HybridYarn* yarn,
                            std::span<float> c, std::span<float> s) {
    if (c.size() != s.size()) throw std::invalid_argument("hybrid RoPE: table sizes differ");
    const double dim = 2.0 * static_cast<double>(c.size());
    double low = 0, high = 0;
    if (yarn) {
        const auto corrected = [&](double rotations) {
            return dim * std::log(yarn->original_context / (rotations * 2.0 * std::numbers::pi)) /
                   (2.0 * std::log(base));
        };
        low = std::max(std::floor(corrected(yarn->beta_fast)), 0.0);
        high = std::min(std::ceil(corrected(yarn->beta_slow)), dim - 1.0);
    }
    for (std::size_t i = 0; i < c.size(); ++i) {
        const double pair = static_cast<double>(i);
        double f = 1.0 / std::pow(base, (2.0 * pair) / dim);
        if (yarn) {
            const double ramp = std::clamp((pair - low) / std::max(high - low, 1e-3), 0.0, 1.0);
            const double smooth = 1.0 - ramp;
            f = f / yarn->factor * (1.0 - smooth) + f * smooth;
        }
        const double angle = static_cast<double>(position) * f;
        c[i] = static_cast<float>(std::cos(angle));
        s[i] = static_cast<float>(std::sin(angle));
    }
}

STRIX_HYBRID_ROPE_HD void rope_pair(float& x0, float& x1, float c, float s, bool inverse) {
    const float a = x0, b = x1;
    if (!inverse) { x0 = a * c - b * s; x1 = a * s + b * c; } else { x0 = a * c + b * s; x1 = b * c - a * s; }
}

inline void hybrid_rope_tail(std::span<float> heads, unsigned width,
                             std::span<const float> c, std::span<const float> s, bool inverse) {
    if (width == 0 || heads.size() % width != 0 || c.size() != s.size() || c.size() > width / 2U)
        throw std::invalid_argument("hybrid RoPE: invalid head shape");
    for (std::size_t head = 0; head < heads.size(); head += width) {
        auto tail = heads.subspan(head + width - 2U * c.size(), 2U * c.size());
        for (std::size_t pair = 0; pair < c.size(); ++pair)
            rope_pair(tail[2U * pair], tail[2U * pair + 1U], c[pair], s[pair], inverse);
    }
}

} // namespace strix::attn

#undef STRIX_HYBRID_ROPE_HD
