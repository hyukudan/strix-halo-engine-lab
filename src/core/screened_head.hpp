#pragma once
// EXACT screened greedy LM head: backend-neutral bound math and the host plane builder (no HIP/CUDA code).
//
// The head computes logits y_i = sum_k w_ik x_k (FP32, the incumbent dense_linear order), G(y_i) = softcap(bf16(y_i)) and
// takes the first-max over i. A narrow "plane" of the same rows (per 32-element group one BF16 scale and 4- or 8-bit
// symmetric integer codes, w_hat = s * q exactly representable in FP32) gives y_hat_i cheaply together with a rigorous
// per-row error bound e_i, so that  L_i = y_hat_i - e_i <= y_i <= U_i = y_hat_i + e_i  holds for the incumbent's own FP32
// result (any summation order, FMA or not). With Lstar = max_i L_i, every row with G(U_i) < G(Lstar) is provably not the
// argmax and not tied with it (G is monotone non-decreasing over BF16 inputs, which the device test proves
// exhaustively). The rows that remain (candidates) are recomputed with the incumbent kernel and picked with the
// incumbent rule, so the token and the status are bit-identical to the full head by construction.
//
// Bound (n = width, u = 2^-24, gamma(m) = m u / (1 - m u)):
//   y* = sum w_k x_k and y^* = sum w_hat_k x_k exactly (reals). |y* - y^*| <= R_i ||x|| (Cauchy-Schwarz, R_i = ||w_i - w_hat_i||).
//   incumbent y   : |y - y*|   <= gamma(n)  sum |w_k x_k|     <= gamma(n) ||w_i||     ||x||   (+ underflow term)
//   pass-1 y_hat  : |y_hat - y^*| <= gamma(n) sum |w_hat_k x_k| <= gamma(n) ||w_hat_i|| ||x||   (+ underflow term)
// so |y - y_hat| <= ||x|| (R_i + gamma(n) A_i) + slack with A_i = ||w_i|| + ||w_hat_i||. We use gamma(2n) (every
// summation tree and FMA/non-FMA form of either side has at most 2n roundings on a path), ||x|| from an inflated FP32
// norm, and a final factor (1 + 2^-20) that covers the 7 roundings of evaluating the bound in FP32. The slack is the
// absolute error of products/sums that underflow (flush-to-zero worst case 2^-126 per rounding).
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::core::screened_head {

inline constexpr unsigned kGroup = 32U;           // elements per scale
inline constexpr unsigned kUnit = 8U;             // elements per device load unit (4 units per group)
inline constexpr float kOverflowGuard = 1.0e37F;  // nx * max_i A_i must stay below this, else the full head runs

constexpr bool layout_supported(unsigned width, unsigned bits) noexcept {
    return (bits == 4U || bits == 8U) && width != 0U && width % kGroup == 0U && width < (1U << 22U);
}

struct PlaneLayout {
    unsigned width = 0, bits = 0;
    std::size_t scales_offset = 0;  // bytes of codes before the BF16 scales
    std::size_t row_bytes = 0;      // 16-byte multiple, never a multiple of 2048 (gfx1151 DRAM channel rule)
};

constexpr PlaneLayout plane_layout(unsigned width, unsigned bits) noexcept {
    PlaneLayout layout{width, bits, std::size_t{width} * bits / 8U, 0};
    const std::size_t raw = layout.scales_offset + std::size_t{width / kGroup} * 2U;
    layout.row_bytes = (raw + 15U) / 16U * 16U;
    if (layout.row_bytes % 2048U == 0U) layout.row_bytes += 16U;
    return layout;
}

// Rounded-up conversions (host, double -> FP32).
inline float up_float(double v) {
    auto f = static_cast<float>(v);
    if (static_cast<double>(f) < v) f = std::nextafter(f, std::numeric_limits<float>::infinity());
    return f;
}

struct Plane {
    PlaneLayout layout;
    unsigned rows = 0;
    bool usable = false;  // false: a non-finite weight or an unsupported shape; the caller keeps the full head
    std::vector<std::uint8_t> data;   // rows * layout.row_bytes
    std::vector<float> residual;      // R_i, rounded up
    std::vector<float> amplitude;     // A_i = ||w_i|| + ||w_hat_i||, rounded up
    float max_amplitude = 0.0F;
    [[nodiscard]] std::size_t bytes() const noexcept {
        return data.size() + (residual.size() + amplitude.size()) * sizeof(float);
    }
};

// bf16: row-major [>= live_rows][width] BF16 bit patterns. Only the first live_rows rows enter the plane.
inline Plane build_plane(std::span<const std::uint16_t> bf16, unsigned width, unsigned live_rows, unsigned bits) {
    Plane plane;
    plane.layout = plane_layout(width, bits);
    plane.rows = live_rows;
    if (!layout_supported(width, bits) || live_rows == 0U || bf16.size() / width < live_rows) return plane;
    const int qmax = bits == 4U ? 7 : 127, qmin = bits == 4U ? -8 : -128;
    const auto& L = plane.layout;
    plane.data.assign(std::size_t{live_rows} * L.row_bytes, std::uint8_t{0});
    plane.residual.assign(live_rows, 0.0F);
    plane.amplitude.assign(live_rows, 0.0F);
    std::vector<float> w(width);
    for (unsigned r = 0; r < live_rows; ++r) {
        auto* row = plane.data.data() + std::size_t{r} * L.row_bytes;
        double w2 = 0.0, wh2 = 0.0, d2 = 0.0;
        for (unsigned k = 0; k < width; ++k) {
            w[k] = weight::scalar_float::u2f(std::uint32_t{bf16[std::size_t{r} * width + k]} << 16U);
            if (!std::isfinite(w[k])) { plane = Plane{}; plane.layout = plane_layout(width, bits); plane.rows = live_rows; return plane; }
        }
        for (unsigned g = 0; g < width / kGroup; ++g) {
            double mx = 0.0;
            for (unsigned k = 0; k < kGroup; ++k) mx = std::max(mx, std::fabs(static_cast<double>(w[g * kGroup + k])));
            const float s = weight::scalar_float::bf16_round(static_cast<float>(mx / qmax));
            const std::uint16_t s_bits = static_cast<std::uint16_t>(weight::scalar_float::f2u(s) >> 16U);
            row[L.scales_offset + std::size_t{g} * 2U] = static_cast<std::uint8_t>(s_bits & 0xffU);
            row[L.scales_offset + std::size_t{g} * 2U + 1U] = static_cast<std::uint8_t>(s_bits >> 8U);
            for (unsigned k = 0; k < kGroup; ++k) {
                const unsigned e = g * kGroup + k;
                int q = 0;
                if (s > 0.0F && std::isfinite(s)) q = std::clamp(static_cast<int>(std::lround(static_cast<double>(w[e]) / s)), qmin, qmax);
                const double wh = static_cast<double>(s) * q;
                w2 += static_cast<double>(w[e]) * w[e];
                wh2 += wh * wh;
                d2 += (static_cast<double>(w[e]) - wh) * (static_cast<double>(w[e]) - wh);
                if (bits == 4U) {
                    row[e / 2U] = static_cast<std::uint8_t>(row[e / 2U] | (static_cast<unsigned>(q + 8) << ((e % 2U) * 4U)));
                } else {
                    row[e] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
                }
            }
        }
        // Double sums of <= 2^22 squares carry a relative error < 1e-9; inflate by 1e-6 before rounding up.
        plane.residual[r] = up_float(std::sqrt(d2) * (1.0 + 1.0e-6));
        plane.amplitude[r] = up_float((std::sqrt(w2) + std::sqrt(wh2)) * (1.0 + 1.0e-6));
        plane.max_amplitude = std::max(plane.max_amplitude, plane.amplitude[r]);
    }
    plane.usable = std::isfinite(plane.max_amplitude);
    return plane;
}

// Decoded plane (host reference of the device decode): scale of a group, integer code of an element, w_hat = s * q.
inline float plane_scale(const Plane& plane, unsigned row, unsigned group) {
    const auto& L = plane.layout;
    const auto* r = plane.data.data() + std::size_t{row} * L.row_bytes + L.scales_offset + std::size_t{group} * 2U;
    return weight::scalar_float::u2f((std::uint32_t{r[0]} | (std::uint32_t{r[1]} << 8U)) << 16U);
}
inline int plane_code(const Plane& plane, unsigned row, unsigned k) {
    const auto* r = plane.data.data() + std::size_t{row} * plane.layout.row_bytes;
    return plane.layout.bits == 4U ? static_cast<int>((r[k / 2U] >> ((k % 2U) * 4U)) & 15U) - 8
                                   : static_cast<int>(static_cast<std::int8_t>(r[k]));
}
inline float plane_element(const Plane& plane, unsigned row, unsigned k) {
    return plane_scale(plane, row, k / kGroup) * static_cast<float>(plane_code(plane, row, k));
}

// Scalars the device kernels take (all rounded up).
struct Bound {
    float gamma;       // gamma(2n)
    float sum_inflate; // 1 / (1 - gamma(2n)): the FP32 sum of squares may be low by this factor
    float sum_abs;     // absolute underflow slack of the sum of squares
    float margin;      // 1 + 2^-20
    float abs_slack;   // absolute underflow slack of a logit
    unsigned width;
};

inline Bound make_bound(unsigned width) {
    if (!layout_supported(width, 4U)) throw std::invalid_argument("screened head: unsupported width");
    constexpr double u = 1.0 / 16777216.0, tiny = 1.1754943508222875e-38;  // 2^-126
    const double m = 2.0 * width * u, g = m / (1.0 - m);
    return {up_float(g), up_float(1.0 / (1.0 - g)), up_float(4.0 * width * tiny), 1.0F + 0x1.0p-20F,
            up_float(16.0 * width * tiny), width};
}

// Host mirror of the device norm: FP32 sum of squares in the given order (any order is covered by the bound).
inline float inflated_norm(std::span<const float> x, const Bound& b) {
    float s = 0.0F;
    for (const float v : x) s = std::fma(v, v, s);
    return std::sqrt((s + b.sum_abs) * b.sum_inflate);
}

// True when the screened step may be used: finite inflated norm and no overflow in any intermediate.
inline bool guard_ok(float nx, float max_amplitude) { return nx * max_amplitude < kOverflowGuard; }

// e_i, evaluated in FP32 exactly as the device does (7 roundings, covered by `margin`).
inline float error_bound(float nx, float residual, float amplitude, const Bound& b) {
    return (nx * (residual + b.gamma * amplitude)) * b.margin + b.abs_slack;
}
// Outward-rounded interval (host: one ulp outward after the RN add; device: __fadd_rd / __fadd_ru).
inline void interval(float yhat, float e, float& lo, float& hi) {
    hi = std::nextafter(yhat + e, std::numeric_limits<float>::infinity());
    lo = std::nextafter(yhat - e, -std::numeric_limits<float>::infinity());
}

// The incumbent's per-logit transform: bf16 rounding, then softcap (Bf16Softcap in core/row_transforms.hpp).
inline float softcap_g(float v, float cap) {
    const float r = weight::scalar_float::bf16_round(v);
    return std::isfinite(r) ? cap * std::tanh(r / cap) : r;
}

// The shared greedy rule (hip_argmax_core.hpp): larger value wins, equal values (both zeros equal) lowest token; any
// non-finite logit sets status bit0, no finite logit sets bit1 (token 0).
struct Pick { std::uint32_t token = 0, status = 0; };
inline Pick pick_reference(std::span<const float> logits, std::span<const std::uint32_t> ids = {}) {
    Pick p;
    bool valid = false;
    float best = 0.0F;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float v = logits[i];
        const auto token = ids.empty() ? static_cast<std::uint32_t>(i) : ids[i];
        if (!std::isfinite(v)) { p.status |= 1U; continue; }
        if (!valid || v > best || (v == best && token < p.token)) { best = v; p.token = token; valid = true; }
    }
    if (!valid) p.status |= 2U;
    return p;
}

}  // namespace strix::core::screened_head
