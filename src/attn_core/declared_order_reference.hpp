#pragma once

// Declared arithmetic order of the shared dense decode attention and of the BF16 row norms, as backend-neutral host
// reference code. A device kernel that binds the declared exponential (attn_core/dense_decode.hpp DeclaredExpMath,
// ops/norm lane-strided rows) computes these exact bits, so a host reference built from this header is bitwise equal to it.
// Compile with contraction disabled (-ffp-contract=off), as for the engine's device path.
//
// Attention (dense_decode attend_main_kernel with the sink): the Dim columns of a row are split over 32 lanes, lane t owns
// the contiguous columns [t Dim / 32, (t + 1) Dim / 32); a score is the lane's ascending fused chain from +0, then the
// xor butterfly d = 16 .. 1 (s = s + s'), then one multiplication by the scale. Key position p belongs to wave p % 8; a wave
// walks its positions in ascending order with the online update (m, l, acc: corr = exp(m - m'), w = exp(s - m'),
// l = l corr + w, acc = acc corr + w v, every product rounded before its sum), starting from m = -inf, l = 0, acc = 0.
// The 8 waves merge: M = max(m_w), then max with the sink; L = exp(sink - M) first, then for w = 0 .. 7: f = (m_w == -inf ?
// 0 : exp(m_w - M)), L = L + l_w f, O = O + acc_w f; the head is O / L. The sink is a zero value that only joins the
// denominator and the maximum.
//
// Norm (ops/norm lane_strided_fma_lds_halving, 256 lanes): ops::norm::lane_strided_square_sum; r = 1 / sqrt(ss / n + eps).

#include "core/bf16_reference.hpp"
#include "cpu/core/chunk16_reduce.hpp"
#include "strix/numeric/declared.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::attn_core::declared_order {

inline constexpr unsigned kLanes = 32U;
inline constexpr unsigned kWaves = 8U;  // dense_decode::waves

// Dot of the shared row sweep (core/dense_linear.hpp body, gemv_rows_direct.inc): 16-element chunks on 32 lanes, each
// lane adds its unfused products in input order from +0, then the 16 .. 1 butterfly (cpu::chunk16_kernel order).
inline float row_sweep_dot(const float* weights, const float* input, std::uint32_t width) {
    return cpu::detail::chunk16_reduce(weights, input, width, [](const float* w, const float* x, float* products) {
        for (unsigned i = 0; i < 16U; ++i) products[i] = w[i] * x[i];
    });
}

// Dot of two rows of `dim` columns (dim a multiple of 32) given as two segments each: columns [0, first_a.size()) from
// a, the rest from b. Columns past the segments are zero: a fused chain over +0 x +0 leaves the sum unchanged
// (the chain starts at +0 and never reaches -0), so they are skipped.
inline float lane_tree_dot(std::span<const float> q_a, std::span<const float> k_a, std::span<const float> q_b,
                           std::span<const float> k_b, unsigned dim) {
    if (dim == 0U || dim % kLanes != 0U || q_a.size() != k_a.size() || q_b.size() != k_b.size() ||
        q_a.size() + q_b.size() > dim)
        throw std::invalid_argument("declared attention: invalid dot geometry");
    const unsigned per_lane = dim / kLanes, split = static_cast<unsigned>(q_a.size());
    const unsigned live = split + static_cast<unsigned>(q_b.size());
    std::array<float, kLanes> part{};
    for (unsigned lane = 0; lane < kLanes; ++lane) {
        float s = 0.0F;
        const unsigned end = std::min(live, (lane + 1U) * per_lane);
        for (unsigned c = lane * per_lane; c < end; ++c)
            s = std::fma(c < split ? q_a[c] : q_b[c - split], c < split ? k_a[c] : k_b[c - split], s);
        part[lane] = s;
    }
    for (unsigned o = kLanes / 2U; o > 0U; o >>= 1U) {
        std::array<float, kLanes> next{};
        for (unsigned lane = 0; lane < kLanes; ++lane) next[lane] = part[lane] + part[lane ^ o];
        part = next;
    }
    return part[0];
}

inline float lane_tree_dot(std::span<const float> q, std::span<const float> k, unsigned dim) {
    return lane_tree_dot(q, k, {}, {}, dim);
}

// One head over `keys` rows. score_at(key) = the unscaled lane_tree_dot, value_at(key) = a span of at least out.size()
// values; out.size() is the live width of the value rows (columns of the device row past it are not stored).
template <class ScoreAt, class ValueAt>
void sink_head(std::span<float> out, std::size_t keys, float sink, float scale, ScoreAt score_at, ValueAt value_at) {
    constexpr float ninf = -std::numeric_limits<float>::infinity();
    const auto exp = [](float x) { return numeric::declared_exp(x); };
    const std::size_t width = out.size();
    std::array<float, kWaves> m_w{}, l_w{};
    std::vector<float> acc(std::size_t{kWaves} * width, 0.0F);
    for (unsigned w = 0; w < kWaves; ++w) {
        float m = ninf, l = 0.0F;
        float* a = acc.data() + std::size_t{w} * width;
        for (std::size_t key = w; key < keys; key += kWaves) {
            float s = score_at(key);
            s *= scale;
            const float mn = std::max(m, s), corr = exp(m - mn), weight = exp(s - mn);
            const float lc = l * corr;
            l = lc + weight;
            const auto value = value_at(key);
            for (std::size_t d = 0; d < width; ++d) {
                const float ac = a[d] * corr, wv = weight * value[d];
                a[d] = ac + wv;
            }
            m = mn;
        }
        m_w[w] = m;
        l_w[w] = l;
    }
    float big = ninf;
    for (unsigned w = 0; w < kWaves; ++w) big = std::max(big, m_w[w]);
    big = std::max(big, sink);
    for (std::size_t d = 0; d < width; ++d) {
        float denominator = exp(sink - big), numerator = 0.0F;
        for (unsigned w = 0; w < kWaves; ++w) {
            const float f = m_w[w] == ninf ? 0.0F : exp(m_w[w] - big);
            const float lf = l_w[w] * f, af = acc[std::size_t{w} * width + d] * f;
            denominator = denominator + lf;
            numerator = numerator + af;
        }
        out[d] = numerator / denominator;
    }
}

// 1 / sqrt(sum of squares / n + eps) in the lane-strided halving order of the device row norms.
inline float rms_inv(std::span<const float> row, float epsilon) {
    if (row.empty()) throw std::invalid_argument("declared RMS: empty row");
    return core::rms_inv_lane_tree(row, epsilon);
}

}  // namespace strix::attn_core::declared_order
