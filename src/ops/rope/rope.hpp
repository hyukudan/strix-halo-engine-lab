#pragma once
// Shared RoPE op (one family per backend; HIP body in rope_device.hpp). Table-driven rotation of adjacent pairs
// (2p, 2p + 1) of a slice of every vector, in place, FP32 storage:
//   for r < rows, v < vectors, p < pairs: a, b = x[r * row_stride + v * vector_stride + offset + 2p + {0, 1}],
//   t = positions[r] * pairs + p, c = cos[t], s = inverse ? -sin[t] : sin[t],
//   a' = R(a * c - b * s), b' = R(a * s + b * c), R = identity (Round::none) or BF16 RNE (Round::bf16).
// Round::none keeps the products / difference as the binder's compiler emits them (DS41: contracted, its bits are
// unchanged by the lift). Round::bf16 evaluates every product and sum separately (no contraction), so the device
// equals reference_apply bit for bit.
// The tables are data: a binder builds them with its own policy (DS41: YaRN, inverse on the attention output) or with
// build_native_table (no scaling, one theta per layer kind, e.g. theta 1e7 on one layer kind, 1e4 on another).
// Binders: DS41 (hip_ds41_layer.hip enqueue_rope, Round::none), Round::bf16 bindings (native tables, cos / sin BF16).
#include "strix/ds41_layer.hpp"  // round_to_bf16: the declared BF16 RNE shared by the declared-arithmetic models

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strix::ops::rope {

struct View {
    float* x = nullptr;
    std::uint32_t rows = 0, vectors = 1, row_stride = 0, vector_stride = 0, offset = 0, pairs = 32;
    const float* cos = nullptr;
    const float* sin = nullptr;
    const std::uint32_t* positions = nullptr;
    bool inverse = false;
};

enum class Round : std::uint32_t { none, bf16 };

inline void require_view(const View& v) {
    if (!v.x || !v.cos || !v.sin || !v.positions || !v.rows || !v.vectors || !v.pairs)
        throw std::invalid_argument("RoPE view");
    if (v.offset % 2U != 0 || v.row_stride % 2U != 0 || v.vector_stride % 2U != 0)
        throw std::invalid_argument("RoPE view alignment");
}

// Native table rows for positions first .. first + count - 1 (row i = position first + i, `pairs` entries each):
// inv_p = 1 / powf(theta, 2p / (2 pairs)) and angle = float(position) * inv_p in FP32, cos / sin evaluated in double on
// that FP32 angle and narrowed to FP32, then rounded by `round` (e.g. BF16, when the reference model rounds the tables
// to the activation dtype). Host only.
inline void build_native_table(float theta, std::uint32_t pairs, std::uint32_t first, std::uint32_t count, Round round,
                               float* cos_out, float* sin_out) {
    if (!(theta > 1.0F) || pairs == 0 || !cos_out || !sin_out) throw std::invalid_argument("RoPE native table");
    const float dims = static_cast<float>(2U * pairs);
    for (std::uint32_t p = 0; p < pairs; ++p) {
        const float inv = 1.0F / std::pow(theta, static_cast<float>(2U * p) / dims);
        for (std::uint32_t i = 0; i < count; ++i) {
            const float angle = static_cast<float>(first + i) * inv;
            float c = static_cast<float>(std::cos(static_cast<double>(angle)));
            float s = static_cast<float>(std::sin(static_cast<double>(angle)));
            if (round == Round::bf16) {
                c = ds41::round_to_bf16(c);
                s = ds41::round_to_bf16(s);
            }
            cos_out[std::size_t{i} * pairs + p] = c;
            sin_out[std::size_t{i} * pairs + p] = s;
        }
    }
}

// Host reference of the device op (Round::bf16 is bitwise the device; Round::none matches up to contraction).
inline void reference_apply(const View& v, Round round) {
    require_view(v);
    for (std::uint32_t r = 0; r < v.rows; ++r)
        for (std::uint32_t vec = 0; vec < v.vectors; ++vec)
            for (std::uint32_t p = 0; p < v.pairs; ++p) {
                float* x = v.x + std::size_t{r} * v.row_stride + std::size_t{vec} * v.vector_stride + v.offset;
                const std::size_t t = std::size_t{v.positions[r]} * v.pairs + p;
                const float c = v.cos[t], s = v.inverse ? -v.sin[t] : v.sin[t];
                const float a = x[2U * p], b = x[2U * p + 1U];
                const float ac = a * c, bs = b * s, as = a * s, bc = b * c;
                const float y0 = ac - bs, y1 = as + bc;
                x[2U * p] = round == Round::bf16 ? ds41::round_to_bf16(y0) : y0;
                x[2U * p + 1U] = round == Round::bf16 ? ds41::round_to_bf16(y1) : y1;
            }
}

}  // namespace strix::ops::rope
