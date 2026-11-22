#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "strix/weight/formats/scalar_float.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_ATTN_BIAS_FN __host__ __device__ constexpr
#define STRIX_ATTN_BIAS_HD __host__ __device__ inline
#else
#define STRIX_ATTN_BIAS_FN constexpr
#define STRIX_ATTN_BIAS_HD inline
#endif

namespace strix::attn_core {

// Bias logits in natural units, projected once per query row and head.
// The position arguments are absolute token positions, even with a wrapped KV cache.
struct RelativeBiasRows {
    const float* logits;  // [row][head][backward distance]
    unsigned heads;
    unsigned extent;     // Exclusive bound on backward distance. Zero disables bias.

    STRIX_ATTN_BIAS_FN float score_bias(unsigned row, unsigned head, unsigned query_pos, unsigned key_pos) const {
        if (key_pos > query_pos) return 0.0F;
        const unsigned distance = query_pos - key_pos;
        if (distance >= extent) return 0.0F;
        const std::size_t query = std::size_t(row) * heads + head;
        return logits[query * extent + distance];
    }
};

// The same bias projected where a consumer reads it, instead of a [row][head][distance] image:
// logit(row, head, d) = bf16_round(x[0] w[0][d] + ... + x[Width - 1] w[Width - 1][d]) as ONE FMA chain from 0, k
// ascending, x = features[row][head][0 .. Width), w = BF16 weights [Width][extent]. This is the chain and rounding of
// core::row_transforms::ShortBf16Projection<Width> (ascending), so a consumer gets the bits of the projected rows.
struct RelativeBiasProjection {
    const float* features;         // [row][head][Width] FP32
    const std::uint16_t* weights;  // [Width][extent] BF16
    unsigned heads;
    unsigned extent;               // Exclusive bound on backward distance. Zero disables bias.
};

// One logit of the chain. `pairs` holds the Width BF16 weights of one distance, k-contiguous, as little-endian
// 32-bit words (weight k is the low half of word k / 2 for even k, the high half for odd k).
template <unsigned Width>
STRIX_ATTN_BIAS_HD float relative_bias_chain(const float* x, const std::uint32_t* pairs) {
    static_assert(Width % 2U == 0U);
    namespace sf = weight::scalar_float;
    float result = 0.0F;
    for (unsigned k = 0; k < Width; ++k) {
        const std::uint32_t word = pairs[k / 2U];
        result = std::fma(x[k], sf::u2f((k % 2U) != 0U ? word & 0xFFFF0000U : word << 16U), result);
    }
    return sf::bf16_round(result);
}

// True when every logit of these features is finite for weights with |w| <= w_max (w_max = the largest |weight|,
// infinite or NaN when a weight is). Each partial sum of a chain is at most sum |x_k| w_max (1 + 2^-24)^Width; the
// FP32 bound below is at least that sum times (1 - 2^-24)^Width, so bound < 2^126 keeps every partial below 2^127
// and bf16_round keeps |v| <= 2^127 finite. Non-finite x or w_max make the bound NaN or infinite (false). When this
// returns false a consumer that must flag non-finite logits evaluates every distance.
template <unsigned Width>
STRIX_ATTN_BIAS_HD bool relative_bias_bounded(const float* x, float w_max) {
    float bound = 0.0F;
    for (unsigned k = 0; k < Width; ++k) bound = std::fma(std::fabs(x[k]), w_max, bound);
    return bound < 0x1p126F;
}

// Each sink is an extra softmax logit in natural units with an all-zero value.
struct AttentionSinkRows {
    const float* logits;  // [row][head]
    unsigned heads;

    STRIX_ATTN_BIAS_FN float sink(unsigned row, unsigned head) const {
        return logits[std::size_t(row) * heads + head];
    }
};

} // namespace strix::attn_core

#undef STRIX_ATTN_BIAS_FN
#undef STRIX_ATTN_BIAS_HD
