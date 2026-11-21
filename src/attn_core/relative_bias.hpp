#pragma once

#include <cstddef>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_ATTN_BIAS_FN __host__ __device__ constexpr
#else
#define STRIX_ATTN_BIAS_FN constexpr
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
