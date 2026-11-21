#pragma once

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/pointwise.hpp"
#include "strix/attn/cache_rows.hpp"

#include <limits>
#endif

namespace strix::attn_core {

struct PreparedGqaAttentionRows {
    const float* queries;          // [row][query head][dim] FP32
    const float* keys;             // [row][KV head][dim] FP32
    const float* values;           // [row][KV head][dim] FP32
    const float* relative_logits;  // [row][query head][backward distance] FP32, natural units
    const float* sinks;            // [row][query head] FP32, natural units
    float* output;                 // [row][query head][dim] FP32
    unsigned first_position;
    unsigned count;
};

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace detail {

struct FiniteAttentionOperands {
    PreparedGqaAttentionRows rows;
    unsigned heads;
    unsigned kv_heads;
    unsigned width;
    unsigned bias_extent;
    unsigned stride;
    unsigned* failure;

    __device__ void operator()(std::size_t i) const {
        namespace cr = strix::attn::cache_rows;
        const auto query_head = i / stride;
        const auto column = i % stride;
        bool valid = column != 0U || !rows.sinks || cr::finite(rows.sinks[query_head]);
        if (column < width) {
            const auto q = query_head * width + column;
            valid = valid && cr::encodes_finite(rows.queries[q]) && cr::finite(rows.output[q]);
            const auto head = query_head % heads;
            if (head < kv_heads) {
                const auto kv = (query_head / heads * kv_heads + head) * width + column;
                valid = valid && cr::encodes_finite(rows.keys[kv]) && cr::encodes_finite(rows.values[kv]);
            }
        }
        if (column < bias_extent) valid = valid && cr::finite(rows.relative_logits[query_head * bias_extent + column]);
        if (!valid) atomicOr(failure, 1U);
    }
};

}  // namespace detail

// Check all operands, including relative logits omitted by the causal mask.
// The caller has already admitted the attention geometry and enqueued the output.
template <class Stream>
inline auto enqueue_validate_prepared_finite(const PreparedGqaAttentionRows& rows, unsigned heads,
                                              unsigned kv_heads, unsigned width, unsigned bias_extent,
                                              unsigned* failure, Stream stream) {
    const unsigned stride = width >= bias_extent ? width : bias_extent;
    const auto query_heads = std::size_t{rows.count} * heads;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (stride == 0U || query_heads > max_elements / stride) return core::pointwise::invalid;
    return core::pointwise::elements(query_heads * stride,
        detail::FiniteAttentionOperands{rows, heads, kv_heads, width, bias_extent, stride, failure}, stream);
}
#endif

}  // namespace strix::attn_core
