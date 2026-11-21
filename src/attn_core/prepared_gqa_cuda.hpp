#pragma once

#include "attn_core/prepared_gqa_attention.hpp"
#include "strix/cuda_core/attention.hpp"

namespace strix::attn_core {

struct PreparedGqaCudaCache {
    std::uint16_t* keys;
    std::uint16_t* values;
    unsigned slots;
    bool wrapped;
};

// Both appends and attention use the logical visit's slot-major BF16 slice.
// Positions remain absolute for the attention mask and relative bias.
// Inputs and output must be disjoint and Q, K/V cache, and output aligned to 16 bytes.
template <class Shape>
inline cudaError_t enqueue_prepared_gqa_cuda(
    unsigned window, const PreparedGqaAttentionRows& rows, PreparedGqaCudaCache cache,
    cudaStream_t stream, cuda_core::FlashPipe pipe = cuda_core::FlashPipe::pad2, unsigned* failure = nullptr) {
    if (rows.first_position >= Shape::kMaxContext ||
        rows.count > Shape::kMaxContext - rows.first_position ||
        !rows.queries || !rows.keys || !rows.values || !rows.relative_logits || !rows.sinks || !rows.output) {
        return cudaErrorInvalidValue;
    }
    if (cache.slots == 0U) return cudaErrorInvalidValue;

    static_assert(sizeof(__nv_bfloat16) == sizeof(std::uint16_t));
    auto* keys = reinterpret_cast<__nv_bfloat16*>(cache.keys);
    auto* values = reinterpret_cast<__nv_bfloat16*>(cache.values);
    const cuda_core::AttentionDims dims{Shape::kQueryHeads, Shape::kKvHeads, Shape::kHeadWidth, Shape::kScoreScale, cache.slots};
    const cuda_core::PreparedBf16Flash prepared{rows.relative_logits, rows.sinks, Shape::kRelativeExtent, window, cache.wrapped};
    return cuda_core::append_attention_prepared_bf16(
        dims, prepared, rows.queries, rows.keys, rows.values, keys, values,
        rows.first_position, rows.count, rows.output, stream, pipe, failure);
}

}  // namespace strix::attn_core
