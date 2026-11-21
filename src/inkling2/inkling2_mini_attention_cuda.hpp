#pragma once

#include "inkling2_mini_state.hpp"
#include "strix/cuda_core/attention.hpp"

namespace strix::inkling2 {

struct MiniCudaAttentionRows {
    const float* queries;          // [row][query head][dim] FP32
    const float* keys;             // [row][KV head][dim] FP32
    const float* values;           // [row][KV head][dim] FP32
    const float* relative_logits;  // [row][query head][backward distance] FP32, natural units
    const float* sinks;            // [row][query head] FP32, natural units
    float* output;                 // [row][query head][dim] FP32 before the Mini output gate
    unsigned first_position;
    unsigned count;
};

// Both appends and attention use the logical visit's slot-major BF16 slice.
// Positions remain absolute for the attention mask and relative bias.
// Inputs and output must be disjoint and Q, K/V cache, and output aligned to 16 bytes.
inline cudaError_t enqueue_mini_cuda_attention(
    LayerVisit visit, const MiniCudaAttentionRows& rows, const MiniSessionStatePlan& session,
    MiniSessionKvPlanes planes, cudaStream_t stream, cuda_core::FlashPipe pipe = cuda_core::FlashPipe::pad2) {
    if (!session.admits(rows.first_position, rows.count) ||
        !rows.queries || !rows.keys || !rows.values || !rows.relative_logits || !rows.sinks || !rows.output) {
        return cudaErrorInvalidValue;
    }
    const auto cache = mini_visit_kv_cache(session, visit, planes);
    if (cache.slots == 0U) return cudaErrorInvalidValue;

    static_assert(sizeof(__nv_bfloat16) == sizeof(std::uint16_t));
    auto* keys = reinterpret_cast<__nv_bfloat16*>(cache.keys);
    auto* values = reinterpret_cast<__nv_bfloat16*>(cache.values);
    const unsigned window = is_full_attention(visit.physical_layer) ? 0U : kSlidingWindow;
    const cuda_core::AttentionDims dims{kQueryHeads, kKvHeads, kHeadWidth, 0.08838834764831844F, cache.slots};
    const cuda_core::PreparedBf16Flash prepared{rows.relative_logits, rows.sinks, kRelativeExtent, window, cache.wrapped};
    if (!cuda_core::can_launch_prepared_bf16_flash(
            dims, prepared, rows.queries, keys, values, rows.first_position, rows.count, rows.output, pipe)) {
        return cudaErrorInvalidValue;
    }

    const auto status = cuda_core::kv_store_prepared_bf16(
        dims, rows.keys, rows.values, keys, values, rows.first_position, rows.count, cache.wrapped, stream);
    if (status != cudaSuccess) return status;
    return cuda_core::attention_prefill_prepared_bf16(
        dims, prepared, rows.queries, keys, values, rows.first_position, rows.count, rows.output, stream, pipe);
}

}  // namespace strix::inkling2
