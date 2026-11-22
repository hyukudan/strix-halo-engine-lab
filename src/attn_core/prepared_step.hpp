#pragma once

#include "core/decode_state_rows.hpp"
#include "prepared_gqa_attention.hpp"

namespace strix::attn_core {

struct PreparedStepCache {
    core::decode_state::PositionView keys, values;
    // Pending images are dense [step row][KV head][column]. S1 publication
    // copies only the accepted branch into the committed position views. The
    // two images and committed storage are disjoint and 16-byte aligned.
    std::uint16_t* pending_keys;
    std::uint16_t* pending_values;
};

struct PreparedStepKv {
    const exec::StepBlock* step;
    PreparedStepCache cache;
    const float* keys;
    const float* values;
    unsigned kv_heads, width;
};

struct PreparedStepFlash {
    const exec::StepBlock* step;
    PreparedStepCache cache;
    const float* queries;          // [step row][query head][128]
    const float* relative_logits;  // [step row][query head][backward distance]
    const float* sinks;            // [step row][query head]
    float* output;                 // [step row][query head][128]
    unsigned heads, kv_heads, relative_extent, window;
    float score_scale;
};

inline bool valid_prepared_cache(const PreparedStepCache& cache, unsigned heads, unsigned width) {
    const auto bytes = std::uint64_t{heads} * width * sizeof(std::uint16_t);
    return cache.pending_keys && cache.pending_values && cache.pending_keys != cache.pending_values &&
        cache.keys.valid(bytes) && cache.values.valid(bytes);
}

}  // namespace strix::attn_core

#if defined(__CUDACC__) && !defined(__HIPCC__)
namespace strix::cuda_core {
template <unsigned Rows> cudaError_t prepared_step_kv(const attn_core::PreparedStepKv&, cudaStream_t);
template <unsigned Rows> cudaError_t prepared_step_flash(const attn_core::PreparedStepFlash&, cudaStream_t);
}  // namespace strix::cuda_core
#endif
