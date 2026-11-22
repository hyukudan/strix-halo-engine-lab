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

// CUDA bodies: strix::cuda_core::prepared_step_kv / prepared_step_flash (declared by their binder,
// src/exec/cuda_decode_wr2.hpp; defined in src/cuda/core). This contract header stays SDK-free.
