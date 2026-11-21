#pragma once

#include <cstdint>

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

}  // namespace strix::attn_core
