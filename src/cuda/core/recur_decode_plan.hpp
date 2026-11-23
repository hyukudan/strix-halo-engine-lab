#pragma once

#include "strix/recurrent_state_type.hpp"

#include <cstdint>

namespace strix::cuda::recur {

// Layout of the gfx1151_v1 striped GDN decode form. Positions are relative to
// the caller-selected row and state images. The state is key-major per head.
struct Gfx1151Decode {
    static constexpr unsigned hidden = 2560, key_heads = 16, value_heads = 48, head_dim = 128, taps = 4;
    static constexpr unsigned key_width = key_heads * head_dim, value_width = value_heads * head_dim;
    static constexpr unsigned channels = 2 * key_width + value_width;
    static constexpr unsigned matrix_elements = value_heads * head_dim * head_dim;
    static constexpr unsigned history_elements = channels * taps;
    static constexpr unsigned a8_pitch = 256, a8_deltas = value_width / 32 * a8_pitch;

    [[nodiscard]] static constexpr std::uint64_t matrix_bytes(RecurrentStateType type) noexcept {
        return recurrent_state_bytes(matrix_elements, type);
    }
};

struct DecodeWeights {
    const float* beta;  // [value_heads][hidden]
    const float* alpha; // [value_heads][hidden]
    const float* dt;
    const float* a;     // negative exp(A_log)
    const float* conv;  // [channels][taps], oldest first
    const float* gain;  // [head_dim]
    float l2_eps = 1e-6F;
    float rms_eps = 1e-6F;
};

struct DecodeRow {
    const std::uint16_t* input;  // BF16 [hidden]
    const float* qkv;           // [Q, K, V]
    const float* z;             // [value_width], BF16-valued FP32 producer output
    float* prepared;            // [channels], 16-byte aligned; can equal qkv
    float* beta;                // [value_heads]
    float* log_decay;           // [value_heads]
    float* out;                 // [value_width]
    float* raw_capture = nullptr; // optional [channels], before BF16 conversion
};

struct DecodeState {
    std::uint16_t* history;     // BF16 [channels][4], shifted in place
    const void* matrix_in;      // [value_heads][key][value], type below
    void* matrix_out;           // may equal matrix_in
    RecurrentStateType type = RecurrentStateType::f32;
};

struct DecodeA8 {
    std::int8_t* codes = nullptr; // [value_width]
    float* deltas = nullptr;      // [value_width / 32][a8_pitch], writes row 0
};

}  // namespace strix::cuda::recur
