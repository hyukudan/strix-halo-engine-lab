#pragma once

#include "strix/recurrent_state_type.hpp"

#include <cstdint>
#include <limits>

namespace strix::ops::recur {

// Row and state extents are in elements. The matrix is key-major per head.
struct GdnDecodeShape {
    unsigned hidden, key_heads, value_heads, head_dim, taps, a8_pitch;

    [[nodiscard]] constexpr bool operator==(const GdnDecodeShape&) const noexcept = default;

    [[nodiscard]] constexpr bool valid() const noexcept {
        constexpr auto limit = std::numeric_limits<unsigned>::max();
        if (!hidden || !key_heads || !value_heads || !head_dim || head_dim % 32U ||
            value_heads % key_heads || taps < 2U || !a8_pitch ||
            key_heads > limit / head_dim || value_heads > limit / head_dim)
            return false;
        const auto width = 2U * std::uint64_t{key_width()} + value_width();
        return width <= limit / taps && value_width() <= limit / head_dim &&
            value_heads <= limit / hidden && value_width() / 32U <= limit / a8_pitch;
    }

    // Extents below require a validated shape.
    [[nodiscard]] constexpr unsigned key_width() const noexcept { return key_heads * head_dim; }
    [[nodiscard]] constexpr unsigned value_width() const noexcept { return value_heads * head_dim; }
    [[nodiscard]] constexpr unsigned channels() const noexcept { return 2U * key_width() + value_width(); }
    [[nodiscard]] constexpr unsigned history_elements() const noexcept { return channels() * taps; }
    [[nodiscard]] constexpr unsigned a8_deltas() const noexcept { return value_width() / 32U * a8_pitch; }
    [[nodiscard]] constexpr unsigned matrix_elements() const noexcept { return value_width() * head_dim; }
    [[nodiscard]] constexpr std::uint64_t matrix_bytes(RecurrentStateType type) const noexcept {
        return recurrent_state_bytes(matrix_elements(), type);
    }
};

namespace shape {

inline constexpr GdnDecodeShape gdn_2560_16x48x128{2560, 16, 48, 128, 4, 256};

}  // namespace shape

template <GdnDecodeShape Shape>
struct GdnDecodeLayout {
    static_assert(Shape.valid());
    static constexpr unsigned hidden = Shape.hidden, key_heads = Shape.key_heads, value_heads = Shape.value_heads;
    static constexpr unsigned head_dim = Shape.head_dim, taps = Shape.taps, a8_pitch = Shape.a8_pitch;
    static constexpr unsigned key_width = Shape.key_width(), value_width = Shape.value_width();
    static constexpr unsigned channels = Shape.channels(), history_elements = Shape.history_elements();
    static constexpr unsigned a8_deltas = Shape.a8_deltas(), matrix_elements = Shape.matrix_elements();

    template <RecurrentStateType Type>
    static constexpr auto matrix_bytes = Shape.matrix_bytes(Type);
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
    std::uint16_t* history;     // BF16 [channels][taps], shifted in place
    const void* matrix_in;      // [value_heads][key][value], type below
    void* matrix_out;           // may equal matrix_in
    RecurrentStateType type = RecurrentStateType::f32;
};

struct DecodeA8 {
    std::int8_t* codes = nullptr; // [value_width]
    float* deltas = nullptr;      // [value_width / 32][a8_pitch], writes row 0
};

}  // namespace strix::ops::recur
