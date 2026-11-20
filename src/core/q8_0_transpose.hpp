#pragma once
// Shared host weight transform: Q8_0 [batch][rows][cols] (blocks of 32 along cols) -> Q8_0 [batch][cols][rows] (blocks of
// 32 along rows), for a GEMM that contracts the other axis of a stored Q8_0 matrix (MLA expanded-MHA prefill: attn_k_b is
// stored [head][kv_lora][qk_nope] for the absorbed query, the expanded key k_nope = W^T c contracts over kv_lora).
// Every source value d * q is exact in FP32 (11-bit scale x 8-bit code); each output block is re-quantised by the ggml
// Q8_0 rule (d = amax / 127, id = 1 / d in FP32, scale stored as f16 RNE, code = roundf(v * id)). FAST: one extra rounding
// per element, |v' - v| <= d' / 2 (+ the f16 rounding of d' times |q'|); exact when every output block's amax is 127 x a
// power-of-two source scale shared by the block (host test). Host only, deterministic, no device code.
#include "core/iu4_onfly.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strix::core::q8_0 {

inline constexpr std::size_t kBlock = 32U, kBlockBytes = 34U;
constexpr std::size_t row_bytes(std::size_t width) { return width / kBlock * kBlockBytes; }

// Value of element (r, c) of one [rows][cols] Q8_0 matrix at m.
inline float value(const std::uint8_t* m, std::size_t cols, std::size_t r, std::size_t c) {
    const std::uint8_t* b = m + r * row_bytes(cols) + c / kBlock * kBlockBytes;
    const auto d = static_cast<std::uint16_t>(b[0] | (b[1] << 8U));
    return onfly::f16_bits_to_f32(d) * static_cast<float>(static_cast<std::int8_t>(b[2U + c % kBlock]));
}

// One Q8_0 block of 32 FP32 values by the ggml reference rule.
inline void quantize_block(const float* x, std::uint8_t* out) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < kBlock; ++j) amax = std::fmax(amax, std::fabs(x[j]));
    const float d = amax / 127.0F, id = d != 0.0F ? 1.0F / d : 0.0F;
    const std::uint16_t h = onfly::f32_to_f16_bits(d);
    out[0] = static_cast<std::uint8_t>(h & 0xffU);
    out[1] = static_cast<std::uint8_t>(h >> 8U);
    for (std::size_t j = 0; j < kBlock; ++j) out[2U + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::roundf(x[j] * id)));
}

// src: batch matrices of [rows][cols] Q8_0 (rows x row_bytes(cols) bytes each). Returns batch x [cols][rows] Q8_0.
inline std::vector<std::uint8_t> transpose(const std::uint8_t* src, std::size_t batch, std::size_t rows, std::size_t cols) {
    if (src == nullptr || batch == 0U || rows == 0U || cols == 0U || rows % kBlock != 0U || cols % kBlock != 0U)
        throw std::invalid_argument("q8_0::transpose: rows and cols must be non-zero multiples of 32");
    const std::size_t in_bytes = rows * row_bytes(cols), out_row = row_bytes(rows), out_bytes = cols * out_row;
    std::vector<std::uint8_t> out(batch * out_bytes);
    float x[kBlock];
    for (std::size_t b = 0; b < batch; ++b) {
        const std::uint8_t* m = src + b * in_bytes;
        std::uint8_t* t = out.data() + b * out_bytes;
        for (std::size_t c = 0; c < cols; ++c)
            for (std::size_t r0 = 0; r0 < rows; r0 += kBlock) {
                for (std::size_t j = 0; j < kBlock; ++j) x[j] = value(m, cols, r0 + j, c);
                quantize_block(x, t + c * out_row + r0 / kBlock * kBlockBytes);
            }
    }
    return out;
}

}  // namespace strix::core::q8_0
