#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::core {

// SDK-free reference operator contract. The caller owns model bytes and all
// host spans. Each operation completes before returning or throwing.
struct Q8Rows {
    const std::uint8_t* data;
    std::uint32_t rows, width;
    std::size_t row_bytes;
};

struct DenseAttentionRow {
    std::span<const float> query, keys, values; // Q[head][d], K/V[position][kv_head][d]
    std::size_t positions;
    std::uint32_t heads, kv_heads, width;
};

class DenseReference {
public:
    virtual ~DenseReference() = default;
    virtual void gather(Q8Rows, std::uint32_t row, std::span<float> output) = 0;
    virtual void linear(Q8Rows, std::span<const float> input, std::span<float> output) = 0;
    virtual void rms_norm(std::span<const float> input, std::span<const float> scale,
                          std::span<float> output, std::uint32_t width, float epsilon) = 0;
    virtual void rope(std::span<float> heads, std::span<const float> interleaved,
                      std::uint32_t width) = 0;
    virtual void attention(const DenseAttentionRow&, std::span<float> output) = 0;
};

} // namespace strix::core
