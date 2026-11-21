#pragma once

#include "attn_core/causal_gqa_reference.hpp"
#include "core/linear_reference.hpp"

namespace strix::attn_core {

// Standard bidirectional GQA over a preselected, nonempty key range. All
// query rows see the same keys. The caller owns windowing and cache lifetime.
inline void bidirectional_gqa_reference(unsigned query_heads, unsigned kv_heads, unsigned head_dim,
                                       std::span<const float> queries, std::span<const float> keys,
                                       std::span<const float> values, std::span<float> output) {
    if (head_dim == 0 || kv_heads == 0 || query_heads == 0 || query_heads % kv_heads != 0)
        throw std::invalid_argument("bidirectional GQA: invalid head geometry");
    const auto query_width = std::size_t{query_heads} * head_dim;
    const auto kv_width = std::size_t{kv_heads} * head_dim;
    if (keys.empty() || keys.size() % kv_width != 0 || values.size() != keys.size() ||
        queries.size() % query_width != 0 || output.size() != queries.size())
        throw std::invalid_argument("bidirectional GQA: invalid row shape");
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    for (std::size_t row = 0; row < queries.size(); row += query_width) {
        for (unsigned head = 0; head < query_heads; ++head) {
            const auto q_offset = row + std::size_t{head} * head_dim;
            const auto q = queries.subspan(q_offset, head_dim);
            const auto kv_head = std::size_t{head / (query_heads / kv_heads)} * head_dim;
            auto result = output.subspan(q_offset, head_dim);
            std::ranges::fill(result, 0.0F);
            detail::causal_fp32_head(result, keys.size() / kv_width, -std::numeric_limits<float>::infinity(),
                [&](std::size_t i) { return scale * core::dot_f32(q, keys.subspan(i * kv_width + kv_head, head_dim)); },
                [&](std::size_t i) { return values.subspan(i * kv_width + kv_head, head_dim); });
            if (!detail::finite_attention_row(result))
                throw std::runtime_error("bidirectional GQA: non-finite output");
        }
    }
}

// Rotate split-half heads at one absolute position. BF16 is a caller-owned
// rounding boundary, so this also serves FP32 references.
inline void split_half_rotary_reference(std::span<float> row, unsigned head_dim, float theta,
                                       std::uint64_t position) {
    if (head_dim == 0 || head_dim % 2U != 0 || row.size() % head_dim != 0 || !(theta > 0.0F))
        throw std::invalid_argument("split-half rotary: invalid head geometry");
    const auto half = head_dim / 2U;
    for (unsigned channel = 0; channel < half; ++channel) {
        const auto angle = static_cast<float>(position) /
                           std::pow(theta, static_cast<float>(channel) / static_cast<float>(half));
        const auto cosine = std::cos(angle), sine = std::sin(angle);
        for (std::size_t head = 0; head < row.size(); head += head_dim) {
            auto& a = row[head + channel];
            auto& b = row[head + channel + half];
            const auto x = a, y = b;
            a = x * cosine - y * sine;
            b = x * sine + y * cosine;
        }
    }
}

}  // namespace strix::attn_core
