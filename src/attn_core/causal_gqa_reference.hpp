#pragma once

#include "attn_core/masks.hpp"
#include "attn_core/relative_bias.hpp"
#include "strix/attn/cache_rows.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::attn_core {

enum class ReferenceAttentionArithmetic { fp32, bf16_eager };

struct ReferenceAttentionShape {
    unsigned query_heads;
    unsigned kv_heads;
    unsigned head_dim;
    unsigned window = 0;  // 0 means the full causal prefix, otherwise includes the current row.
    ReferenceAttentionArithmetic arithmetic = ReferenceAttentionArithmetic::fp32;
};

struct ReferenceAttentionInput {
    unsigned first_position;
    std::span<const float> queries;  // [rows][query_heads][head_dim]
    std::span<const float> keys;     // [rows][kv_heads][head_dim]
    std::span<const float> values;   // [rows][kv_heads][head_dim]
    unsigned relative_extent;       // 0 disables the relative bias.
    std::span<const float> relative_logits;  // [rows][query_heads][backward distance]
    std::span<const float> sink_logits;      // [rows][query_heads], or empty to disable sinks.
};

struct ReferenceAttentionCache {
    std::span<float> keys;    // [slots][kv_heads][head_dim]
    std::span<float> values;  // [slots][kv_heads][head_dim]
};

namespace detail {

inline bool finite_attention_row(std::span<const float> row) {
    return std::ranges::all_of(row, [](float value) { return strix::attn::cache_rows::finite(value); });
}

inline float attention_round_bf16(float value) {
    using namespace strix::attn::cache_rows;
    if (!finite(value)) throw std::invalid_argument("causal attention: non-finite intermediate");
    const auto rounded = encode(value);
    if (!finite(rounded)) throw std::invalid_argument("causal attention: BF16 overflow");
    return decode(rounded);
}

template<class ScoreAt, class ValueAt>
void causal_fp32_head(std::span<float> result, std::size_t keys, float sink,
                      ScoreAt score_at, ValueAt value_at) {
    // The zero-value sink seeds only the online softmax denominator.
    float maximum = sink;
    float denominator = std::isfinite(sink) ? 1.0F : 0.0F;
    for (std::size_t key = 0; key < keys; ++key) {
        const float score = score_at(key);
        const float new_maximum = std::max(maximum, score);
        const float previous_weight = std::exp(maximum - new_maximum);
        const float weight = std::exp(score - new_maximum);
        denominator = denominator * previous_weight + weight;
        const auto value = value_at(key);
        for (std::size_t d = 0; d < result.size(); ++d) {
            result[d] = result[d] * previous_weight + weight * value[d];
        }
        maximum = new_maximum;
    }
    for (auto& value : result) value /= denominator;
}

template<class ScoreAt, class ValueAt>
void causal_bf16_head(std::span<float> result, std::span<float> weights, float sink,
                      ScoreAt score_at, ValueAt value_at) {
    float maximum = sink;
    for (std::size_t key = 0; key < weights.size(); ++key) {
        weights[key] = score_at(key);
        maximum = std::max(maximum, weights[key]);
    }

    // Materialize the normalized probability before multiplying by V so the
    // BF16 round happens after normalization and before the FP32 PV sum.
    float denominator = 0.0F;
    for (auto& weight : weights) {
        weight = std::exp(weight - maximum);
        denominator += weight;
    }
    denominator += std::exp(sink - maximum);
    for (std::size_t key = 0; key < weights.size(); ++key) {
        const float probability = attention_round_bf16(weights[key] / denominator);
        const auto value = value_at(key);
        for (std::size_t d = 0; d < result.size(); ++d) result[d] += probability * value[d];
    }
    for (auto& value : result) value = attention_round_bf16(value);
}

}  // namespace detail

// Scalar reference for a contiguous sequence. Pass already projected and
// normalized Q/K/V and finite score operands. The caller owns the cache and must
// have stored the visible prefix through first_position - 1 at the same layout.
// With a window, slots [0, window) form the ring. Otherwise slot = position.
// Writes each row's K/V before attending and before appending any future row, so
// a chunk larger than the ring is safe. Output has the same shape as Q. The
// mutable buffers (cache and output) must be disjoint from the other spans.
// BF16 eager arithmetic expects BF16-rounded operands widened to float. It
// emulates stage boundaries, without prescribing a parallel reduction order.
inline void causal_gqa_reference(
    const ReferenceAttentionShape& shape, const ReferenceAttentionInput& input,
    ReferenceAttentionCache cache, std::span<float> output) {
    if (shape.head_dim == 0 || shape.query_heads == 0 || shape.kv_heads == 0 ||
        shape.query_heads % shape.kv_heads != 0 ||
        shape.query_heads > std::numeric_limits<unsigned>::max() / shape.head_dim) {
        throw std::invalid_argument("causal attention: invalid head geometry");
    }

    const unsigned q_width = shape.query_heads * shape.head_dim;
    const unsigned kv_width = shape.kv_heads * shape.head_dim;
    const auto rows = input.queries.size() / q_width;
    const auto head_rows = input.queries.size() / shape.head_dim;
    if (input.queries.size() % q_width != 0 || output.size() != input.queries.size() ||
        input.keys.size() % kv_width != 0 || input.keys.size() / kv_width != rows ||
        input.values.size() != input.keys.size() || cache.values.size() != cache.keys.size() ||
        cache.keys.size() % kv_width != 0 ||
        (!input.sink_logits.empty() && input.sink_logits.size() != head_rows)) {
        throw std::invalid_argument("causal attention: invalid row shape");
    }

    const unsigned extent = input.relative_extent;
    if (extent == 0 ? !input.relative_logits.empty()
                    : input.relative_logits.size() % extent != 0 ||
                          input.relative_logits.size() / extent != head_rows) {
        throw std::invalid_argument("causal attention: invalid relative bias shape");
    }
    if (rows == 0) return;
    if (rows - 1 > std::numeric_limits<unsigned>::max() - input.first_position) {
        throw std::invalid_argument("causal attention: position overflow");
    }
    const auto last_position = input.first_position + static_cast<unsigned>(rows - 1);
    const auto slots = cache.keys.size() / kv_width;
    if (shape.window == 0 ? slots <= last_position : slots < shape.window) {
        throw std::invalid_argument("causal attention: insufficient cache rows");
    }
    if (!(detail::finite_attention_row(input.queries) && detail::finite_attention_row(input.keys) &&
          detail::finite_attention_row(input.values) && detail::finite_attention_row(input.relative_logits) &&
          detail::finite_attention_row(input.sink_logits))) {
        throw std::invalid_argument("causal attention: non-finite input");
    }

    const strix::attn::cache_rows::Rows layout{kv_width, shape.window};
    const RelativeBiasRows bias{input.relative_logits.data(), shape.query_heads, extent};
    const AttentionSinkRows sinks{input.sink_logits.data(), shape.query_heads};
    const float scale = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));
    const unsigned queries_per_kv = shape.query_heads / shape.kv_heads;
    const bool bf16 = shape.arithmetic == ReferenceAttentionArithmetic::bf16_eager;
    std::vector<float> weights;
    if (bf16) weights.resize(std::size_t(last_position - mask::window_lo(last_position, shape.window)) + 1);

    for (std::size_t row = 0; row < rows; ++row) {
        const auto position = input.first_position + static_cast<unsigned>(row);
        const auto row_offset = strix::attn::cache_rows::row_offset(layout, position);
        std::ranges::copy(input.keys.subspan(row * kv_width, kv_width), cache.keys.subspan(row_offset).begin());
        std::ranges::copy(input.values.subspan(row * kv_width, kv_width), cache.values.subspan(row_offset).begin());
        const unsigned first_key = mask::window_lo(position, shape.window);
        const std::size_t keys = std::size_t(position - first_key) + 1;
        for (unsigned head = 0; head < shape.query_heads; ++head) {
            const auto query_head = row * q_width + head * shape.head_dim;
            const unsigned kv_head = (head / queries_per_kv) * shape.head_dim;
            auto result = output.subspan(query_head, shape.head_dim);
            std::ranges::fill(result, 0.0F);
            const float sink = input.sink_logits.empty() ? -std::numeric_limits<float>::infinity()
                                                       : sinks.sink(static_cast<unsigned>(row), head);
            const auto cache_offset = [&](unsigned key_position) {
                return strix::attn::cache_rows::row_offset(layout, key_position) + kv_head;
            };
            const auto score_at = [&](std::size_t key) {
                const auto key_position = first_key + static_cast<unsigned>(key);
                const auto kv_offset = cache_offset(key_position);
                float dot = 0.0F;
                for (unsigned d = 0; d < shape.head_dim; ++d) {
                    dot += input.queries[query_head + d] * cache.keys[kv_offset + d];
                }
                const float rel = bias.score_bias(static_cast<unsigned>(row), head, position, key_position);
                if (bf16) {
                    const float scaled = detail::attention_round_bf16(detail::attention_round_bf16(dot) * scale);
                    return detail::attention_round_bf16(scaled + rel);
                }
                const float score = dot * scale + rel;
                if (!std::isfinite(score)) throw std::invalid_argument("causal attention: non-finite score");
                return score;
            };
            const auto value_at = [&](std::size_t key) {
                const auto offset = cache_offset(first_key + static_cast<unsigned>(key));
                return cache.values.subspan(offset, shape.head_dim);
            };
            if (bf16) {
                detail::causal_bf16_head(result, std::span(weights).first(keys), sink, score_at, value_at);
            } else {
                detail::causal_fp32_head(result, keys, sink, score_at, value_at);
                if (!detail::finite_attention_row(result)) throw std::invalid_argument("causal attention: non-finite output");
            }
        }
    }
}

}  // namespace strix::attn_core
