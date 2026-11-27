#pragma once

#include "attn_core/hybrid_reference.hpp"
#include "attn_core/reference_cache.hpp"
#include "attn_core/rotary_reference.hpp"
#include "core/bf16_reference.hpp"
#include "core/block_fp8_activation_reference.hpp"
#include "strix/attn/reference_history.hpp"

namespace strix::attn_core {

// After the fold, preserve the FP32 values through the inverse rotation.
inline void compressed_mqa_output_input_reference(std::span<float> heads, unsigned width,
    const RotaryReferenceRow& rotary, unsigned fp8_columns) {
    rotary.apply(heads, width, true);
    if (fp8_columns != 0) core::round_block_fp8_activation_reference(heads, fp8_columns);
    else core::round_bf16_reference(heads);
}

struct CompressedMqaStep {
    unsigned position, width, window;
    bool record_history, keep_compressor, bf16, separate_values, same_rotary_domain;
};

struct CompressedMqaInputs {
    std::span<float> query, key, local_value;
    std::span<const float> sinks;
    std::span<const float> carry_value, carry_logit, pooled;
};

// Shared local-window and compressed-history reduction. On the same-domain
// path a pooled row is its own key and value. The split-domain path converts
// pooled values to the local frame, retaining one common softmax.
inline std::vector<float> compressed_mqa_reference(const CompressedMqaStep& step, const CompressedMqaInputs& in,
    const RotaryReferenceRow& window_rope, const RotaryReferenceRow& compressed_rope,
    const attn::CompressedReferenceHistory& state) {
    std::vector<float> compressed_query(in.query.begin(), in.query.end());
    window_rope.apply(in.query, step.width);
    window_rope.apply(in.key, step.width);
    if (step.separate_values) window_rope.apply(in.local_value, step.width);
    compressed_rope.apply(compressed_query, step.width);
    if (step.bf16) {
        core::round_bf16_reference(in.query);
        core::round_bf16_reference(in.key);
        core::round_bf16_reference(compressed_query);
    }

    const auto window_start = mask::window_lo(step.position, step.window);
    const auto window_at = [&](std::size_t row) -> std::span<const float> {
        const auto token = window_start + row;
        if (token == step.position) return in.key;
        return std::span(state.window).subspan((token % step.window) * step.width, step.width);
    };
    const auto window_value = [&](std::size_t row) -> std::span<const float> {
        if (!step.separate_values) return window_at(row);
        const auto token = window_start + row;
        if (token == step.position) return in.local_value;
        return std::span(state.window_values).subspan((token % step.window) * step.width, step.width);
    };

    std::vector<float> heads(in.query.size());
    if (!step.record_history) {
        std::vector<float> keys(step.width), values(step.width);
        const auto heads_count = static_cast<unsigned>(in.sinks.size());
        causal_gqa_reference({heads_count, 1, step.width},
            {0, in.query, in.key, window_value(0), 0, {}, in.sinks}, {keys, values}, heads);
        return heads;
    }

    // A group can be visible alongside the overlapping raw window. Reduce
    // all window keys first, then complete compressed rows.
    const auto compressed_count = (state.compressed.size() + in.pooled.size()) / step.width;
    const auto pooled_key = [&](std::size_t row) -> std::span<const float> {
        if (row == state.compressed.size() / step.width) return in.pooled;
        return std::span(state.compressed).subspan(row * step.width, step.width);
    };
    std::vector<float> compressed_values;
    if (!step.same_rotary_domain) {
        compressed_values = state.compressed;
        compressed_values.insert(compressed_values.end(), in.pooled.begin(), in.pooled.end());
        compressed_rope.apply(compressed_values, step.width, true);
        window_rope.apply(compressed_values, step.width);
    }
    const auto pooled_value = [&](std::size_t row) {
        if (step.same_rotary_domain) return pooled_key(row);
        return std::span<const float>(compressed_values).subspan(row * step.width, step.width);
    };
    hybrid_mqa_reference({step.width, in.query, compressed_query, in.sinks},
        mask::window_count(step.position, step.window), compressed_count, window_at, window_value, pooled_key, pooled_value, heads);
    return heads;
}

template<unsigned Ratio>
void validate_compressed_mqa_history(const CompressedMqaStep& step, const attn::CompressedReferenceHistory& state) {
    const auto ring_size = std::size_t{step.window} * step.width;
    const auto carry_size = std::size_t{Ratio - 1U} * step.width;
    if (step.position != 0 && (state.window.size() != ring_size ||
        (step.keep_compressor && state.carry_values.size() != carry_size)))
        throw std::invalid_argument("compressed attention: history geometry changed");
}

template<unsigned Ratio>
void commit_compressed_mqa_history(const CompressedMqaStep& step, const CompressedMqaInputs& in,
                                  attn::CompressedReferenceHistory& state) {
    if (!step.record_history) return;
    if (step.position == 0) {
        const auto ring_size = std::size_t{step.window} * step.width;
        state.window.resize(ring_size);
        if (step.separate_values) state.window_values.resize(ring_size);
        if (step.keep_compressor) {
            const auto carry_size = std::size_t{Ratio - 1U} * step.width;
            state.carry_values.resize(carry_size);
            state.carry_logits.resize(carry_size);
        }
    }
    detail::reserve_append(state.compressed, in.pooled.size());
    const auto window_slot = std::size_t{step.position % step.window} * step.width;
    std::ranges::copy(in.key, std::span(state.window).subspan(window_slot, step.width).begin());
    if (step.separate_values)
        std::ranges::copy(in.local_value, std::span(state.window_values).subspan(window_slot, step.width).begin());
    if (step.keep_compressor) {
        if (in.pooled.empty()) {
            const auto carry_slot = std::size_t{step.position % Ratio} * step.width;
            std::ranges::copy(in.carry_value, std::span(state.carry_values).subspan(carry_slot, step.width).begin());
            std::ranges::copy(in.carry_logit, std::span(state.carry_logits).subspan(carry_slot, step.width).begin());
        } else {
            state.compressed.insert(state.compressed.end(), in.pooled.begin(), in.pooled.end());
        }
    }
}

} // namespace strix::attn_core
