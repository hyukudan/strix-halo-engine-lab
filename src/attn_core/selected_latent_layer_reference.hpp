#pragma once

// One selected-latent attention layer (low-rank query and latent key/value projection, an indexer
// that selects history rows, value-head post-mix, sigmoid output gate), composed from the shared
// primitives. The model declares the roles (checkpoint tensor names), the dimensions and the
// rotary rows.

#include "attn_core/layer_reference_common.hpp"
#include "attn_core/selected_latent_reference.hpp"
#include "attn_core/sparse_index_reference.hpp"
#include "core/bf16_reference.hpp"
#include "core/linear_reference.hpp"
#include "strix/attn/head_rms_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace strix::attn_core {

struct SelectedLatentLayerRoles {
    std::string_view q_down, q_norm, q_up, kv_down, kv_norm, kv_up;
    std::string_view index_key, index_key_norm_weight, index_key_norm_bias;
    std::string_view index_query, index_head_weights, index_gate, index_bias;
    std::string_view softmax_offset, o_proj, gate;
    ValueHeadMixRoles mix;
};

struct SelectedLatentLayerShape {
    SelectedLatentShape latent;
    unsigned hidden, q_rank, vha_rank;
    unsigned index_dim, index_heads, top_k, prefill_index_heads, window;
    float eps, index_eps;
    bool checkpoint;
    bool key_inverse_rms; // reference path: divide each key by its RMS (otherwise the learned gain stands alone)
    bool vha_swapped;
    // One packed index key (E2M1 pairs plus one scale byte per 32 elements).
    [[nodiscard]] unsigned packed_index_bytes() const { return index_dim / 2U + index_dim / 32U; }
};

// Rows the indexer keeps for `position`; the window of recent rows joins on the checkpoint path.
template <class Weights>
std::vector<std::uint32_t> select_indexed_rows_reference(const Weights& w, const SelectedLatentLayerRoles& r,
    const SelectedLatentLayerShape& s, const RotaryReferenceRow& rope, std::span<const float> input,
    std::span<const float> q_lora, unsigned position, std::span<const std::uint8_t> index_history,
    std::span<const std::uint8_t> current_key, bool prefill) {
    const bool checkpoint = s.checkpoint;
    const SparseWindowReference window(position, checkpoint ? s.window : 0U, s.top_k);
    if (window.dense()) return window.dense_rows();
    std::vector<float> query(std::size_t{s.index_heads} * s.index_dim), weights(s.index_heads);
    w.linear(r.index_query, q_lora, query, 1U);
    rotate_index_rows_reference(query, s.index_dim, rope, checkpoint, checkpoint);
    if (checkpoint) quantize_index_query_reference(query, s.index_dim);

    w.linear(r.index_head_weights, input, weights, 1U);
    const auto packed = s.packed_index_bytes();
    const auto key_at = [&](std::size_t row) {
        return row == position ? current_key : index_history.subspan(row * packed, packed);
    };
    if (checkpoint) {
        const float scale = 1.0F / std::sqrt(static_cast<float>(s.index_dim * s.index_heads));
        for (auto& weight : weights) weight *= scale;
        if (prefill) {
            const auto gate = w.tensor(r.index_gate, input.size() * s.index_heads);
            std::vector<float> logits(s.index_heads);
            core::linear_reference(gate, input, logits);
            const auto bias = w.tensor(r.index_bias, s.index_heads);
            weights = index_head_weights_reference(weights, logits, bias, s.prefill_index_heads);
        }
        auto selected = weighted_index_top_k_reference(query, weights, s.index_dim, window.history, s.top_k, key_at);
        window.append_window(selected);
        return selected;
    }
    std::vector<float> gate(s.index_heads);
    w.linear(r.index_gate, input, gate, 1U);
    const auto bias = w.tensor(r.index_bias, s.index_heads);
    for (std::size_t h = 0; h < s.index_heads; ++h) gate[h] += bias[h];
    return gated_index_top_k_reference(query, weights, gate, s.index_dim, window.visible, s.top_k, key_at);
}

// rotary(dim, position, compressed) -> RotaryReferenceRow (this form uses compressed = false)
template <class Weights, class Rotary>
std::vector<float> selected_latent_layer_reference(const Weights& w, const SelectedLatentLayerRoles& r,
    const SelectedLatentLayerShape& s, std::span<const float> input, unsigned position,
    attn::LatentReferenceHistory& state, std::vector<float>& folded_mix, const attn::ReferenceHeadExecutor& execute,
    bool keep_index, bool index_prefill, const Rotary& rotary) {
    const auto& shape = s.latent;
    validate_latent_history(shape, position, state);

    std::vector<float> q_lora(s.q_rank), query(std::size_t{shape.heads} * shape.qk()),
        combined(shape.latent + shape.rotary), inverse(shape.heads), output_heads(std::size_t{shape.heads} * shape.value);
    const auto latent = std::span(combined).first(shape.latent);
    const auto rotary_key = std::span(combined).last(shape.rotary);
    const bool checkpoint = s.checkpoint;
    w.linear(r.q_down, input, q_lora, 1U);
    core::rms_norm_bf16_reference(q_lora, w.tensor(r.q_norm, q_lora.size()), s.eps);
    w.linear(r.q_up, q_lora, query, 1U);
    if (!checkpoint) attn::head_rms_reference(query, shape.qk(), s.eps);
    w.linear(r.kv_down, input, combined, 1U);
    core::rms_norm_bf16_reference(latent, w.tensor(r.kv_norm, latent.size()), s.eps);

    const auto kv_b = w.tensor(r.kv_up, std::size_t{shape.heads} * (shape.nope + shape.value) * shape.latent);
    const auto projections = split_latent_projection(shape, kv_b, false);
    if (checkpoint || !s.key_inverse_rms) std::ranges::fill(inverse, 1.0F);
    else latent_key_inverse_rms_reference(shape, projections, latent, rotary_key, s.eps, inverse, execute);
    const auto rope = rotary(shape.rotary, position, false);
    rope.apply(query, shape.qk());
    rope.apply(rotary_key, shape.rotary);
    if (checkpoint) {
        core::round_bf16_reference(query);
        core::round_bf16_reference(rotary_key);
    }

    std::vector<std::uint8_t> index_entry;
    if (keep_index) {
        std::vector<float> index_key(s.index_dim);
        w.linear(r.index_key, input, index_key, 1U);
        index_layer_norm_reference(index_key, w.tensor(r.index_key_norm_weight, s.index_dim),
            w.tensor(r.index_key_norm_bias, s.index_dim), s.index_eps);
        rotate_index_rows_reference(index_key, s.index_dim, rope, checkpoint, checkpoint);
        index_entry = pack_index_fp4_reference(index_key);
    }
    const auto selected = select_indexed_rows_reference(w, r, s, rope, input, q_lora, position, state.index_keys,
                                                        index_entry, index_prefill);
    const SelectedLatentRow current{latent, rotary_key, inverse};
    const auto row_at = [&](std::size_t row) {
        return row == position ? current : cached_latent_row(shape, row, state);
    };
    selected_latent_attention_reference(shape, projections, query, w.tensor(r.softmax_offset, shape.heads), selected,
                                        row_at, output_heads, execute);
    if (checkpoint) core::round_bf16_reference(output_heads);
    value_head_postmix_reference(w, r.mix, shape.heads, s.vha_rank, s.vha_swapped, checkpoint, output_heads,
                                 folded_mix, execute);
    if (checkpoint) core::round_bf16_reference(output_heads);
    sigmoid_gate_values_reference(w, r.gate, q_lora, output_heads, checkpoint);
    std::vector<float> output(s.hidden);
    w.linear(r.o_proj, output_heads, output, 1U);

    commit_latent_history(state, current, index_entry);
    return output;
}

} // namespace strix::attn_core
