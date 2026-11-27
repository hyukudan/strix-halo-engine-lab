#pragma once

// One compressed-history multi-query attention layer (windowed local rows plus pooled compressed
// rows, grouped output projection, value-head post-mix, sigmoid output gate), composed from the
// shared primitives. The model declares the roles (checkpoint tensor names), the dimensions and the
// rotary rows; `Ratio` is the compression ratio of the layer.

#include "attn_core/compressed_history_reference.hpp"
#include "attn_core/layer_reference_common.hpp"
#include "core/bf16_reference.hpp"
#include "core/linear_reference.hpp"
#include "strix/attn/head_rms_reference.hpp"

#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::attn_core {

struct CompressedMqaLayerRoles {
    std::string_view q_down, q_norm, q_up, kv_proj, kv_norm;
    std::string_view compressor_value, compressor_gate, compressor_ape, compressor_norm;
    std::string_view sink, o_group, o_proj, gate;
    ValueHeadMixRoles mix;
};

struct CompressedMqaLayerShape {
    unsigned hidden, heads, head_dim, rope_dim, q_rank, o_groups, o_rank, vha_rank, window;
    float eps;
    StorePrecision store;      // BF16 rounding of the pooled row and the attention heads
    HeadNorm key_norm;         // head RMS of the key/value row after its projection norm
    bool shared_rotary_domain; // window rows use the compressed rotary domain and share it with pooled rows
    OutputRope output_rope;    // where the inverse rotary of the output heads is applied
    FoldForm fold;
    core::SigmoidGatePrecision gate_precision;
    attn::ReferenceOrder order = attn::ReferenceOrder::sequential; // arithmetic order of the norms and the head reduction
};

// rotary(dim, position, compressed) -> RotaryReferenceRow
template <unsigned Ratio, class Weights, class Rotary>
std::vector<float> compressed_mqa_layer_reference(const Weights& w, const CompressedMqaLayerRoles& r,
    const CompressedMqaLayerShape& s, std::span<const float> input, unsigned position,
    attn::CompressedReferenceHistory& state, std::vector<float>& folded_mix, bool record_history,
    bool keep_compressor, const attn::ReferenceHeadExecutor& execute, const Rotary& rotary) {
    const auto width = s.head_dim;
    const bool bf16 = s.store == StorePrecision::bf16;
    const CompressedMqaStep step{position, width, s.window, record_history, keep_compressor,
                                 bf16, false, s.shared_rotary_domain};
    validate_compressed_mqa_history<Ratio>(step, state);

    std::vector<float> q_lora(s.q_rank), query(std::size_t{s.heads} * width), kv(width);
    w.linear(r.q_down, input, q_lora, 1U);
    order_rms_norm_bf16_reference(s.order, q_lora, w.tensor(r.q_norm, q_lora.size()), s.eps);
    w.linear(r.q_up, q_lora, query, 1U);
    w.linear(r.kv_proj, input, kv, 1U);
    order_rms_norm_bf16_reference(s.order, kv, w.tensor(r.kv_norm, kv.size()), s.eps);
    order_head_rms_reference(s.order, query, width, s.eps);
    if (s.key_norm == HeadNorm::head_rms) order_head_rms_reference(s.order, kv, width, s.eps);

    std::vector<float> carry_value, carry_logit, pooled;
    if (keep_compressor) {
        carry_value.resize(width);
        carry_logit.resize(width);
        w.linear(r.compressor_value, input, carry_value, 1U);
        w.linear(r.compressor_gate, input, carry_logit, 1U);
    }
    const auto carry_row = position % Ratio;
    const auto window_rope = rotary(s.rope_dim, position, s.shared_rotary_domain);
    const auto compressed_rope = rotary(s.rope_dim, position, true);
    if (keep_compressor && carry_row == Ratio - 1U) {
        pooled.resize(width);
        projected_ape_pool_reference<Ratio>(state.carry_values, state.carry_logits, carry_value, carry_logit,
            w.tensor(r.compressor_ape, std::size_t{Ratio} * width), pooled, s.order);
        order_rms_norm_bf16_reference(s.order, pooled, w.tensor(r.compressor_norm, width), s.eps);
        rotary(s.rope_dim, position + 1U - Ratio, true).apply(pooled, width);
        if (bf16) core::round_bf16_reference(pooled);
    }

    const auto sinks = w.tensor(r.sink, s.heads);
    const CompressedMqaInputs operands{query, kv, {}, sinks, carry_value, carry_logit, pooled};
    auto heads = compressed_mqa_reference(step, operands, window_rope, compressed_rope, state, execute, s.order);
    if (bf16) core::round_bf16_reference(heads);
    if (s.output_rope == OutputRope::before_mix) window_rope.apply(heads, width, true);
    value_head_postmix_reference(w, r.mix, s.heads, s.vha_rank, s.fold, heads, folded_mix, execute);
    if (s.output_rope == OutputRope::after_mix) compressed_mqa_output_input_reference(heads, width, window_rope, 0U);
    std::vector<float> grouped(std::size_t{s.o_groups} * s.o_rank);
    w.linear(r.o_group, heads, grouped, s.o_groups);
    std::vector<float> output(s.hidden);
    sigmoid_gate_values_reference(w, r.gate, q_lora, grouped, s.gate_precision, s.order);
    w.linear(r.o_proj, grouped, output, 1U);
    commit_compressed_mqa_history<Ratio>(step, operands, state);
    return output;
}

} // namespace strix::attn_core
