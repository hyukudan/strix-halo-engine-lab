#include "strix/ernie/attention.hpp"

#include "attn_core/compressed_history_reference.hpp"
#include "attn_core/selected_latent_reference.hpp"
#include "attn_core/sparse_index_reference.hpp"
#include "attn_core/value_head_mix_reference.hpp"
#include "core/bf16_reference.hpp"
#include "strix/attn/head_rms_reference.hpp"

namespace strix::ernie {
namespace {

namespace a = data::attention;
using model::narrow;
using model::nearest_float;
constexpr auto kEpsilon = nearest_float(a::rms_norm_eps);
constexpr auto kWindow = narrow<unsigned>(a::sliding_window);
// The model binds one ratio for its compressed layers.
constexpr auto kCompression = narrow<unsigned>(a::compress_ratios[0]);
constexpr auto kIndexDim = narrow<unsigned>(a::indexer::index_head_dim);
constexpr auto kIndexHeads = narrow<unsigned>(a::indexer::index_n_heads);
constexpr auto kTopK = narrow<unsigned>(a::indexer::index_topk);
constexpr auto kOutputFp8Block = narrow<unsigned>(data::weights::checkpoint::weight_block_size[1]);
constexpr auto kPackedIndex = kIndexDim / 2U + kIndexDim / 32U;
constexpr attn::HybridYarn kYarn{static_cast<double>(a::rope_scaling::original_max_position_embeddings),
    a::rope_scaling::factor, a::rope_scaling::beta_fast, a::rope_scaling::beta_slow};
static_assert(a::rope_interleave && a::qk_norm == "per_head" && a::use_fast_hadamard &&
              a::use_attn_merger && a::add_full_attention_sink_bias && a::mla::split_kv_b_projection);
static_assert(kWindow > 0 && kCompression >= 2 && kIndexDim % 32U == 0 && kTopK > 0);

attn_core::RotaryReferenceRow rotary_row(unsigned dim, unsigned position, bool compressed,
                                        const HostContracts& contract) {
    const double base = contract.rope_base > 0 ? contract.rope_base : static_cast<double>(a::rope_theta);
    return {dim, position + contract.position_offset, compressed ? a::compress_rope_theta : base,
            compressed ? &kYarn : nullptr};
}

struct LayerWeights {
    const ReferenceRead& read;
    const AttentionProject& project;
    std::string prefix;

    [[nodiscard]] std::vector<float> tensor(std::string_view role, std::size_t count) const {
        std::vector<float> out(count);
        read(prefix + std::string(role), out, 0);
        return out;
    }
    void linear(std::string_view role, std::span<const float> input, std::span<float> output,
                unsigned groups = 1) const {
        if (project) project(prefix + std::string(role), input, output, groups);
        else core::grouped_linear_reference(tensor(role, input.size() / groups * output.size()), input, output, groups);
    }
};

void postmix(const LayerWeights& w, const ReferenceShape& s, std::span<float> heads, std::vector<float>& folded) {
    if (s.contracts.vha == VhaContract::off) return;
    if (!folded.empty()) {
        attn_core::apply_folded_value_head_mix_reference(heads, s.heads, folded);
        return;
    }
    const auto count = std::size_t{s.heads} * s.vha_rank;
    auto u = w.tensor("vha_postmix_U", count), v = w.tensor("vha_postmix_V", count);
    if (s.contracts.vha == VhaContract::swapped) std::swap(u, v);
    const auto residual = s.contracts.vha != VhaContract::non_residual;
    if (s.contracts.checkpoint_attention) {
        folded = attn_core::fold_value_head_mix_reference(s.heads, s.vha_rank, u, v, residual);
        attn_core::apply_folded_value_head_mix_reference(heads, s.heads, folded);
    } else {
        attn_core::residual_value_head_mix(heads, s.heads, s.vha_rank, u, v, residual);
    }
}

void gate_values(const LayerWeights& w, std::span<const float> query_lora, std::span<float> values,
                 bool checkpoint_attention) {
    std::vector<float> gate(values.size());
    w.linear("gate_proj.weight", query_lora, gate);
    core::sigmoid_product_reference(values, gate, values, checkpoint_attention
        ? core::SigmoidGatePrecision::bf16_sigmoid : core::SigmoidGatePrecision::f32);
}

std::vector<float> hca_reference(const LayerWeights& w, const ReferenceShape& s,
    std::span<const float> input, unsigned position, attn::CompressedReferenceHistory& state,
    std::vector<float>& folded_mix, bool record_history, bool keep_compressor) {
    const auto width = s.head_dim;
    const auto eps = s.contracts.rms_epsilon > 0 ? s.contracts.rms_epsilon : kEpsilon;
    const bool checkpoint = s.contracts.checkpoint_attention;
    const auto packed_block = s.contracts.hca_fp8_epilogue ? kOutputFp8Block : 0U;
    if (packed_block != 0 && (!checkpoint || width % packed_block != 0 || s.o_rank % packed_block != 0))
        throw std::invalid_argument("ERNIE HCA: packed FP8 epilogue requires checkpoint stores and aligned rows");
    const bool separate_values = !checkpoint && s.contracts.key_norm == KeyNormContract::value_before_head_norm;
    const attn_core::CompressedMqaStep step{position, width, kWindow, record_history, keep_compressor,
                                          checkpoint, separate_values, checkpoint};
    attn_core::validate_compressed_mqa_history<kCompression>(step, state);

    std::vector<float> q_lora(s.q_rank), query(std::size_t{s.heads} * width), kv(width);
    w.linear("linear_q_down_proj.weight", input, q_lora);
    core::rms_norm_bf16_reference(q_lora, w.tensor("q_layernorm.weight", q_lora.size()), eps);
    w.linear("linear_q_up_proj.weight", q_lora, query);
    w.linear("linear_kv_proj.weight", input, kv);
    core::rms_norm_bf16_reference(kv, w.tensor("kv_layernorm.weight", kv.size()), eps);
    auto local_value = separate_values ? kv : std::vector<float>{};
    if (checkpoint || s.contracts.key_norm != KeyNormContract::none) attn::head_rms_reference(query, width, eps);
    if (!checkpoint && s.contracts.key_norm != KeyNormContract::weighted_only &&
        s.contracts.key_norm != KeyNormContract::none)
        attn::head_rms_reference(kv, width, eps);

    constexpr std::string_view compressor = "core_attention.compressor.";
    std::vector<float> carry_value, carry_logit, pooled;
    if (keep_compressor) {
        carry_value.resize(width);
        carry_logit.resize(width);
        w.linear(std::string(compressor) + "linear_wkv.weight", input, carry_value);
        w.linear(std::string(compressor) + "linear_wgate.weight", input, carry_logit);
    }
    const auto carry_row = position % kCompression;
    const auto window_rope = rotary_row(s.rope_dim, position, checkpoint, s.contracts);
    const auto compressed_rope = rotary_row(s.rope_dim, position, true, s.contracts);
    if (keep_compressor && carry_row == kCompression - 1U) {
        pooled.resize(width);
        attn_core::projected_ape_pool_reference<kCompression>(state.carry_values, state.carry_logits,
            carry_value, carry_logit, w.tensor(std::string(compressor) + "ape", std::size_t{kCompression} * width),
            pooled);
        core::rms_norm_bf16_reference(pooled, w.tensor(std::string(compressor) + "norm.weight", width), eps);
        rotary_row(s.rope_dim, position + 1U - kCompression, true, s.contracts).apply(pooled, width);
        if (checkpoint) core::round_bf16_reference(pooled);
    }

    const auto sinks = w.tensor("core_attention.attn_sink", s.heads);
    const attn_core::CompressedMqaInputs operands{query, kv, local_value, sinks, carry_value, carry_logit, pooled};
    auto heads = attn_core::compressed_mqa_reference(step, operands, window_rope, compressed_rope, state);
    if (checkpoint) core::round_bf16_reference(heads);
    else if (!s.contracts.no_hca_inverse_rope) window_rope.apply(heads, width, true);
    postmix(w, s, heads, folded_mix);
    if (checkpoint) attn_core::compressed_mqa_output_input_reference(heads, width, window_rope, packed_block);
    std::vector<float> grouped(std::size_t{s.o_groups} * s.o_rank);
    w.linear("linear_o_group_proj", heads, grouped, s.o_groups);
    std::vector<float> output(s.hidden);
    gate_values(w, q_lora, grouped, checkpoint);
    if (packed_block != 0) core::round_block_fp8_activation_reference(grouped, packed_block);
    w.linear("o_proj.weight", grouped, output);
    attn_core::commit_compressed_mqa_history<kCompression>(step, operands, state);
    return output;
}

std::vector<std::uint32_t> select_mla_rows(const LayerWeights& w, const attn_core::RotaryReferenceRow& rope,
    std::span<const float> input, std::span<const float> q_lora, unsigned position,
    std::span<const std::uint8_t> index_history, std::span<const std::uint8_t> current_key,
    bool checkpoint, bool prefill) {
    const attn_core::SparseWindowReference window(position, checkpoint ? kWindow : 0U, kTopK);
    if (window.dense()) return window.dense_rows();
    constexpr std::string_view prefix = "core_attention.indexer.";
    std::vector<float> query(std::size_t{kIndexHeads} * kIndexDim), weights(kIndexHeads);
    w.linear(std::string(prefix) + "wq_b.weight", q_lora, query);
    attn_core::rotate_index_rows_reference(query, kIndexDim, rope, checkpoint, checkpoint);
    if (checkpoint) attn_core::quantize_index_query_reference(query, kIndexDim);

    w.linear(std::string(prefix) + "weights_proj.weight", input, weights);
    const auto key_at = [&](std::size_t row) {
        return row == position ? current_key : index_history.subspan(row * kPackedIndex, kPackedIndex);
    };
    if (checkpoint) {
        const float scale = 1.0F / std::sqrt(static_cast<float>(kIndexDim * kIndexHeads));
        for (auto& weight : weights) weight *= scale;
        if (prefill) {
            const auto gate = w.tensor(std::string(prefix) + "linear_gate.weight", input.size() * kIndexHeads);
            std::vector<float> logits(kIndexHeads);
            core::linear_reference(gate, input, logits);
            const auto bias = w.tensor(std::string(prefix) + "flashindexer_bias", kIndexHeads);
            weights = attn_core::index_head_weights_reference(weights, logits, bias, 8);
        }
        auto selected = attn_core::weighted_index_top_k_reference(query, weights, kIndexDim, window.history, kTopK, key_at);
        window.append_window(selected);
        return selected;
    }
    std::vector<float> gate(kIndexHeads);
    w.linear(std::string(prefix) + "linear_gate.weight", input, gate);
    const auto bias = w.tensor(std::string(prefix) + "flashindexer_bias", kIndexHeads);
    for (std::size_t h = 0; h < kIndexHeads; ++h) gate[h] += bias[h];
    return attn_core::gated_index_top_k_reference(query, weights, gate, kIndexDim, window.visible, kTopK, key_at);
}

std::vector<float> mla_reference(const LayerWeights& w, const ReferenceShape& s,
    std::span<const float> input, unsigned position, attn::LatentReferenceHistory& state,
    std::vector<float>& folded_mix, const AttentionHeadExecutor& execute, bool keep_index, bool index_prefill) {
    const attn_core::SelectedLatentShape shape{s.heads, narrow<unsigned>(a::mla::kv_lora_rank),
        narrow<unsigned>(a::mla::qk_nope_head_dim), narrow<unsigned>(a::mla::qk_rope_head_dim),
        narrow<unsigned>(a::mla::v_head_dim)};
    attn_core::validate_latent_history(shape, position, state);

    std::vector<float> q_lora(s.q_rank), query(std::size_t{shape.heads} * shape.qk()),
        combined(shape.latent + shape.rotary), inverse(shape.heads), output_heads(std::size_t{shape.heads} * shape.value);
    const auto latent = std::span(combined).first(shape.latent);
    const auto rotary_key = std::span(combined).last(shape.rotary);
    const auto eps = s.contracts.rms_epsilon > 0 ? s.contracts.rms_epsilon : kEpsilon;
    const bool checkpoint = s.contracts.checkpoint_attention;
    w.linear("q_a_proj.weight", input, q_lora);
    core::rms_norm_bf16_reference(q_lora, w.tensor("q_a_layernorm.weight", q_lora.size()), eps);
    w.linear("q_b_proj.weight", q_lora, query);
    if (!checkpoint && s.contracts.key_norm != KeyNormContract::none)
        attn::head_rms_reference(query, shape.qk(), eps);
    w.linear("kv_a_proj_with_mqa.weight", input, combined);
    core::rms_norm_bf16_reference(latent, w.tensor("kv_a_layernorm.weight", latent.size()), eps);

    const auto kv_b = w.tensor("kv_b_proj.weight",
        std::size_t{shape.heads} * (shape.nope + shape.value) * shape.latent);
    const auto projections = attn_core::split_latent_projection(shape, kv_b, s.contracts.global_kv_split);
    if (checkpoint || s.contracts.key_norm == KeyNormContract::weighted_only || s.contracts.key_norm == KeyNormContract::none)
        std::ranges::fill(inverse, 1.0F);
    else attn_core::latent_key_inverse_rms_reference(shape, projections, latent, rotary_key, eps, inverse, execute);
    const auto rotary = rotary_row(shape.rotary, position, false, s.contracts);
    rotary.apply(query, shape.qk());
    rotary.apply(rotary_key, shape.rotary);
    if (checkpoint) {
        core::round_bf16_reference(query);
        core::round_bf16_reference(rotary_key);
    }

    std::vector<std::uint8_t> index_entry;
    if (keep_index) {
        std::vector<float> index_key(kIndexDim);
        w.linear("core_attention.indexer.wk.weight", input, index_key);
        attn_core::index_layer_norm_reference(index_key, w.tensor("core_attention.indexer.k_norm.weight", kIndexDim),
            w.tensor("core_attention.indexer.k_norm.bias", kIndexDim), kEpsilon);
        attn_core::rotate_index_rows_reference(index_key, kIndexDim, rotary, checkpoint, checkpoint);
        index_entry = attn_core::pack_index_fp4_reference(index_key);
    }
    const auto selected = select_mla_rows(w, rotary, input, q_lora, position, state.index_keys, index_entry,
                                          checkpoint, index_prefill);
    const attn_core::SelectedLatentRow current{latent, rotary_key, inverse};
    const auto row_at = [&](std::size_t row) {
        return row == position ? current : attn_core::cached_latent_row(shape, row, state);
    };
    attn_core::selected_latent_attention_reference(shape, projections, query,
        w.tensor("core_attention.softmax_offset", shape.heads), selected, row_at, output_heads, execute);
    if (checkpoint) core::round_bf16_reference(output_heads);
    if (s.contracts.gate_before_vha) gate_values(w, q_lora, output_heads, checkpoint);
    postmix(w, s, output_heads, folded_mix);
    if (checkpoint) core::round_bf16_reference(output_heads);
    if (!s.contracts.gate_before_vha) gate_values(w, q_lora, output_heads, checkpoint);
    std::vector<float> output(s.hidden);
    w.linear("o_proj.weight", output_heads, output);

    attn_core::commit_latent_history(state, current, index_entry);
    return output;
}

} // namespace

AttentionHistory::AttentionHistory(std::size_t max_rows) : max_rows_(max_rows) {
    if (max_rows_ == 0 || max_rows_ > data::dims::max_context)
        throw std::invalid_argument("ERNIE attention: invalid sequence bound");
}

std::vector<float> attention_reference(unsigned layer, const ReferenceShape& shape, const ReferenceRead& read,
    std::span<const float> input, AttentionHistory& history, const AttentionProject& project,
    const AttentionHeadExecutor& execute_heads, bool index_prefill) {
    if (layer >= kLayerCount || history.position_ >= history.max_rows_ ||
        (history.position_ != 0 && history.layer_ != layer) || !read || shape.hidden != input.size() ||
        shape.hidden == 0 || shape.q_rank == 0 || shape.heads == 0 || shape.head_dim == 0 ||
        shape.o_groups == 0 || shape.o_rank == 0 || shape.heads % shape.o_groups != 0 ||
        shape.vha_rank == 0 || shape.rope_dim > shape.head_dim || shape.rope_dim % 2U != 0)
        throw std::invalid_argument("ERNIE attention: incompatible row, layer or history");
    const LayerWeights weights{read, project, target_layer_prefix(layer) + "self_attn."};
    // A bounded sequence needs no state for an attention branch it can never reach.
    const auto indexed_prefix_limit = kTopK + (shape.contracts.checkpoint_attention ? kWindow : 0U);
    auto result = layer_is_mla(layer)
        ? mla_reference(weights, shape, input, history.position_, history.mla_, history.folded_mix_, execute_heads,
                        history.max_rows_ > indexed_prefix_limit, index_prefill)
        : hca_reference(weights, shape, input, history.position_, history.hca_, history.folded_mix_,
                        true, history.max_rows_ >= kCompression);
    history.layer_ = layer;
    ++history.position_;
    return result;
}

std::vector<float> attention_reference_first_hca(const std::string& prefix, const ReferenceShape& shape,
    const ReferenceRead& read, std::span<const float> input, const AttentionProject& project) {
    AttentionHistory state;
    return hca_reference({read, project, prefix}, shape, input, 0, state.hca_, state.folded_mix_, false, false);
}

} // namespace strix::ernie
