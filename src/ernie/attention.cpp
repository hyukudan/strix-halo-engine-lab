#include "strix/ernie/attention.hpp"

#include "attn_core/hybrid_reference.hpp"
#include "attn_core/selected_latent_reference.hpp"
#include "attn_core/sparse_index_reference.hpp"
#include "attn_core/value_head_mix_reference.hpp"
#include "core/bf16_reference.hpp"
#include "core/hadamard_reference.hpp"
#include "strix/attn/head_rms_reference.hpp"
#include "strix/attn/hybrid_rope.hpp"

#include <numeric>

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
constexpr auto kPackedIndex = kIndexDim / 2U + kIndexDim / 32U;
constexpr attn::HybridYarn kYarn{static_cast<double>(a::rope_scaling::original_max_position_embeddings),
    a::rope_scaling::factor, a::rope_scaling::beta_fast, a::rope_scaling::beta_slow};
static_assert(a::rope_interleave && a::qk_norm == "per_head" && a::use_fast_hadamard &&
              a::use_attn_merger && a::add_full_attention_sink_bias && a::mla::split_kv_b_projection);
static_assert(kWindow > 0 && kCompression >= 2 && kIndexDim % 32U == 0 && kTopK > 0);

struct RotaryRow {
    std::vector<float> c, s;
    RotaryRow(unsigned dim, unsigned position, bool compressed, const HostContracts& contract)
        : c(dim / 2U), s(dim / 2U) {
        const double base = contract.rope_base > 0 ? contract.rope_base : static_cast<double>(a::rope_theta);
        attn::hybrid_rope_row(position + contract.position_offset, compressed ? a::compress_rope_theta : base,
                             compressed ? &kYarn : nullptr, c, s);
    }
    void apply(std::span<float> rows, unsigned width, bool inverse = false) const {
        attn::hybrid_rope_tail(rows, width, c, s, inverse);
    }
    void apply_index(std::span<float> rows, bool leading) const {
        if (!leading) {
            apply(rows, kIndexDim);
            return;
        }
        const auto dim = static_cast<unsigned>(2U * c.size());
        if (dim == 0) return;
        for (std::size_t offset = 0; offset < rows.size(); offset += kIndexDim)
            apply(rows.subspan(offset, dim), dim);
    }
};

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

template<class T>
void reserve_append(std::vector<T>& values, std::size_t elements) {
    const auto required = values.size() + elements;
    if (required > values.capacity())
        values.reserve(std::max(required, values.capacity() + values.capacity() / 2U));
}

std::vector<float> hca_reference(const LayerWeights& w, const ReferenceShape& s,
    std::span<const float> input, unsigned position, auto& state, bool record_history, bool keep_compressor) {
    const auto width = s.head_dim;
    const auto ring_size = std::size_t{kWindow} * width;
    const auto carry_size = std::size_t{kCompression - 1U} * width;
    const auto eps = s.contracts.rms_epsilon > 0 ? s.contracts.rms_epsilon : kEpsilon;
    const bool checkpoint = s.contracts.checkpoint_attention;
    const bool separate_values = !checkpoint && s.contracts.key_norm == KeyNormContract::value_before_head_norm;
    if (position != 0 && (state.window.size() != ring_size ||
        (keep_compressor && state.carry_values.size() != carry_size)))
        throw std::invalid_argument("ERNIE HCA: history geometry changed");

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
    const RotaryRow window_rope(s.rope_dim, position, checkpoint, s.contracts),
                    compressed_rope(s.rope_dim, position, true, s.contracts);
    if (keep_compressor && carry_row == kCompression - 1U) {
        pooled.resize(width);
        attn_core::projected_ape_pool_reference<kCompression>(state.carry_values, state.carry_logits,
            carry_value, carry_logit, w.tensor(std::string(compressor) + "ape", std::size_t{kCompression} * width),
            pooled);
        core::rms_norm_bf16_reference(pooled, w.tensor(std::string(compressor) + "norm.weight", width), eps);
        RotaryRow(s.rope_dim, position + 1U - kCompression, true, s.contracts).apply(pooled, width);
    }

    auto compressed_query = query;
    window_rope.apply(query, width);
    window_rope.apply(kv, width);
    if (separate_values) window_rope.apply(local_value, width);
    compressed_rope.apply(compressed_query, width);
    if (checkpoint) {
        core::round_bf16_reference(query);
        core::round_bf16_reference(kv);
        core::round_bf16_reference(compressed_query);
        core::round_bf16_reference(pooled);
    }
    const auto window_start = attn_core::mask::window_lo(position, kWindow);
    const auto window_at = [&](std::size_t row) -> std::span<const float> {
        const auto token = window_start + row;
        if (token == position) return kv;
        return std::span(state.window).subspan((token % kWindow) * width, width);
    };
    const auto window_value = [&](std::size_t row) -> std::span<const float> {
        if (!separate_values) return window_at(row);
        const auto token = window_start + row;
        if (token == position) return local_value;
        return std::span(state.window_values).subspan((token % kWindow) * width, width);
    };
    // The just-completed group and the current window may overlap. Both are
    // visible, with raw-window keys first in the reduction.
    const auto compressed_count = (state.compressed.size() + pooled.size()) / width;
    const auto pooled_key = [&](std::size_t row) -> std::span<const float> {
        if (row == state.compressed.size() / width) return pooled;
        return std::span(state.compressed).subspan(row * width, width);
    };
    std::vector<float> compressed_values;
    if (!checkpoint) {
        compressed_values = state.compressed;
        compressed_values.insert(compressed_values.end(), pooled.begin(), pooled.end());
        compressed_rope.apply(compressed_values, width, true);
        window_rope.apply(compressed_values, width);
    }
    const auto pooled_value = [&](std::size_t row) {
        if (checkpoint) return pooled_key(row);
        return std::span<const float>(compressed_values).subspan(row * width, width);
    };
    std::vector<float> heads(query.size());
    const auto sinks = w.tensor("core_attention.attn_sink", s.heads);
    if (record_history) {
        attn_core::hybrid_mqa_reference({width, query, compressed_query, sinks},
            position - window_start + 1U, compressed_count, window_at, window_value, pooled_key, pooled_value, heads);
    } else {
        std::vector<float> keys(width), values(width);
        attn_core::causal_gqa_reference({s.heads, 1, width}, {0, query, kv, window_value(0), 0, {}, sinks},
                                       {keys, values}, heads);
    }
    if (checkpoint) core::round_bf16_reference(heads);
    else if (!s.contracts.no_hca_inverse_rope) window_rope.apply(heads, width, true);
    postmix(w, s, heads, state.folded_mix);
    if (checkpoint) {
        window_rope.apply(heads, width, true);
        core::round_bf16_reference(heads);
    }
    std::vector<float> grouped(std::size_t{s.o_groups} * s.o_rank);
    w.linear("linear_o_group_proj", heads, grouped, s.o_groups);
    std::vector<float> output(s.hidden);
    gate_values(w, q_lora, grouped, checkpoint);
    w.linear("o_proj.weight", grouped, output);
    if (!record_history) return output;

    // Prepare allocation before touching any live row. An interrupted read or
    // projection leaves the visible prefix intact.
    if (position == 0) {
        state.window.reserve(ring_size);
        state.window.resize(ring_size);
        if (separate_values) state.window_values.resize(ring_size);
        if (keep_compressor) {
            state.carry_values.resize(carry_size);
            state.carry_logits.resize(carry_size);
        }
    }
    reserve_append(state.compressed, pooled.size());
    std::ranges::copy(kv, std::span(state.window).subspan(std::size_t{position % kWindow} * width, width).begin());
    if (separate_values)
        std::ranges::copy(local_value, std::span(state.window_values).subspan(std::size_t{position % kWindow} * width, width).begin());
    if (keep_compressor) {
        if (pooled.empty()) {
            std::ranges::copy(carry_value, std::span(state.carry_values).subspan(std::size_t{carry_row} * width, width).begin());
            std::ranges::copy(carry_logit, std::span(state.carry_logits).subspan(std::size_t{carry_row} * width, width).begin());
        } else {
            state.compressed.insert(state.compressed.end(), pooled.begin(), pooled.end());
        }
    }
    return output;
}

void index_key_norm(std::span<float> key, std::span<const float> weight, std::span<const float> bias) {
    const float width = static_cast<float>(key.size());
    const float mean = std::accumulate(key.begin(), key.end(), 0.0F) / width;
    float square = 0.0F;
    for (const float value : key) square = std::fma(value - mean, value - mean, square);
    const float inv = 1.0F / std::sqrt(square / width + kEpsilon);
    for (std::size_t i = 0; i < key.size(); ++i) key[i] = ((key[i] - mean) * inv) * weight[i] + bias[i];
    core::round_bf16_reference(key);
}

std::vector<std::uint32_t> select_mla_rows(const LayerWeights& w, const RotaryRow& rope,
    std::span<const float> input, std::span<const float> q_lora, unsigned position,
    std::span<const std::uint8_t> index_history, std::span<const std::uint8_t> current_key,
    bool checkpoint, bool prefill) {
    const std::size_t visible = std::size_t{position} + 1U;
    const auto candidates = checkpoint ? visible - std::min<std::size_t>(visible, kWindow) : visible;
    if (candidates <= kTopK) {
        std::vector<std::uint32_t> dense(visible);
        std::iota(dense.begin(), dense.end(), 0U);
        return dense;
    }
    constexpr std::string_view prefix = "core_attention.indexer.";
    std::vector<float> query(std::size_t{kIndexHeads} * kIndexDim), weights(kIndexHeads);
    w.linear(std::string(prefix) + "wq_b.weight", q_lora, query);
    rope.apply_index(query, checkpoint);
    for (std::size_t h = 0; h < kIndexHeads; ++h) {
        auto row = std::span(query).subspan(h * kIndexDim, kIndexDim);
        if (checkpoint) core::round_bf16_reference(row);
        core::hadamard_reference(row);
        if (checkpoint) {
            core::round_bf16_reference(row);
            const auto packed = attn_core::pack_index_fp4_reference(row);
            attn_core::unpack_index_fp4_reference(packed, row);
        }
    }

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
        auto selected = attn_core::weighted_index_top_k_reference(query, weights, kIndexDim, candidates, kTopK, key_at);
        for (auto row = candidates; row < visible; ++row) selected.push_back(static_cast<std::uint32_t>(row));
        return selected;
    }
    std::vector<float> gate(kIndexHeads);
    w.linear(std::string(prefix) + "linear_gate.weight", input, gate);
    const auto bias = w.tensor(std::string(prefix) + "flashindexer_bias", kIndexHeads);
    for (std::size_t h = 0; h < kIndexHeads; ++h) gate[h] += bias[h];
    return attn_core::gated_index_top_k_reference(query, weights, gate, kIndexDim, visible, kTopK, key_at);
}

std::vector<float> mla_reference(const LayerWeights& w, const ReferenceShape& s,
    std::span<const float> input, unsigned position, auto& state, const AttentionHeadExecutor& execute,
    bool keep_index, bool index_prefill) {
    const attn_core::SelectedLatentShape shape{s.heads, narrow<unsigned>(a::mla::kv_lora_rank),
        narrow<unsigned>(a::mla::qk_nope_head_dim), narrow<unsigned>(a::mla::qk_rope_head_dim),
        narrow<unsigned>(a::mla::v_head_dim)};
    if (state.inverse_key_rms.size() != std::size_t{position} * shape.heads)
        throw std::invalid_argument("ERNIE MLA: history geometry changed");

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
    const RotaryRow rotary(shape.rotary, position, false, s.contracts);
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
        index_key_norm(index_key, w.tensor("core_attention.indexer.k_norm.weight", kIndexDim),
                                  w.tensor("core_attention.indexer.k_norm.bias", kIndexDim));
        rotary.apply_index(index_key, checkpoint);
        if (checkpoint) core::round_bf16_reference(index_key);
        core::hadamard_reference(index_key);
        if (checkpoint) core::round_bf16_reference(index_key);
        index_entry = attn_core::pack_index_fp4_reference(index_key);
    }
    const auto selected = select_mla_rows(w, rotary, input, q_lora, position, state.index_keys, index_entry,
                                          checkpoint, index_prefill);
    const auto row_at = [&](std::size_t row) -> attn_core::SelectedLatentRow {
        if (row == position) return {latent, rotary_key, inverse};
        return {std::span(state.latents).subspan(row * shape.latent, shape.latent),
                std::span(state.rotary_keys).subspan(row * shape.rotary, shape.rotary),
                std::span(state.inverse_key_rms).subspan(row * shape.heads, shape.heads)};
    };
    attn_core::selected_latent_attention_reference(shape, projections, query,
        w.tensor("core_attention.softmax_offset", shape.heads), selected, row_at, output_heads, execute);
    if (checkpoint) core::round_bf16_reference(output_heads);
    if (s.contracts.gate_before_vha) gate_values(w, q_lora, output_heads, checkpoint);
    postmix(w, s, output_heads, state.folded_mix);
    if (checkpoint) core::round_bf16_reference(output_heads);
    if (!s.contracts.gate_before_vha) gate_values(w, q_lora, output_heads, checkpoint);
    std::vector<float> output(s.hidden);
    w.linear("o_proj.weight", output_heads, output);

    reserve_append(state.latents, latent.size());
    reserve_append(state.rotary_keys, rotary_key.size());
    reserve_append(state.inverse_key_rms, inverse.size());
    reserve_append(state.index_keys, index_entry.size());
    state.latents.insert(state.latents.end(), latent.begin(), latent.end());
    state.rotary_keys.insert(state.rotary_keys.end(), rotary_key.begin(), rotary_key.end());
    state.inverse_key_rms.insert(state.inverse_key_rms.end(), inverse.begin(), inverse.end());
    state.index_keys.insert(state.index_keys.end(), index_entry.begin(), index_entry.end());
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
        ? mla_reference(weights, shape, input, history.position_, history.mla_, execute_heads,
                        history.max_rows_ > indexed_prefix_limit, index_prefill)
        : hca_reference(weights, shape, input, history.position_, history.hca_, true, history.max_rows_ >= kCompression);
    history.layer_ = layer;
    ++history.position_;
    return result;
}

std::vector<float> attention_reference_first_hca(const std::string& prefix, const ReferenceShape& shape,
    const ReferenceRead& read, std::span<const float> input, const AttentionProject& project) {
    AttentionHistory state;
    return hca_reference({read, project, prefix}, shape, input, 0, state.hca_, false, false);
}

} // namespace strix::ernie
