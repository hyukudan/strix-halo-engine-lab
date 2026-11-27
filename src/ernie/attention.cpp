#include "strix/ernie/attention.hpp"

#include "attn_core/compressed_mqa_layer_reference.hpp"
#include "attn_core/selected_latent_layer_reference.hpp"
#include "core/linear_reference.hpp"

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
constexpr auto kIndexPrefillHeads = narrow<unsigned>(a::indexer::prefill_head_topk);
constexpr attn::HybridYarn kYarn{static_cast<double>(a::rope_scaling::original_max_position_embeddings),
    a::rope_scaling::factor, a::rope_scaling::beta_fast, a::rope_scaling::beta_slow};
static_assert(a::rope_interleave && a::qk_norm == "per_head" && a::use_fast_hadamard &&
              a::use_attn_merger && a::add_full_attention_sink_bias && a::mla::split_kv_b_projection);
static_assert(kWindow > 0 && kCompression >= 2 && kIndexDim % 32U == 0 && kTopK > 0);
static_assert(kIndexPrefillHeads > 0 && kIndexPrefillHeads <= kIndexHeads);

attn_core::RotaryReferenceRow rotary_row(unsigned dim, unsigned position, bool compressed) {
    return {dim, position, compressed ? a::compress_rope_theta : static_cast<double>(a::rope_theta),
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

// The checkpoint stores the two post-mix factors exchanged: the tensor named V expands, U contracts.
constexpr attn_core::ValueHeadMixRoles kMixRoles{"vha_postmix_V", "vha_postmix_U"};

constexpr attn_core::CompressedMqaLayerRoles kHcaRoles{
    "linear_q_down_proj.weight", "q_layernorm.weight", "linear_q_up_proj.weight", "linear_kv_proj.weight",
    "kv_layernorm.weight", "core_attention.compressor.linear_wkv.weight",
    "core_attention.compressor.linear_wgate.weight", "core_attention.compressor.ape",
    "core_attention.compressor.norm.weight", "core_attention.attn_sink", "linear_o_group_proj", "o_proj.weight",
    "gate_proj.weight", kMixRoles};

constexpr attn_core::SelectedLatentLayerRoles kMlaRoles{
    "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight", "kv_b_proj.weight", "core_attention.indexer.wk.weight",
    "core_attention.indexer.k_norm.weight", "core_attention.indexer.k_norm.bias",
    "core_attention.indexer.wq_b.weight", "core_attention.indexer.weights_proj.weight",
    "core_attention.indexer.linear_gate.weight", "core_attention.indexer.flashindexer_bias",
    "core_attention.softmax_offset", "o_proj.weight", "gate_proj.weight", kMixRoles};

// The checkpoint folds the value-head mix once per history and rounds the output gate through BF16.
attn_core::FoldForm fold_form(const ReferenceShape& s) {
    return s.contracts.checkpoint_attention ? attn_core::FoldForm::folded : attn_core::FoldForm::residual;
}

core::SigmoidGatePrecision gate_precision(const ReferenceShape& s) {
    return s.contracts.checkpoint_attention ? core::SigmoidGatePrecision::bf16_sigmoid
                                            : core::SigmoidGatePrecision::f32;
}

std::vector<float> hca_reference(const LayerWeights& w, const ReferenceShape& s,
    std::span<const float> input, unsigned position, attn::CompressedReferenceHistory& state,
    std::vector<float>& folded_mix, bool record_history, bool keep_compressor, const AttentionHeadExecutor& execute) {
    const attn_core::CompressedMqaLayerShape shape{s.hidden, s.heads, s.head_dim, s.rope_dim, s.q_rank, s.o_groups,
        s.o_rank, s.vha_rank, kWindow, kEpsilon, s.contracts.checkpoint_attention, fold_form(s), gate_precision(s)};
    return attn_core::compressed_mqa_layer_reference<kCompression>(w, kHcaRoles, shape, input, position, state,
        folded_mix, record_history, keep_compressor, execute,
        [&](unsigned dim, unsigned at, bool compressed) { return rotary_row(dim, at, compressed); });
}

std::vector<float> mla_reference(const LayerWeights& w, const ReferenceShape& s,
    std::span<const float> input, unsigned position, attn::LatentReferenceHistory& state,
    std::vector<float>& folded_mix, const AttentionHeadExecutor& execute, bool keep_index, bool index_prefill) {
    const attn_core::SelectedLatentLayerShape shape{
        {s.heads, narrow<unsigned>(a::mla::kv_lora_rank), narrow<unsigned>(a::mla::qk_nope_head_dim),
         narrow<unsigned>(a::mla::qk_rope_head_dim), narrow<unsigned>(a::mla::v_head_dim)},
        s.hidden, s.q_rank, s.vha_rank, kIndexDim, kIndexHeads, kTopK, kIndexPrefillHeads, kWindow,
        kEpsilon, kEpsilon, s.contracts.checkpoint_attention, fold_form(s), gate_precision(s)};
    return attn_core::selected_latent_layer_reference(w, kMlaRoles, shape, input, position, state, folded_mix,
        execute, keep_index, index_prefill,
        [&](unsigned dim, unsigned at, bool compressed) { return rotary_row(dim, at, compressed); });
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
                        true, history.max_rows_ >= kCompression, execute_heads);
    history.layer_ = layer;
    ++history.position_;
    return result;
}

std::vector<float> attention_reference_first_hca(const std::string& prefix, const ReferenceShape& shape,
    const ReferenceRead& read, std::span<const float> input, const AttentionProject& project) {
    AttentionHistory state;
    return hca_reference({read, project, prefix}, shape, input, 0, state.hca_, state.folded_mix_, false, false, {});
}

} // namespace strix::ernie
