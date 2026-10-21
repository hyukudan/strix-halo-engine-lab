#include "strix/deepseek41_attention_host.hpp"

#include "strix/ds41_layer.hpp"

#include <stdexcept>

namespace strix::deepseek41 {

namespace attn = ds41::attn;

struct AttentionHost::Layer {
    std::vector<std::uint8_t> ring;               // ring_capacity x 528 B
    attn::CompressState carry;                    // KV sources: ratio 2's pending group
    std::vector<std::uint8_t> kv_cache, idx_cache;  // KV sources: entries at position / ratio
    std::vector<std::uint32_t> topk;              // index sources: the last selection
};

namespace {

std::vector<float> f16_linear(std::span<const std::uint8_t> w, std::span<const float> x, std::size_t in, std::size_t out) {
    std::vector<float> y(out);
    ds41::f16_rows(reinterpret_cast<const std::uint16_t*>(w.data()), in, 0, out, x.data(), y.data());
    return y;
}

std::span<const float> f32(std::span<const std::uint8_t> bytes) {
    return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / 4U};
}

std::span<const std::uint16_t> f16(std::span<const std::uint8_t> bytes) {
    return {reinterpret_cast<const std::uint16_t*>(bytes.data()), bytes.size() / 2U};
}

}  // namespace

AttentionHost::AttentionHost(const TargetModel& model, std::uint32_t layers, std::uint32_t max_positions, TensorBytes tensor)
    : model_(model), tensor_(std::move(tensor)), roles_(attn::released_plan()) {
    const auto& c = model.config;
    if (c.head_dim != attn::kHeadDim || c.num_attention_heads != attn::kHeads || c.qk_rope_head_dim != attn::kRopeDim ||
        c.sliding_window != attn::kWindow || c.index_n_heads != attn::kIdxHeads || c.index_head_dim != attn::kIdxDim ||
        c.index_topk != attn::kTopK || c.candidate_topk_blocks != attn::kCandBlocks || c.candidate_block_size != attn::kCandBlockSize ||
        model.layers.size() != roles_.size())
        throw std::invalid_argument("deepseek41 attention: model geometry differs from the attention piece's");
    std::uint32_t source_ratio = 0;
    for (std::uint32_t l = 0; l < layers; ++l) {
        const auto& a = model.layers[l];
        const auto& r = roles_[l];
        if (a.compress_ratio != r.ratio || a.kv_source != r.owns_kv || a.index_source != r.owns_index ||
            a.candidate_source != r.candidate_source || a.uses_candidates != r.candidate_user ||
            (r.ratio != 0U && (a.kv_source_layer != r.kv_source || a.index_source_layer != r.index_source)))
            throw std::invalid_argument("deepseek41 attention: layer " + std::to_string(l) + " roles differ from the attention piece's plan");
        auto L = std::make_unique<Layer>();
        L->ring.assign(std::size_t{ring_capacity} * attn::kWindowEntryBytes, 0U);
        if (r.owns_kv) {
            const std::size_t entries = max_positions / r.ratio + 1U;
            L->kv_cache.assign(entries * attn::kKvEntryBytes, 0U);
            L->idx_cache.assign(entries * attn::kIdxEntryBytes, 0U);
        }
        if (r.candidate_source) source_ratio = r.ratio;
        layers_.push_back(std::move(L));
    }
    if (source_ratio != 0U) candidates_.assign((max_positions / source_ratio + attn::kCandBlockSize) / attn::kCandBlockSize, 0U);
}

AttentionHost::~AttentionHost() = default;

const std::vector<std::uint32_t>& AttentionHost::topk(std::uint32_t layer) const { return layers_.at(layer)->topk; }

void AttentionHost::window_write(std::uint32_t layer, std::span<const float> kv, std::uint32_t position) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    const std::uint64_t pos = position;
    attn::WindowArgs a{kv, f32(tensor_(p + "attn_kv_a_norm.weight")), {&pos, 1}, ring_capacity, roles_[layer].rope};
    attn::window_write(a, layers_.at(layer)->ring);
}

void AttentionHost::compress(std::uint32_t layer, std::span<const float> x, std::uint32_t position) {
    const auto& c = model_.config;
    const auto& r = roles_[layer];
    auto& L = *layers_.at(layer);
    const std::string p = "blk." + std::to_string(layer) + ".";
    const std::uint64_t pos = position;
    const auto kv = f16_linear(tensor_(p + "attn_compressor_kv.weight"), x, c.hidden_size, c.head_dim);
    std::vector<float> score;
    if (r.ratio == 2U) score = f16_linear(tensor_(p + "attn_compressor_gate.weight"), x, c.hidden_size, c.head_dim);
    attn::CompressArgs a{r.ratio, kv, score, f32(tensor_(p + "attn_compressor_norm.weight")), f16(tensor_(p + "indexer.attn_k.weight")),
                         f32(tensor_(p + "indexer.k_norm.weight")), {&pos, 1}};
    attn::compress(a, L.carry, L.kv_cache, L.idx_cache);
}

void AttentionHost::index(std::uint32_t layer, std::span<const float> x, std::span<const float> qr, std::uint32_t position) {
    const auto& c = model_.config;
    const auto& r = roles_[layer];
    const auto& source = *layers_.at(static_cast<std::size_t>(r.kv_source));
    const std::string p = "blk." + std::to_string(layer) + ".";
    const std::uint64_t pos = position;
    const auto q = f16_linear(tensor_(p + "indexer.attn_q_b.weight"), qr, c.q_lora_rank, std::size_t{attn::kIdxHeads} * attn::kIdxDim);
    const auto w = f16_linear(tensor_(p + "indexer.proj.weight"), x, c.hidden_size, attn::kIdxHeads);
    attn::IndexArgs a{q, w, {&pos, 1}, roles_[static_cast<std::size_t>(r.kv_source)].ratio, r.rope};
    std::vector<std::vector<float>> scores;
    attn::index_scores(a, source.idx_cache, scores);
    const auto pool = r.candidate_source ? attn::Pool::candidate_source : r.candidate_user ? attn::Pool::candidate_user : attn::Pool::plain;
    layers_.at(layer)->topk = attn::select_topk(scores[0], pool, candidates_);
}

void AttentionHost::attention(std::uint32_t layer, std::span<const float> q, std::uint32_t position, std::span<float> o) const {
    const std::string p = "blk." + std::to_string(layer) + ".";
    const auto& r = roles_[layer];
    const std::uint64_t pos = position;
    std::vector<std::uint32_t> idx(attn::kTopK, 0U);
    std::uint32_t count = 0;
    std::span<const std::uint8_t> kv_cache;
    if (r.ratio != 0U) {
        const auto& sel = layers_.at(static_cast<std::size_t>(r.index_source))->topk;
        std::copy(sel.begin(), sel.end(), idx.begin());
        count = static_cast<std::uint32_t>(sel.size());
        kv_cache = layers_.at(static_cast<std::size_t>(r.kv_source))->kv_cache;
    }
    attn::AttentionArgs a{q, f32(tensor_(p + "attn_sinks.weight")), {&pos, 1}, ring_capacity, r.rope, idx, {&count, 1}, attn::kTopK};
    attn::attention(a, layers_.at(layer)->ring, kv_cache, o);
}

}  // namespace strix::deepseek41
