// Hy4-flash (GGUF architecture "hy_v4_flash") tensor naming map: data only. Names and shapes come from the GGUF directory of the
// public Q4_K_M pack (1416 tensors in two split files); the symbols resolve from the file's own metadata, so another quant build
// of the same architecture checks against the same table. Components feed the planner's byte cut (per layer, per component).
#include "strix/weight/tensor_map.hpp"

#include <string>

namespace strix::weight::tensor_map {
namespace {

using L = Layers;
using S = Sym;
constexpr std::uint64_t Q = kLinearTypes, F = kF32, LK = kLookupTypes;
constexpr Extent X(S a, S b = S::one) { return Extent{a, b}; }
constexpr std::array<Extent, 4> D(Extent a, Extent b = {}, Extent c = {}, Extent d = {}) { return {a, b, c, d}; }

constexpr Entry kHy4Flash[] = {
    // Global: token embedding, head (hyper-connection head mix + norm), Engram lookup table (all four Engram layers, one tensor).
    {"token_embd.weight", L::global, "embedding", D(X(S::hidden), X(S::vocab)), Q | LK},
    {"output.weight", L::global, "head", D(X(S::hidden), X(S::vocab)), Q},
    {"output_norm.weight", L::global, "head.norm", D(X(S::hidden)), F},
    {"output_hc_fn.weight", L::global, "head.hc", D(X(S::hc, S::hidden), X(S::hc)), F},
    {"output_hc_base.weight", L::global, "head.hc", D(X(S::hc)), F},
    {"output_hc_scale.weight", L::global, "head.hc", D(X(S::one)), F},
    {"per_layer_token_embd.weight", L::global, "engram.table", D(X(S::lookup_dim), X(S::lookup_rows)), LK},
    // Every block (including the next-token block): sandwich norms and MLA attention with sinks and output gate.
    {"blk.{L}.attn_norm.weight", L::all, "norm", D(X(S::hidden)), F},
    {"blk.{L}.attn_q_a_norm.weight", L::all, "norm", D(X(S::q_lora)), F},
    {"blk.{L}.attn_kv_a_norm.weight", L::all, "norm", D(X(S::kv_lora)), F},
    {"blk.{L}.ffn_norm.weight", L::all, "norm", D(X(S::hidden)), F},
    {"blk.{L}.post_attention_norm.weight", L::all, "norm", D(X(S::hidden)), F},
    {"blk.{L}.post_ffw_norm.weight", L::all, "norm", D(X(S::hidden)), F},
    {"blk.{L}.attn_sinks.weight", L::all, "attention.sinks", D(X(S::heads)), F},
    {"blk.{L}.attn_q_a.weight", L::all, "attention.q_a", D(X(S::hidden), X(S::q_lora)), Q},
    {"blk.{L}.attn_kv_a_mqa.weight", L::all, "attention.kv_a", D(X(S::hidden), X(S::kv_a)), Q},
    {"blk.{L}.attn_q_b.weight", L::all, "attention.q_b", D(X(S::q_lora), X(S::heads, S::qk_head)), Q},
    {"blk.{L}.attn_k_b.weight", L::all, "attention.k_b", D(X(S::qk_nope), X(S::kv_lora), X(S::heads)), Q},
    {"blk.{L}.attn_v_b.weight", L::all, "attention.v_b", D(X(S::kv_lora), X(S::v_head), X(S::heads)), Q},
    {"blk.{L}.attn_gate.weight", L::all, "attention.gate", D(X(S::hidden), X(S::heads, S::v_head)), Q},
    {"blk.{L}.attn_output.weight", L::all, "attention.o", D(X(S::heads, S::v_head), X(S::hidden)), Q},
    // Hyper-connection pre/post mixes of the main blocks (attention and FFN sublayers).
    {"blk.{L}.hc_attn_fn.weight", L::main, "hyper_connection", D(X(S::hc, S::hidden), X(S::hc_mix)), F},
    {"blk.{L}.hc_attn_base.weight", L::main, "hyper_connection", D(X(S::hc_mix)), F},
    {"blk.{L}.hc_attn_scale.weight", L::main, "hyper_connection", D(X(S::two)), F},
    {"blk.{L}.hc_ffn_fn.weight", L::main, "hyper_connection", D(X(S::hc, S::hidden), X(S::hc_mix)), F},
    {"blk.{L}.hc_ffn_base.weight", L::main, "hyper_connection", D(X(S::hc_mix)), F},
    {"blk.{L}.hc_ffn_scale.weight", L::main, "hyper_connection", D(X(S::two)), F},
    // Sparse-attention indexer with pooled (compressed) keys.
    {"blk.{L}.indexer.k_norm.weight", L::sparse_index, "indexer", D(X(S::index_key)), F},
    {"blk.{L}.indexer_compressor_ape.weight", L::sparse_index, "indexer", D(X(S::index_key), X(S::index_pool)), F},
    {"blk.{L}.indexer_compressor_norm.weight", L::sparse_index, "indexer", D(X(S::index_key)), F},
    {"blk.{L}.indexer.attn_k.weight", L::sparse_index, "indexer", D(X(S::hidden), X(S::index_key)), Q},
    {"blk.{L}.indexer.attn_q_b.weight", L::sparse_index, "indexer", D(X(S::q_lora), X(S::index_heads, S::index_key)), Q},
    {"blk.{L}.indexer.proj.weight", L::sparse_index, "indexer", D(X(S::hidden), X(S::index_heads)), Q},
    {"blk.{L}.indexer_compressor_gate.weight", L::sparse_index, "indexer", D(X(S::index_key), X(S::index_key)), Q},
    {"blk.{L}.indexer_compressor_value.weight", L::sparse_index, "indexer", D(X(S::index_key), X(S::index_key)), Q},
    // Leading dense SwiGLU block(s).
    {"blk.{L}.ffn_gate.weight", L::dense_ffn, "dense_ffn", D(X(S::hidden), X(S::dense_ff)), Q},
    {"blk.{L}.ffn_up.weight", L::dense_ffn, "dense_ffn", D(X(S::hidden), X(S::dense_ff)), Q},
    {"blk.{L}.ffn_down.weight", L::dense_ffn, "dense_ffn", D(X(S::dense_ff), X(S::hidden)), Q},
    // Latent MoE: router (+ selection bias), latent down/up around the routed experts, shared expert.
    {"blk.{L}.ffn_gate_inp.weight", L::moe, "router", D(X(S::hidden), X(S::experts)), Q},
    {"blk.{L}.exp_probs_b.bias", L::moe, "router", D(X(S::experts)), F},
    {"blk.{L}.ffn_routed_down.weight", L::moe, "latent", D(X(S::hidden), X(S::latent)), Q},
    {"blk.{L}.ffn_routed_up.weight", L::moe, "latent", D(X(S::latent), X(S::hidden)), Q},
    {"blk.{L}.ffn_gate_exps.weight", L::moe, "experts.gate", D(X(S::latent), X(S::expert_ff), X(S::experts)), Q},
    {"blk.{L}.ffn_up_exps.weight", L::moe, "experts.up", D(X(S::latent), X(S::expert_ff), X(S::experts)), Q},
    {"blk.{L}.ffn_down_exps.weight", L::moe, "experts.down", D(X(S::expert_ff), X(S::latent), X(S::experts)), Q},
    {"blk.{L}.ffn_gate_shexp.weight", L::moe, "shared_expert", D(X(S::hidden), X(S::shared_ff)), Q},
    {"blk.{L}.ffn_up_shexp.weight", L::moe, "shared_expert", D(X(S::hidden), X(S::shared_ff)), Q},
    {"blk.{L}.ffn_down_shexp.weight", L::moe, "shared_expert", D(X(S::shared_ff), X(S::hidden)), Q},
    // Engram blocks: value + per-stream key projections (one matrix), gates, lookup norms.
    {"blk.{L}.engram_kv.weight", L::engram, "engram.proj", D(X(S::lookup_heads, S::lookup_dim), X(S::hidden, S::hc_plus1)), Q},
    {"blk.{L}.engram_kv_b.weight", L::engram, "engram.proj", D(X(S::hidden), X(S::hc_plus1)), F},
    {"blk.{L}.engram_gate.weight", L::engram, "engram.proj", D(X(S::hidden), X(S::hc)), F},
    {"blk.{L}.engram_hc_scale.weight", L::engram, "engram.proj", D(X(S::hidden), X(S::hc)), F},
    {"blk.{L}.engram_lookup_norm.weight", L::engram, "engram.proj", D(X(S::lookup_dim), X(S::lookup_heads)), F},
    {"blk.{L}.engram_pre_gamma.weight", L::engram, "engram.proj", D(X(S::hidden)), F},
    // Next-token (MTP) block: embedding/hidden fusion.
    {"blk.{L}.nextn.eh_proj.weight", L::nextn, "nextn", D(X(S::hidden, S::two), X(S::hidden)), Q},
    {"blk.{L}.nextn.enorm.weight", L::nextn, "nextn", D(X(S::hidden)), F},
    {"blk.{L}.nextn.hnorm.weight", L::nextn, "nextn", D(X(S::hidden)), F},
    {"blk.{L}.nextn.shared_head_norm.weight", L::nextn, "nextn", D(X(S::hidden)), F},
};

struct Meta {
    const gguf::FileInfo& file;
    std::string prefix;
    std::string& error;
    const gguf::MetadataEntry* get(const char* key) const {
        const auto* entry = file.find_metadata(prefix + key);
        if (entry == nullptr && error.empty()) error = "missing metadata " + prefix + key;
        return entry;
    }
    std::uint64_t u(const char* key) const {
        const auto* entry = get(key);
        if (entry == nullptr) return 0;
        if (entry->unsigned_value) return *entry->unsigned_value;
        if (entry->signed_value && *entry->signed_value > 0) return static_cast<std::uint64_t>(*entry->signed_value);
        if (error.empty()) error = "metadata " + prefix + key + " is not a positive integer";
        return 0;
    }
    // Retained integer or boolean array (gguf::Limits::max_retained_array_elements must cover it).
    std::vector<std::uint64_t> array(const char* key) const {
        const auto* entry = get(key);
        if (entry == nullptr) return {};
        if (!entry->unsigned_array.empty()) return entry->unsigned_array;
        std::vector<std::uint64_t> out;
        for (const auto value : entry->signed_array) {
            if (value < 0) { if (error.empty()) error = "metadata " + prefix + key + " has a negative entry"; return {}; }
            out.push_back(static_cast<std::uint64_t>(value));
        }
        if (out.empty() && error.empty()) error = "metadata array " + prefix + key + " not retained or empty";
        return out;
    }
};

bool hy4_model(const gguf::FileInfo& file, Model& model, std::string& error) {
    error.clear();
    const Meta m{file, "hy_v4_flash.", error};
    model = Model{};
    const auto set = [&](Sym symbol, std::uint64_t value) { model.value[static_cast<std::size_t>(symbol)] = value; };
    const auto blocks = m.u("block_count"), nextn = m.u("nextn_predict_layers"), dense = m.u("leading_dense_block_count");
    const auto rope = m.u("rope.dimension_count"), kv_lora = m.u("attention.kv_lora_rank"), qk = m.u("attention.key_length_mla");
    const auto hc = m.u("hyper_connection.count"), ngram = m.u("ple.ngram_size");
    set(S::hidden, m.u("embedding_length"));
    set(S::vocab, m.u("vocab_size"));
    set(S::q_lora, m.u("attention.q_lora_rank"));
    set(S::kv_lora, kv_lora);
    set(S::kv_a, kv_lora + rope);
    set(S::qk_head, qk);
    set(S::qk_nope, qk > rope ? qk - rope : 0);
    set(S::v_head, m.u("attention.value_length_mla"));
    set(S::dense_ff, m.u("feed_forward_length"));
    set(S::experts, m.u("expert_count"));
    set(S::expert_ff, m.u("expert_feed_forward_length"));
    set(S::latent, m.u("expert_latent_length"));
    set(S::shared_ff, m.u("expert_feed_forward_length") * m.u("expert_shared_count"));
    set(S::index_heads, m.u("attention.indexer.head_count"));
    set(S::index_key, m.u("attention.indexer.key_length"));
    set(S::index_pool, m.u("attention.indexer.kpool"));
    set(S::hc, hc);
    set(S::hc_mix, 2 * hc);
    set(S::hc_plus1, hc + 1);
    set(S::lookup_dim, m.u("embedding_length_per_layer_input"));
    set(S::lookup_heads, ngram > 1 ? (ngram - 1) * m.u("ple.heads_per_ngram") : 0);
    const auto heads = m.array("attention.head_count");
    const auto sparse = m.array("attention.indexer.types");
    const auto engram = m.array("ple.layers");
    const auto offsets = m.array("ple.head_offsets");
    const auto sizes = m.array("ple.head_vocab_sizes");
    if (!error.empty()) return false;
    if (blocks == 0 || blocks > 4096 || nextn >= blocks || dense > blocks) { error = "block counts out of range"; return false; }
    if (heads.size() != blocks) { error = "attention.head_count has " + std::to_string(heads.size()) + " entries for " + std::to_string(blocks) + " blocks"; return false; }
    if (sizes.size() != offsets.size() || sizes.size() != engram.size() * model.value[static_cast<std::size_t>(S::lookup_heads)]) {
        error = "ple head tables do not match ple.layers x lookup heads"; return false;
    }
    std::uint64_t rows = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        // Head tables are contiguous: each Engram layer is one byte range of the lookup table (placement cuts by layer).
        if (offsets[i] != rows || sizes[i] == 0 || sizes[i] > UINT64_MAX - rows) { error = "ple head offsets not contiguous"; return false; }
        rows += sizes[i];
    }
    set(S::lookup_rows, rows);
    model.blocks = static_cast<std::uint32_t>(blocks);
    model.per_layer[static_cast<std::size_t>(S::heads)] = heads;
    model.layer_class.assign(blocks, 0);
    const auto mark = [&](std::uint64_t layer, Layers c) { model.layer_class[layer] |= 1U << static_cast<std::uint32_t>(c); };
    const auto main_blocks = blocks - nextn;
    for (std::uint64_t layer = 0; layer < blocks; ++layer) {
        mark(layer, L::all);
        if (layer < main_blocks) mark(layer, L::main); else mark(layer, L::nextn);
        if (layer < dense) mark(layer, L::dense_ffn); else mark(layer, L::moe);
        // indexer.types covers the main blocks; the next-token block is a sparse-attention block (it carries the indexer).
        if (layer < sparse.size() ? sparse[layer] != 0 : layer >= main_blocks) mark(layer, L::sparse_index);
    }
    for (const auto layer : engram) {
        if (layer >= main_blocks) { error = "ple.layers entry outside the main blocks"; return false; }
        mark(layer, L::engram);
    }
    return true;
}

}  // namespace

const FamilyMap& hy4_flash_map() {
    // Retained arrays: head_count / indexer.types / swa pattern (blocks), ple head tables (64); token_map stays a summary.
    static const FamilyMap map{"hy_v4_flash", kHy4Flash, &hy4_model, 4096};
    return map;
}

}  // namespace strix::weight::tensor_map
