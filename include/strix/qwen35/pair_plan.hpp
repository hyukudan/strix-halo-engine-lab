#pragma once
// qwen35 (Qwen3.8-27B dense) binding of the shared layer-chain planner (strix/pair/chain.hpp). The model supplies
// only its per-layer table (weight bytes from the GGUF tensor sizes, prefix state from its config, decode reads);
// every placement decision (ranks used, contiguous layer ranges, block, decoder, memory admission) is the shared
// planner's, over any number of ranks declared on a line (rank i links to rank i+1).
// Home: specific (model table: qwen35 GDN/attention interleave and state layout, src/qwen35/hip/engine.hip:419-420);
// placement is shared (include/strix/pair/chain.hpp).
#include <cstdint>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "strix/pair/chain_topology.hpp"
#include "strix/pair/placement_inputs.hpp"
#include "strix/qwen35/config.hpp"

namespace strix::qwen35::pair_plan {
namespace chain = strix::pair::chain;

// Per-layer table of the model: weights (bytes resident), prefix state per row (KV), fixed state (GDN matrix + conv).
struct LayerTable {
    std::vector<double> weight_bytes;   // per layer, from the tensors named blk.<l>.*
    double embed_bytes = 0.0;           // token_embd.*
    double head_bytes = 0.0;            // output.*, output_norm.* (read per decode token)
};

// Sums tensor bytes by owner: "blk.<l>." -> layer l (< layers; MTP blocks at l >= layers are not trunk layers and are
// left out), "token_embd" -> embedding, "output" / "output_norm" -> head. Any range of {name, byte_size} records.
template<class Tensors> LayerTable layer_table(const Tensors& tensors, std::uint32_t layers) {
    LayerTable t;
    t.weight_bytes.assign(layers, 0.0);
    for (const auto& tensor : tensors) {
        const std::string_view name = tensor.name;
        const auto bytes = static_cast<double>(tensor.byte_size);
        if (const auto layer = strix::pair::place::block_of(name); layer >= 0) {   // shared GGUF owner rule
            if (layer < static_cast<std::int64_t>(layers)) t.weight_bytes[static_cast<std::size_t>(layer)] += bytes;
        } else if (name.starts_with("token_embd")) {
            t.embed_bytes += bytes;
        } else if (name.starts_with("output")) {
            t.head_bytes += bytes;
        }
    }
    return t;
}

// Exact engine state layout (src/qwen35/hip/engine.hip:419-420): GDN matrix FP32 [value heads][key dim][value dim],
// conv history BF16 [channels][kernel]; BF16 K and V cache rows per full-attention layer.
inline double gdn_fixed_bytes(const Config& c) {
    return double(c.gdn_value_heads) * c.gdn_key_dim * c.gdn_value_dim * 4.0 + double(c.gdn_conv_channels()) * c.gdn_conv_kernel * 2.0;
}
inline double kv_bytes_per_row(const Config& c) { return 2.0 * c.kv_width() * 2.0; }
// Seam: the FP32 residual stream x [rows][hidden] (engine.hip run(): embed_kernel writes x, every layer updates it).
inline double seam_bytes_per_row(const Config& c) { return double(c.hidden) * 4.0; }

// Model shape for the planner. `context_rows`: the KV rows each decode token reads (decode prediction only).
inline chain::ModelShape shape(const Config& c, const LayerTable& t, std::uint64_t context_rows) {
    if (t.weight_bytes.size() != c.layers) throw std::invalid_argument("layer table does not match the config");
    chain::ModelShape m;
    m.layers = c.layers;
    m.seam_bytes_per_row = seam_bytes_per_row(c);
    m.state_bytes_per_row.assign(c.layers, 0.0);
    m.state_fixed_bytes.assign(c.layers, 0.0);
    m.read_bytes_per_token.assign(c.layers, 0.0);
    m.weight_bytes = t.weight_bytes;
    m.embed_bytes = t.embed_bytes;
    m.head_bytes = t.head_bytes;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const bool attention = c.kind(l) == LayerKind::full_attention;
        m.state_bytes_per_row[l] = attention ? kv_bytes_per_row(c) : 0.0;
        m.state_fixed_bytes[l] = attention ? 0.0 : gdn_fixed_bytes(c);
        // A decode token reads the layer's weights, its KV rows (attention) or reads + writes its GDN state.
        m.read_bytes_per_token[l] = t.weight_bytes[l] +
            (attention ? kv_bytes_per_row(c) * static_cast<double>(context_rows) : 2.0 * gdn_fixed_bytes(c));
    }
    return m;
}

// Relative per-layer prefill cost at the wide window, from the one-node rocprofv3 receipt (2048 tokens, wide 512):
// GDN layer = dense GEMMs 13.2 + recurrence 15.8 + rest ~2 ms; attention layer = GEMMs 13.2 + flash/prep ~3.3 ms per
// window. A search seed for the shape only; the rate comes from the rank's declaration or its measured receipt.
inline std::vector<double> prefill_shape(const Config& c) {
    std::vector<double> s(c.layers);
    for (std::uint32_t l = 0; l < c.layers; ++l) s[l] = c.kind(l) == LayerKind::full_attention ? 16.5 : 31.0;
    return s;
}

// Ranks, hops, override and the receipt line are the shared topology binding (strix/pair/chain_topology.hpp);
// this binding adds only the model's relative per-layer cost shape.
using Rank = chain::RankDecl;
using Hop = chain::HopDecl;
using chain::apply_override;
using chain::describe;
using chain::links;
inline std::vector<chain::Device> devices(const Config& c, std::span<const Rank> ranks) {
    return chain::devices(ranks, c.layers, prefill_shape(c));
}

// Bits: a pair prefill equals the one-node prefill only with the same windows, so the block is the engine's
// prefill window (prefill_rows) and nothing else until window invariance is proven. Every rank may decode.
inline chain::Request request(std::uint64_t rows, std::uint32_t prefill_rows, std::size_t ranks, std::uint64_t state_rows,
                              double decode_tokens, std::string_view override_text = {}) {
    chain::Request q;
    q.rows = rows;
    q.blocks = {prefill_rows};
    for (std::size_t r = 0; r < ranks; ++r) q.decoders.push_back(static_cast<std::uint32_t>(r));
    q.state_rows = state_rows;
    q.decode_tokens = decode_tokens;
    apply_override(override_text, q);
    return q;
}

} // namespace strix::qwen35::pair_plan
