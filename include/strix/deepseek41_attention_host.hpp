#pragma once

#include "strix/deepseek41_model.hpp"
#include "strix/ds41_attention.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

// The attention piece (ds41_attention.hpp) bound to a trimmed deepseek41 model for one decode row per step: per-layer
// window rings, the compressed KV / index-key caches and the ratio-2 carry of every KV source, the candidate pool and
// each index source's top-k. The projections feeding the piece (compressor kv/gate, indexer q_b and weights_proj, all
// F16) are F16 rows of the layer piece (FP32 activations, lane dot), the same declaration the device runner uses.
namespace strix::deepseek41 {

class AttentionHost {
public:
    using TensorBytes = std::function<std::span<const std::uint8_t>(const std::string&)>;
    // Ring entries per layer: the 128-token window plus one decode row.
    static constexpr std::uint32_t ring_capacity = ds41::attn::kWindow + 8U;

    AttentionHost(const TargetModel& model, std::uint32_t layers, std::uint32_t max_positions, TensorBytes tensor);
    ~AttentionHost();

    // kv: wkv(x) of the layer (head_dim values, before kv_norm and RoPE).
    void window_write(std::uint32_t layer, std::span<const float> kv, std::uint32_t position);
    // x: the attention input after attn_norm (hidden values). KV source layers only.
    void compress(std::uint32_t layer, std::span<const float> x, std::uint32_t position);
    // x as above, qr: q_a after q_norm (q_lora_rank values). Index source layers only.
    void index(std::uint32_t layer, std::span<const float> x, std::span<const float> qr, std::uint32_t position);
    // q: heads x head_dim before RoPE; o: heads x head_dim, after the inverse RoPE.
    void attention(std::uint32_t layer, std::span<const float> q, std::uint32_t position, std::span<float> o) const;

    // Roles of the layer (the piece's released plan, checked against the admitted model).
    [[nodiscard]] const ds41::attn::LayerAttention& role(std::uint32_t layer) const { return roles_.at(layer); }
    // Last top-k of an index source (ascending compressed positions).
    [[nodiscard]] const std::vector<std::uint32_t>& topk(std::uint32_t layer) const;
    // KV sources: the compressed KV and index-key caches (entries at position / ratio); the candidate source's blocks of
    // the last row (one byte per block).
    [[nodiscard]] std::span<const std::uint8_t> kv_cache(std::uint32_t layer) const;
    [[nodiscard]] std::span<const std::uint8_t> idx_cache(std::uint32_t layer) const;
    [[nodiscard]] std::span<const std::uint8_t> candidates() const noexcept { return candidates_; }

    struct Layer;

private:
    const TargetModel& model_;
    TensorBytes tensor_;
    std::array<ds41::attn::LayerAttention, 40> roles_{};
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<std::uint8_t> candidates_;  // one byte per block of the candidate source
};

}  // namespace strix::deepseek41
