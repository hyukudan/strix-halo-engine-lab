#pragma once

#include "strix/model_contract.hpp"

#include <algorithm>
#include <cstdint>

namespace strix {

// One KV row per token and layer: [K | V] codes, then (int8_token only) the
// FP32 scales [K head 0..kv_heads) | V head 0..kv_heads)].
struct IndexedAttentionStateLayout {
    std::uint64_t kv_key_offset = 0;
    std::uint64_t kv_key_bytes = 0;
    std::uint64_t kv_value_offset = 0;
    std::uint64_t kv_value_bytes = 0;
    std::uint64_t kv_scale_offset = 0;
    std::uint64_t kv_scale_bytes = 0;
    std::uint64_t kv_row_bytes = 0;
    std::uint64_t raw_index_row_bytes = 0;
};

[[nodiscard]] IndexedAttentionStateLayout
build_indexed_attention_state_layout(
    const ModelContract& model);

// Per-layer cache bytes of the compressed-history and sparse-latent attention forms (candidate compact layouts, not a
// final wire format). The model declares its dimensions; the formulas are shared.
//
// A block-scaled FP8 row: one code byte per element and one FP32 scale per `block` elements.
[[nodiscard]] constexpr std::uint64_t block_scaled_fp8_row_bytes(std::uint64_t width, std::uint64_t block) noexcept {
    return width + (width / block) * sizeof(float);
}

// Rows a compressed-history layer keeps for `context` tokens: a sliding window of raw rows plus one compressed row
// per `ratio` tokens (rounded up).
struct CompressedHistoryRows {
    std::uint32_t history_rows = 0;
    std::uint32_t compressed_rows = 0;
};

[[nodiscard]] constexpr CompressedHistoryRows compressed_history_rows(
    std::uint32_t context, std::uint32_t window, std::uint32_t ratio) noexcept {
    return {std::min(context, window), (context + ratio - 1U) / ratio};
}

// Unfused compression carry: projection and gate kept in FP32 for one complete incomplete group.
[[nodiscard]] constexpr std::uint64_t compressed_history_carry_bytes(std::uint64_t ratio, std::uint64_t width) noexcept {
    return ratio * width * 2U * sizeof(float);
}

// One sparse-latent row: block-scaled FP8 latent, FP8 rotary part, E2M1 index key (two elements per byte) and one
// E8M0 scale byte per `index_block` index elements.
struct SparseLatentRowDims {
    std::uint64_t latent_width = 0, latent_block = 0, rope_width = 0, index_width = 0, index_block = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return latent_block > 0 && index_block > 0 && latent_width % latent_block == 0 && index_width % index_block == 0;
    }
};

[[nodiscard]] constexpr std::uint64_t sparse_latent_row_bytes(const SparseLatentRowDims& d) noexcept {
    return block_scaled_fp8_row_bytes(d.latent_width, d.latent_block) + d.rope_width + d.index_width / 2U +
           d.index_width / d.index_block;
}

}  // namespace strix
