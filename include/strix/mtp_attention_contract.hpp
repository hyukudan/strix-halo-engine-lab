#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

struct MtpAttentionStructuralContract {
    std::uint64_t full_attention_layers = 0;
    std::uint64_t attention_heads = 0;
    std::uint64_t kv_heads = 0;
    std::uint64_t head_dim = 0;
    std::uint64_t rotary_dim = 0;

    bool qsa_indexer_required = false;
    std::uint64_t indexer_heads = 0;
    std::uint64_t indexer_kv_heads = 0;
    std::uint64_t indexer_head_dim = 0;
    std::uint64_t indexer_budget = 0;
    std::uint64_t indexer_compress_ratio = 0;

    std::uint64_t kv_row_bytes = 0;
    std::uint64_t raw_index_row_bytes = 0;

    // Structure/state geometry is known; sparse-selection semantics and tie
    // behavior remain independently unresolved.
    bool sparse_selection_semantics_resolved = false;
};

[[nodiscard]] MtpAttentionStructuralContract
build_mtp_attention_structural_contract(
    const ModelContract& model);

void validate_mtp_attention_structural_contract(
    const ModelContract& model,
    const MtpAttentionStructuralContract& contract);

}  // namespace strix
