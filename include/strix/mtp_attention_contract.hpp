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
    std::uint64_t compressed_block_topk = 0;

    bool indexer_uses_attention_rope = false;
    bool indexer_qk_rmsnorm_required = false;
    bool compressed_keys_use_average_pool = false;

    std::uint64_t kv_row_bytes = 0;
    std::uint64_t raw_index_row_bytes = 0;

    // The canonical numerical path is independently oracled. Upstream leaves
    // tied top-k indices unstable, so the engine contract resolves equal block
    // scores by lower compressed-block index for reproducible serial/MTP replay.
    bool no_tie_selection_semantics_resolved = true;
    bool cutoff_tie_semantics_resolved = true;
    bool sparse_selection_semantics_resolved = true;
};

[[nodiscard]] MtpAttentionStructuralContract
build_mtp_attention_structural_contract(
    const ModelContract& model);

void validate_mtp_attention_structural_contract(
    const ModelContract& model,
    const MtpAttentionStructuralContract& contract);

}  // namespace strix
