#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

struct IndexedAttentionWorkspaceLayout {
    std::uint64_t alignment = 256;
    std::uint64_t index_query_offset = 0;
    std::uint64_t index_query_bytes = 0;
    std::uint64_t current_index_key_offset = 0;
    std::uint64_t current_index_key_bytes = 0;
    std::uint64_t block_scores_offset = 0;
    std::uint64_t block_scores_bytes = 0;
    std::uint64_t selected_indices_offset = 0;
    std::uint64_t selected_indices_bytes = 0;
    std::uint64_t q_gate_offset = 0;
    std::uint64_t q_gate_bytes = 0;
    std::uint64_t current_k_offset = 0;
    std::uint64_t current_k_bytes = 0;
    std::uint64_t current_v_offset = 0;
    std::uint64_t current_v_bytes = 0;
    std::uint64_t attention_scores_offset = 0;
    std::uint64_t attention_scores_bytes = 0;
    std::uint64_t attention_output_offset = 0;
    std::uint64_t attention_output_bytes = 0;
    std::uint64_t total_bytes = 0;

    std::uint64_t maximum_blocks = 0;
    std::uint64_t maximum_selected_tokens = 0;
    std::uint64_t selection_ping_offset = 0, selection_ping_bytes = 0;
    std::uint64_t selection_pong_offset = 0, selection_pong_bytes = 0;
    std::uint64_t selection_status_offset = 0, selection_status_bytes = 0;
    std::uint64_t attention_position_offset = 0, attention_position_bytes = 0;
};

[[nodiscard]] IndexedAttentionWorkspaceLayout
build_indexed_attention_workspace_layout(
    const ModelContract& model,
    std::uint64_t alignment = 256, bool hierarchical_selection = false, std::uint32_t gqa_heads_per_block = 1);

}  // namespace strix
