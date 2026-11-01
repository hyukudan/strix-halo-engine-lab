#include "strix/indexed_attention_workspace.hpp"
#include "strix/qsa_selection_plan.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "indexed-attention workspace addition overflow");
    }
    return left + right;
}

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            "indexed-attention workspace multiplication overflow");
    }
    return left * right;
}

std::uint64_t align_up(
    std::uint64_t value,
    std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "indexed-attention workspace alignment must be non-zero");
    }
    const auto remainder =
        value % alignment;
    return remainder == 0
        ? value
        : checked_add(
              value,
              alignment - remainder);
}

void append(
    std::uint64_t& cursor,
    std::uint64_t bytes,
    std::uint64_t alignment,
    std::uint64_t& offset,
    std::uint64_t& recorded) {
    if (bytes == 0) {
        throw std::invalid_argument(
            "indexed-attention workspace component must be non-zero");
    }
    cursor =
        align_up(
            cursor,
            alignment);
    offset = cursor;
    recorded = bytes;
    cursor =
        checked_add(
            cursor,
            bytes);
}

}  // namespace

IndexedAttentionWorkspaceLayout
build_indexed_attention_workspace_layout(
    const ModelContract& model,
    std::uint64_t alignment, bool hierarchical_selection, std::uint32_t gqa_heads_per_block, bool external_block_scores, bool compact_attention_output) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (alignment == 0) {
        throw std::invalid_argument(
            "indexed-attention workspace alignment must be non-zero");
    }

    if (compact_attention_output && model.bytes_fp32 != sizeof(float))
        throw std::invalid_argument("compact QSA output requires FP32 containers");
    (void)qsa_gqa_organization_marker(gqa_heads_per_block);
    IndexedAttentionWorkspaceLayout layout;
    layout.compact_attention_output = compact_attention_output;
    layout.alignment =
        alignment;
    layout.maximum_blocks =
        checked_add(model.effective_max_positions(),
         model.indexer_compress_ratio - 1ULL) /
        model.indexer_compress_ratio;
    layout.maximum_selected_tokens =
        model.indexer_budget +
        model.indexer_compress_ratio -
        1ULL;

    const auto index_query_width =
        checked_mul(
            model.indexer_n_heads,
            model.indexer_head_dim);
    const auto q_gate_width =
        checked_mul(
            checked_mul(
                model.attention_heads,
                model.head_dim),
            2ULL);
    const auto kv_width =
        checked_mul(
            model.kv_heads,
            model.head_dim);
    const auto attention_width =
        checked_mul(
            model.attention_heads,
            model.head_dim);

    std::uint64_t cursor = 0;

    append(
        cursor,
        checked_mul(
            index_query_width,
            model.bytes_fp32),
        alignment,
        layout.index_query_offset,
        layout.index_query_bytes);

    append(
        cursor,
        checked_mul(
            model.indexer_head_dim,
            model.bytes_fp32),
        alignment,
        layout.current_index_key_offset,
        layout.current_index_key_bytes);

    if (!external_block_scores) append(
        cursor,
        checked_mul(
            layout.maximum_blocks,
            model.bytes_fp32),
        alignment,
        layout.block_scores_offset,
        layout.block_scores_bytes);

    append(
        cursor,
        checked_mul(
            layout.maximum_selected_tokens,
            sizeof(std::uint32_t)),
        alignment,
        layout.selected_indices_offset,
        layout.selected_indices_bytes);

    append(
        cursor,
        checked_mul(
            q_gate_width,
            model.bytes_fp32),
        alignment,
        layout.q_gate_offset,
        layout.q_gate_bytes);

    append(
        cursor,
        checked_mul(
            kv_width,
            model.bytes_fp32),
        alignment,
        layout.current_k_offset,
        layout.current_k_bytes);

    append(
        cursor,
        checked_mul(
            kv_width,
            model.bytes_fp32),
        alignment,
        layout.current_v_offset,
        layout.current_v_bytes);

    append(
        cursor,
        checked_mul(
            checked_mul(
                model.attention_heads,
                layout.maximum_selected_tokens),
            model.bytes_fp32),
        alignment,
        layout.attention_scores_offset,
        layout.attention_scores_bytes);

    append(
        cursor,
        checked_mul(
            attention_width,
            compact_attention_output ? sizeof(std::uint16_t) : model.bytes_fp32),
        alignment,
        layout.attention_output_offset,
        layout.attention_output_bytes);

    if (hierarchical_selection) {
        if (model.indexer_budget != 2048 || model.indexer_compress_ratio != 4 || alignment % 8U)
            throw std::invalid_argument("hierarchical QSA workspace requires budget2048/compression4 and eight-byte alignment");
        const auto selection = build_qsa_hierarchical_selection_layout(
            std::min(model.max_position_embeddings, qsa_hierarchical_selection_max_visible));
        if (selection.scratch_bytes) {
            append(cursor, selection.scratch_bytes, alignment, layout.selection_ping_offset, layout.selection_ping_bytes);
            append(cursor, selection.scratch_bytes, alignment, layout.selection_pong_offset, layout.selection_pong_bytes);
        }
    }
    if (gqa_heads_per_block != 1) {
        if (model.indexer_budget != 2048 || model.indexer_compress_ratio != 4 || model.head_dim != 256 || alignment % 8U)
            throw std::invalid_argument("GQA QSA workspace requires budget2048/compression4/head256 and eight-byte alignment");
        append(cursor, sizeof(std::uint64_t), alignment, layout.attention_position_offset, layout.attention_position_bytes);
    }
    if (hierarchical_selection || gqa_heads_per_block != 1) {
        append(cursor, sizeof(std::uint32_t), alignment, layout.selection_status_offset, layout.selection_status_bytes);
    }
    layout.total_bytes =
        align_up(
            cursor,
            alignment);
    return layout;
}

}  // namespace strix
