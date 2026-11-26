#pragma once

#include "attn_core/selected_block_plan.hpp"
#include "mm_rtx_prep.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strix::minimax::rtxpf::cuda {

static_assert(kQHeads % kKvHeads == 0 && kKvHeads == kIdxHeads);
using MsaAttentionShape = attn::SelectedBlockShape<kHeadDim, kHeadDim, kBlock, kTopK, kQHeads / kKvHeads>;

[[nodiscard]] inline constexpr attn::SelectedBlockSchedule msa_attention_schedule(bool legacy) {
    return legacy ? attn::SelectedBlockSchedule::warp : attn::SelectedBlockSchedule::staged;
}

[[nodiscard]] inline constexpr attn::SelectedBlockSequence msa_attention_sequence(const SeqView& seq) {
    return {seq.k, seq.v, seq.row0, seq.pos0, seq.cap};
}

[[nodiscard]] inline constexpr attn::SelectedBlockPlan plan_msa_attention(std::size_t rows, bool legacy) {
    // Selection takes an unsigned item count, one item per (row, KV group).
    if (rows > std::numeric_limits<unsigned>::max() / kKvHeads)
        throw std::invalid_argument("MMATTN refused MiniMax RTX row extent");
    const auto plan = attn::make_selected_block_plan<MsaAttentionShape>(
        msa_attention_schedule(legacy), static_cast<std::uint32_t>(rows), kKvHeads);
    if (!plan) throw std::invalid_argument("MMATTN refused MiniMax RTX launch geometry");
    return *plan;
}

}  // namespace strix::minimax::rtxpf::cuda
