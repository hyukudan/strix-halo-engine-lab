#pragma once

#include "mm_rtx_select.hpp"
#include "ops/chain/prep_plan.hpp"

#include <stdexcept>
#include <string_view>
#include <utility>

namespace strix::minimax::rtxpf::cuda {

inline constexpr unsigned kHeadDim = 128, kQHeads = 64, kKvHeads = 4;
inline constexpr unsigned kMsaRotaryHalf = 32;
static_assert(kHeadDim == ops::chain::kPrepWidth);

[[nodiscard]] inline constexpr bool admits_msa_prep_gain(std::string_view dtype, std::uint64_t bytes) {
    if (dtype == "F32") return bytes == kHeadDim * sizeof(float);
    return (dtype == "BF16" || dtype == "F16") && bytes == kHeadDim * sizeof(std::uint16_t);
}

[[nodiscard]] inline constexpr ops::chain::PrepSequence msa_prep_sequence(const SeqView& seq) {
    return {seq.row0, seq.rows, seq.pos0, seq.cap};
}

[[nodiscard]] inline ops::chain::PrepPlan plan_msa_prep(
    std::size_t rows, std::size_t rotary_positions, std::span<const ops::chain::PrepSequence> seqs,
    unsigned layer, unsigned layers, std::size_t index_layer_stride) {
    if (layer >= layers) throw std::invalid_argument("MMPREP refused MiniMax RTX layer outside cache");
    if (!std::in_range<unsigned>(rows) || !std::in_range<unsigned>(rotary_positions))
        throw std::invalid_argument("MMPREP refused MiniMax RTX row or rotary table extent");
    const auto plan = ops::chain::plan_prep(
        {static_cast<unsigned>(rows), kQHeads, kKvHeads, kIdxHeads, 1, kMsaRotaryHalf,
         static_cast<unsigned>(rotary_positions)}, seqs, layer, index_layer_stride);
    if (!plan) throw std::invalid_argument("MMPREP refused MiniMax RTX sequence, rotary or cache layout");
    return *plan;
}

}  // namespace strix::minimax::rtxpf::cuda
