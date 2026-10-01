#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct QsaCompressedSelectionPlan {
    std::uint64_t token_budget = 0;
    std::uint64_t compress_ratio = 0;
    std::uint64_t block_topk = 0;
};

[[nodiscard]] QsaCompressedSelectionPlan
build_qsa_compressed_selection_plan(
    std::uint64_t token_budget,
    std::uint64_t compress_ratio);

[[nodiscard]] std::vector<std::uint64_t>
expand_qsa_block_selection_reference(
    const QsaCompressedSelectionPlan& plan,
    std::span<const std::uint64_t> block_indices,
    std::uint64_t sequence_length);

}  // namespace strix
