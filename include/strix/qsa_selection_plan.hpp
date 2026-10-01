#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct QsaCompressedSelectionPlan {
    std::uint64_t token_budget = 0;
    std::uint64_t compress_ratio = 0;
    std::uint64_t block_topk = 0;
    std::uint64_t final_width = 0;
};

[[nodiscard]] QsaCompressedSelectionPlan
build_qsa_compressed_selection_plan(
    std::uint64_t token_budget,
    std::uint64_t compress_ratio);

[[nodiscard]] std::vector<std::int64_t>
expand_qsa_block_selection_reference(
    const QsaCompressedSelectionPlan& plan,
    std::span<const std::int64_t> block_indices,
    std::uint64_t query_position,
    std::uint64_t sequence_length);

[[nodiscard]] std::vector<double>
average_pool_qsa_key_group_reference(
    std::span<const double> values,
    std::uint64_t compress_ratio,
    std::uint64_t row_width);

}  // namespace strix
