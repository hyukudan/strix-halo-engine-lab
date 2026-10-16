#pragma once

#include <cstdint>
#include <array>
#include <atomic>
#include <string_view>
#include <stdexcept>
#include <span>
#include <vector>

namespace strix {
inline constexpr std::string_view qsa_hierarchical_selection_marker = ".hierarchical-top512-v1";
inline constexpr std::uint64_t qsa_hierarchical_selection_max_visible = 131072;
[[nodiscard]] inline std::string_view qsa_gqa_organization_marker(std::uint32_t heads) {
    switch (heads) {
    case 1: return {};
    case 2: return ".gqa-heads2-v1";
    case 4: return ".gqa-heads4-v1";
    default: throw std::invalid_argument("QSA heads per block must be 1, 2 or 4");
    }
}
struct QsaHierarchicalSelectionLayout {
    std::uint64_t scratch_items = 0;
    std::uint64_t scratch_bytes = 0; // EACH of ping and pong; eight-byte candidates.
};
[[nodiscard]] QsaHierarchicalSelectionLayout build_qsa_hierarchical_selection_layout(
    std::uint64_t maximum_visible);
struct QsaHierarchicalSelectionStats {
    std::atomic<std::uint64_t> submissions{0};
    std::array<std::atomic<std::uint64_t>, 48> layer_submissions{};
};


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
