#pragma once

#include "strix/moe_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::moe_core {

struct ExpertMajorTile {
    std::uint32_t expert;
    std::uint32_t first_assignment;
    std::uint32_t rows;
};

struct ExpertMajorPlan {
    MoeGroupingPlan grouping;
    std::vector<float> assignment_weights;    // Expert-major, parallel to assignment_rows.
    std::vector<std::uint32_t> combine_rows;   // [token][combine rank] -> expert-major row.
    std::vector<ExpertMajorTile> tiles;        // Expert-major partitions with no padding.
};

// Extend the common host CSR plan with the operands for grouped execution. The
// incoming [token][route slot] order remains intact in the CSR metadata. Each
// token's inverse permutation is separately ordered by ascending expert ID so
// the grouped combine uses the same addition order as the scalar reference.
[[nodiscard]] inline ExpertMajorPlan build_expert_major_plan(
    const MoeRouteRequest& request, std::span<const float> token_major_weights,
    std::uint32_t rows_per_tile) {
    if (rows_per_tile == 0 || token_major_weights.size() != request.selected_experts.size() ||
        token_major_weights.size() > std::numeric_limits<std::uint32_t>::max() ||
        !std::ranges::all_of(token_major_weights, [](float weight) { return std::isfinite(weight); })) {
        throw std::invalid_argument("invalid expert-major weights or tile geometry");
    }

    auto grouping_request = request;
    grouping_request.packed_weight_bytes_per_expert = 0;
    ExpertMajorPlan plan;
    plan.grouping = build_moe_grouping_plan(grouping_request);
    plan.assignment_weights.resize(token_major_weights.size());
    plan.combine_rows.resize(token_major_weights.size());

    std::vector<std::uint32_t> combine_rank(request.rows);
    for (std::size_t assignment = 0; assignment < plan.assignment_weights.size(); ++assignment) {
        const auto token = plan.grouping.assignment_rows[assignment];
        const auto original_slot = plan.grouping.assignment_slots[assignment];
        const auto base = std::size_t{token} * request.experts_per_row;
        plan.assignment_weights[assignment] = token_major_weights[base + original_slot];
        plan.combine_rows[base + combine_rank[token]++] = static_cast<std::uint32_t>(assignment);
    }

    for (std::uint32_t expert = 0; expert < request.expert_count; ++expert) {
        auto first = plan.grouping.expert_offsets[expert];
        const auto end = plan.grouping.expert_offsets[std::size_t{expert} + 1U];
        while (first < end) {
            const auto rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(end - first, rows_per_tile));
            plan.tiles.push_back({expert, static_cast<std::uint32_t>(first), rows});
            first += rows;
        }
    }
    return plan;
}

}  // namespace strix::moe_core
