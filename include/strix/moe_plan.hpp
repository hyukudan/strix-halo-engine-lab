#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct MoeRouteRequest {
    std::uint32_t rows = 0;
    std::uint32_t experts_per_row = 0;
    std::uint32_t expert_count = 0;
    std::uint64_t packed_weight_bytes_per_expert = 0;
    std::span<const std::uint32_t> selected_experts;
};

struct MoeRoutePlan {
    std::uint32_t rows = 0;
    std::uint32_t experts_per_row = 0;
    std::uint32_t expert_count = 0;
    std::uint32_t distinct_experts = 0;
    std::uint32_t maximum_rows_per_expert = 0;
    std::uint32_t minimum_rows_per_touched_expert = 0;
    double mean_rows_per_touched_expert = 0.0;
    double mean_adjacent_row_overlap = 0.0;

    std::uint64_t assignments = 0;
    std::uint64_t packed_weight_bytes_per_expert = 0;
    std::uint64_t direct_assignment_weight_bytes = 0;
    std::uint64_t ideal_grouped_weight_bytes = 0;
    std::uint64_t ideal_grouped_weight_bytes_saved = 0;

    std::vector<std::uint32_t> expert_assignment_counts;
    std::vector<std::uint32_t> touched_experts;
};

struct MoeExpertBatch {
    std::uint32_t expert = 0;
    std::uint64_t assignment_offset = 0;
    std::uint32_t logical_rows = 0;
};

struct MoeExpertExecutionPlan {
    std::uint64_t assignments = 0;
    std::uint64_t expert_batches = 0;
    std::uint64_t direct_weight_traversals = 0;
    std::uint64_t grouped_weight_traversals = 0;
    std::uint64_t direct_weight_bytes = 0;
    std::uint64_t grouped_weight_bytes = 0;
    std::uint64_t grouped_weight_bytes_saved = 0;
    std::vector<MoeExpertBatch> batches;
};

struct MoeGroupingPlan {
    std::uint32_t expert_count = 0;
    std::uint64_t assignments = 0;

    // CSR-style expert ranges. Size is expert_count + 1.
    std::vector<std::uint64_t> expert_offsets;

    // Grouped assignment payload. Entries in
    // [expert_offsets[e], expert_offsets[e+1]) belong to expert e.
    std::vector<std::uint32_t> assignment_rows;
    std::vector<std::uint32_t> assignment_slots;
};

[[nodiscard]] MoeRoutePlan build_moe_route_plan(
    const MoeRouteRequest& request);

[[nodiscard]] MoeGroupingPlan build_moe_grouping_plan(
    const MoeRouteRequest& request);

[[nodiscard]] MoeExpertExecutionPlan build_moe_expert_execution_plan(
    const MoeRouteRequest& request);

}  // namespace strix
