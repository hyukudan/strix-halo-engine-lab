#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct MoeExpertStorageRequest {
    std::uint32_t gate_up_ggml_type = 0;
    std::uint32_t down_ggml_type = 0;
    std::uint64_t hidden_size = 0;
    std::uint64_t intermediate_size = 0;
    std::uint64_t expert_count = 0;
};

struct MoeExpertStoragePlan {
    bool gate_up_compatible = false;
    bool down_compatible = false;
    std::uint64_t gate_up_bytes_per_expert = 0;
    std::uint64_t down_bytes_per_expert = 0;
    std::uint64_t total_bytes_per_expert = 0;
    std::uint64_t total_bytes_all_experts = 0;
};

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

    // Fused routing metadata. The bitmap uses one bit per expert and the
    // CSR payload groups assignments by expert while preserving source row
    // and top-k slot identity.
    std::vector<std::uint64_t> touched_expert_bitmap;
    std::vector<std::uint64_t> expert_offsets;
    std::vector<std::uint32_t> assignment_rows;
    std::vector<std::uint32_t> assignment_slots;
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

[[nodiscard]] MoeExpertStoragePlan build_moe_expert_storage_plan(
    const MoeExpertStorageRequest& request);

[[nodiscard]] MoeRoutePlan build_moe_route_plan(
    const MoeRouteRequest& request);

[[nodiscard]] MoeGroupingPlan build_moe_grouping_plan(
    const MoeRouteRequest& request);

[[nodiscard]] MoeExpertExecutionPlan build_moe_expert_execution_plan(
    const MoeRouteRequest& request);

}  // namespace strix
