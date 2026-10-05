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

struct MoeRoutingSelection {
    std::vector<std::uint32_t> experts;
    std::vector<float> weights;
};

struct MoeHierarchicalRoutingPlan {
    std::uint32_t expert_count = 0;
    std::uint32_t top_k = 0;
    std::uint32_t group_width = 0;
    std::uint32_t groups = 0;
    std::uint64_t local_candidate_scans = 0;
    std::uint64_t group_reduction_candidates = 0;
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

enum class MoeExpertExecutionStrategy : std::uint8_t {
    DirectAssignment = 0,
    GroupedExpertMajor = 1,
};

struct MoeExpertExecutionCostModel {
    // Selection is disabled until coefficients come from a qualified
    // measurement set for the active target/runtime identity.
    bool qualified = false;

    double direct_fixed_us = 0.0;
    double direct_per_assignment_us = 0.0;
    double direct_weight_byte_us = 0.0;

    double grouped_fixed_us = 0.0;
    double grouped_per_batch_us = 0.0;
    double grouped_weight_byte_us = 0.0;
    double grouped_restore_per_assignment_us = 0.0;
};

struct MoeExpertExecutionDecision {
    MoeExpertExecutionStrategy strategy =
        MoeExpertExecutionStrategy::DirectAssignment;

    double direct_estimated_us = 0.0;
    double grouped_estimated_us = 0.0;

    std::uint64_t assignments = 0;
    std::uint64_t expert_batches = 0;
    std::uint64_t direct_weight_bytes = 0;
    std::uint64_t grouped_weight_bytes = 0;
    std::uint64_t grouped_weight_bytes_saved = 0;

    std::uint32_t distinct_experts = 0;
    std::uint32_t maximum_rows_per_expert = 0;
    double mean_rows_per_touched_expert = 0.0;
    double mean_adjacent_row_overlap = 0.0;
};

struct MoeLayerExecutionCostModel {
    // Complete-layer policy is disabled until routed, shared and combine
    // measurements belong to one qualified target/runtime evidence set.
    bool qualified = false;
    MoeExpertExecutionCostModel routed;
    double shared_expert_us = 0.0;
    double combine_us = 0.0;
    bool routed_shared_parallel = true;
};

struct MoeLayerExecutionDecision {
    MoeExpertExecutionStrategy strategy =
        MoeExpertExecutionStrategy::DirectAssignment;

    MoeExpertExecutionDecision routed;
    double shared_expert_us = 0.0;
    double combine_us = 0.0;
    double direct_layer_estimated_us = 0.0;
    double grouped_layer_estimated_us = 0.0;
    bool routed_shared_parallel = true;
};

struct MoeQualifiedLayerPolicy {
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t measurement_identity = 0;
    std::uint64_t qualification_identity = 0;
    MoeLayerExecutionCostModel cost_model;
};

struct MoeRouteTraceRequest {
    std::uint32_t window_rows = 0;
    std::uint32_t experts_per_row = 0;
    std::uint32_t expert_count = 0;
    std::uint64_t packed_weight_bytes_per_expert = 0;
    std::span<const std::uint32_t> selected_experts;
};

struct MoeRouteTraceSummary {
    std::uint64_t windows = 0;
    std::uint64_t logical_rows = 0;
    std::uint64_t assignments = 0;
    std::uint64_t windows_with_grouped_savings = 0;

    std::uint64_t direct_weight_bytes = 0;
    std::uint64_t grouped_weight_bytes = 0;
    std::uint64_t grouped_weight_bytes_saved = 0;

    std::uint32_t maximum_distinct_experts = 0;
    double mean_distinct_experts = 0.0;
    double mean_maximum_rows_per_expert = 0.0;
    double mean_adjacent_row_overlap = 0.0;

    // Index is the number of distinct experts observed in one window.
    std::vector<std::uint64_t> distinct_expert_histogram;
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

[[nodiscard]] MoeRoutingSelection select_moe_experts_deterministic(
    std::span<const float> logits,
    std::uint32_t top_k,
    bool normalize_selected = true);

[[nodiscard]] MoeHierarchicalRoutingPlan build_moe_hierarchical_routing_plan(
    std::uint32_t expert_count,
    std::uint32_t top_k,
    std::uint32_t group_width = 32);

[[nodiscard]] MoeRoutingSelection select_moe_experts_hierarchical(
    std::span<const float> logits,
    std::uint32_t top_k,
    std::uint32_t group_width = 32,
    bool normalize_selected = true);

[[nodiscard]] MoeRoutePlan build_moe_route_plan(
    const MoeRouteRequest& request);

[[nodiscard]] MoeGroupingPlan build_moe_grouping_plan(
    const MoeRouteRequest& request);

[[nodiscard]] MoeRouteTraceSummary summarize_moe_route_trace(
    const MoeRouteTraceRequest& request);

[[nodiscard]] MoeExpertExecutionPlan build_moe_expert_execution_plan(
    const MoeRouteRequest& request);

[[nodiscard]] MoeExpertExecutionDecision select_moe_expert_execution_strategy(
    const MoeRouteRequest& request,
    const MoeExpertExecutionCostModel& cost_model);

[[nodiscard]] MoeLayerExecutionDecision select_moe_layer_execution_strategy(
    const MoeRouteRequest& request,
    const MoeLayerExecutionCostModel& cost_model);

[[nodiscard]] MoeQualifiedLayerPolicy bind_moe_layer_cost_model(
    std::uint64_t runtime_compatibility_identity,
    std::uint64_t measurement_identity,
    const MoeLayerExecutionCostModel& cost_model);

[[nodiscard]] MoeLayerExecutionDecision
select_qualified_moe_layer_execution_strategy(
    const MoeRouteRequest& request,
    std::uint64_t active_runtime_compatibility_identity,
    const MoeQualifiedLayerPolicy& policy);

}  // namespace strix
