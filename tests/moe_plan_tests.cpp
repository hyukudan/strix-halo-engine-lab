#include "strix/moe_plan.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {

    strix::MoeExpertStorageRequest storage;
    storage.hidden_size = 2560;
    storage.intermediate_size = 640;
    storage.expert_count = 512;

    storage.gate_up_ggml_type = 23;
    storage.down_ggml_type = 23;
    const auto iq4_storage =
        strix::build_moe_expert_storage_plan(storage);
    expect(iq4_storage.gate_up_compatible,
           "IQ4_XS is compatible with the 2560-wide real gate/up input");
    expect(!iq4_storage.down_compatible,
           "IQ4_XS is incompatible with the 640-wide real down projection");
    expect(iq4_storage.total_bytes_per_expert == 0 &&
               iq4_storage.total_bytes_all_experts == 0,
           "mixed-incompatible expert format is not assigned a fake total size");

    storage.gate_up_ggml_type = 8;
    storage.down_ggml_type = 8;
    const auto q8_storage =
        strix::build_moe_expert_storage_plan(storage);
    expect(q8_storage.gate_up_compatible &&
               q8_storage.down_compatible,
           "Q8_0 block geometry supports both real expert projection inputs");
    expect(q8_storage.total_bytes_per_expert > 0 &&
               q8_storage.total_bytes_all_experts ==
                   q8_storage.total_bytes_per_expert * 512ULL,
           "compatible expert storage computes exact aggregate bytes");

    const std::vector<float> cutoff_tie_logits{2.0F, 2.0F, 2.0F, 1.0F};
    const auto cutoff_tie = strix::select_moe_experts_deterministic(
        cutoff_tie_logits, 2, true);
    expect(cutoff_tie.experts.size() == 2 &&
               cutoff_tie.experts[0] == 0 &&
               cutoff_tie.experts[1] == 1,
           "equal MoE cutoff scores prefer lower expert indices");
    expect(cutoff_tie.weights.size() == 2 &&
               std::abs(cutoff_tie.weights[0] - 0.5F) < 1e-6F &&
               std::abs(cutoff_tie.weights[1] - 0.5F) < 1e-6F,
           "selected MoE weights renormalize after deterministic tie resolution");

    const auto raw_cutoff_tie = strix::select_moe_experts_deterministic(
        cutoff_tie_logits, 2, false);
    expect(raw_cutoff_tie.weights.size() == 2 &&
               raw_cutoff_tie.weights[0] == raw_cutoff_tie.weights[1] &&
               raw_cutoff_tie.weights[0] < 0.5F,
           "unnormalized MoE selection preserves full-softmax probabilities");

    const std::vector<float> ordered_tie_logits{0.0F, 3.0F, 3.0F, 2.0F};
    const auto ordered_tie = strix::select_moe_experts_deterministic(
        ordered_tie_logits, 2, true);
    expect(ordered_tie.experts[0] == 1 && ordered_tie.experts[1] == 2,
           "MoE tie policy is score descending then lower expert index");

    const auto hierarchical_plan =
        strix::build_moe_hierarchical_routing_plan(512, 10, 32);
    expect(hierarchical_plan.groups == 16 &&
               hierarchical_plan.local_candidate_scans == 5120 &&
               hierarchical_plan.group_reduction_candidates == 160,
           "Wave32-shaped hierarchical routing plan exposes exact candidate work");

    const auto hierarchical_tie =
        strix::select_moe_experts_hierarchical(
            cutoff_tie_logits,
            2,
            2,
            true);
    expect(hierarchical_tie.experts == cutoff_tie.experts,
           "hierarchical MoE selection preserves deterministic cutoff ties");

    std::vector<float> representative_router_logits(512);
    for (std::uint32_t expert = 0;
         expert < representative_router_logits.size();
         ++expert) {
        representative_router_logits[expert] =
            static_cast<float>(
                (expert * 37U) % 101U) /
            7.0F;
    }
    const auto serial_representative =
        strix::select_moe_experts_deterministic(
            representative_router_logits,
            10,
            true);
    const auto hierarchical_representative =
        strix::select_moe_experts_hierarchical(
            representative_router_logits,
            10,
            32,
            true);
    expect(hierarchical_representative.experts ==
               serial_representative.experts,
           "hierarchical routing matches canonical expert order at target geometry");
    bool hierarchical_weights_match =
        hierarchical_representative.weights.size() ==
        serial_representative.weights.size();
    for (std::size_t index = 0;
         hierarchical_weights_match &&
         index < serial_representative.weights.size();
         ++index) {
        hierarchical_weights_match =
            std::abs(
                hierarchical_representative.weights[index] -
                serial_representative.weights[index]) <
            1e-6F;
    }
    expect(hierarchical_weights_match,
           "hierarchical routing matches canonical normalized weights");

    bool invalid_hierarchical_width_rejected = false;
    try {
        static_cast<void>(
            strix::build_moe_hierarchical_routing_plan(
                512,
                10,
                0));
    } catch (const std::invalid_argument&) {
        invalid_hierarchical_width_rejected = true;
    }
    expect(invalid_hierarchical_width_rejected,
           "hierarchical routing rejects zero-width candidate groups");

    bool nonfinite_router_rejected = false;
    try {
        const std::vector<float> invalid_router{
            0.0F, std::numeric_limits<float>::quiet_NaN()};
        static_cast<void>(
            strix::select_moe_experts_deterministic(invalid_router, 1, true));
    } catch (const std::invalid_argument&) {
        nonfinite_router_rejected = true;
    }
    expect(nonfinite_router_rejected,
           "non-finite MoE router logits are rejected before softmax");

    const std::vector<std::uint32_t> routes{
        1, 2, 3,
        2, 3, 4,
        2, 5, 6,
        1, 2, 6,
    };

    strix::MoeRouteRequest request;
    request.rows = 4;
    request.experts_per_row = 3;
    request.expert_count = 8;
    request.packed_weight_bytes_per_expert = 1000;
    request.selected_experts = routes;

    const auto plan = strix::build_moe_route_plan(request);
    expect(plan.assignments == 12, "MoE plan counts assignments");
    expect(plan.distinct_experts == 6, "MoE plan counts expert union");
    expect(plan.expert_assignment_counts[2] == 4,
           "MoE plan counts rows per expert");
    expect(plan.maximum_rows_per_expert == 4,
           "MoE plan exposes maximum expert reuse");
    expect(plan.minimum_rows_per_touched_expert == 1,
           "MoE plan exposes minimum touched-expert reuse");
    expect(plan.mean_rows_per_touched_expert == 2.0,
           "MoE plan exposes mean touched-expert reuse");
    expect(plan.mean_adjacent_row_overlap == 5.0 / 9.0,
           "MoE plan measures adjacent-row route overlap");
    expect(plan.direct_assignment_weight_bytes == 12'000,
           "direct tiny-row path models one expert-weight traversal per assignment");
    expect(plan.ideal_grouped_weight_bytes == 6'000,
           "ideal grouped path models one traversal per touched expert");
    expect(plan.ideal_grouped_weight_bytes_saved == 6'000,
           "MoE plan exposes grouped weight-traffic upper bound");
    expect(plan.touched_expert_bitmap.size() == 1 &&
               plan.touched_expert_bitmap[0] == 0x7eULL,
           "MoE plan fuses a deterministic touched-expert bitmap");
    expect(plan.expert_offsets.size() == 9 &&
               plan.expert_offsets[0] == 0 &&
               plan.expert_offsets[1] == 0 &&
               plan.expert_offsets[2] == 2 &&
               plan.expert_offsets[3] == 6,
           "MoE plan fuses CSR expert offsets");
    const auto plan_expert_two_begin =
        static_cast<std::size_t>(plan.expert_offsets[2]);
    expect(plan.assignment_rows[plan_expert_two_begin] == 0 &&
               plan.assignment_slots[plan_expert_two_begin] == 1 &&
               plan.assignment_rows[plan_expert_two_begin + 3] == 3 &&
               plan.assignment_slots[plan_expert_two_begin + 3] == 1,
           "fused grouping metadata preserves row and top-k slot identity");


    request.rows = 4;
    request.experts_per_row = 3;
    request.expert_count = 8;
    request.selected_experts = routes;
    const auto grouping =
        strix::build_moe_grouping_plan(request);
    expect(grouping.expert_offsets.size() == 9,
           "grouping plan exposes expert_count + 1 offsets");
    expect(grouping.expert_offsets[0] == 0 &&
               grouping.expert_offsets[1] == 0 &&
               grouping.expert_offsets[2] == 2 &&
               grouping.expert_offsets[3] == 6 &&
               grouping.expert_offsets[4] == 8,
           "grouping plan builds deterministic expert ranges");

    const auto expert_two_begin =
        static_cast<std::size_t>(grouping.expert_offsets[2]);
    const auto expert_two_end =
        static_cast<std::size_t>(grouping.expert_offsets[3]);
    expect(expert_two_end - expert_two_begin == 4,
           "expert two receives four grouped assignments");
    expect(grouping.assignment_rows[expert_two_begin] == 0 &&
               grouping.assignment_slots[expert_two_begin] == 1 &&
               grouping.assignment_rows[expert_two_begin + 1] == 1 &&
               grouping.assignment_slots[expert_two_begin + 1] == 0 &&
               grouping.assignment_rows[expert_two_begin + 2] == 2 &&
               grouping.assignment_slots[expert_two_begin + 2] == 0 &&
               grouping.assignment_rows[expert_two_begin + 3] == 3 &&
               grouping.assignment_slots[expert_two_begin + 3] == 1,
           "grouped assignments preserve row and top-k slot identity");

    const std::vector<std::uint32_t> bitmap_boundary_routes{
        0, 63,
        64, 69,
    };
    request.rows = 2;
    request.experts_per_row = 2;
    request.expert_count = 70;
    request.selected_experts = bitmap_boundary_routes;
    const auto bitmap_boundary = strix::build_moe_route_plan(request);
    expect(bitmap_boundary.touched_expert_bitmap.size() == 2 &&
               bitmap_boundary.touched_expert_bitmap[0] ==
                   ((std::uint64_t{1} << 0U) |
                    (std::uint64_t{1} << 63U)) &&
               bitmap_boundary.touched_expert_bitmap[1] ==
                   ((std::uint64_t{1} << 0U) |
                    (std::uint64_t{1} << 5U)),
           "touched-expert bitmap crosses 64-bit word boundaries exactly");

    const std::vector<std::uint32_t> unique_routes{
        0, 1,
        2, 3,
        4, 5,
    };
    request.rows = 3;
    request.experts_per_row = 2;
    request.expert_count = 6;
    request.selected_experts = unique_routes;
    const auto unique = strix::build_moe_route_plan(request);
    expect(unique.distinct_experts == 6 &&
               unique.ideal_grouped_weight_bytes_saved == 0,
           "grouping has no ideal weight-reuse benefit when every assignment is unique");


    const std::vector<std::uint32_t> reuse_routes{
        0, 1,
        0, 2,
        0, 3,
        0, 4,
        0, 5,
        0, 6,
        0, 7,
        0, 8,
        0, 9,
        0, 10,
    };
    request.rows = 10;
    request.experts_per_row = 2;
    request.expert_count = 11;
    request.packed_weight_bytes_per_expert = 1000;
    request.selected_experts = reuse_routes;
    const auto execution =
        strix::build_moe_expert_execution_plan(request);
    expect(execution.assignments == 20 &&
               execution.direct_weight_traversals == 20,
           "direct expert path traverses one weight set per assignment");
    expect(execution.batches.size() == 12,
           "grouped execution emits twelve row-class batches");
    expect(execution.batches[0].expert == 0 &&
               execution.batches[0].logical_rows == 8 &&
               execution.batches[1].expert == 0 &&
               execution.batches[1].logical_rows == 2,
           "expert execution plan decomposes reuse into 8/4/2/1 row classes");
    expect(execution.grouped_weight_traversals == 12 &&
               execution.grouped_weight_bytes == 12'000 &&
               execution.grouped_weight_bytes_saved == 8'000,
           "execution plan models realizable grouped weight traffic without padding");

    strix::MoeExpertExecutionCostModel cost_model;
    cost_model.qualified = true;
    cost_model.direct_per_assignment_us = 0.6;
    cost_model.direct_weight_byte_us = 0.0004;
    cost_model.grouped_per_batch_us = 0.6;
    cost_model.grouped_weight_byte_us = 0.0004;
    cost_model.grouped_restore_per_assignment_us = 0.05;

    const auto reuse_decision =
        strix::select_moe_expert_execution_strategy(request, cost_model);
    expect(reuse_decision.strategy ==
               strix::MoeExpertExecutionStrategy::GroupedExpertMajor,
           "qualified complete-cost model selects grouped execution when reuse pays");
    expect(reuse_decision.grouped_estimated_us <
               reuse_decision.direct_estimated_us &&
               reuse_decision.expert_batches == 12 &&
               reuse_decision.grouped_weight_bytes_saved == 8'000,
           "MoE strategy decision exposes complete grouped cost and route-derived reuse");

    request.rows = 3;
    request.experts_per_row = 2;
    request.expert_count = 6;
    request.packed_weight_bytes_per_expert = 1000;
    request.selected_experts = unique_routes;
    const auto unique_decision =
        strix::select_moe_expert_execution_strategy(request, cost_model);
    expect(unique_decision.strategy ==
               strix::MoeExpertExecutionStrategy::DirectAssignment,
           "same measured cost model keeps direct execution when route distribution has no reuse");
    expect(unique_decision.distinct_experts == 6 &&
               unique_decision.expert_batches == 6 &&
               unique_decision.grouped_weight_bytes_saved == 0,
           "strategy decision is driven by actual expert union and batch decomposition");

    strix::MoeExpertExecutionCostModel tie_cost_model;
    tie_cost_model.qualified = true;
    const auto tie_decision =
        strix::select_moe_expert_execution_strategy(request, tie_cost_model);
    expect(tie_decision.direct_estimated_us == 0.0 &&
               tie_decision.grouped_estimated_us == 0.0 &&
               tie_decision.strategy ==
                   strix::MoeExpertExecutionStrategy::DirectAssignment,
           "exact complete-cost ties deterministically prefer the direct path");

    bool unqualified_cost_rejected = false;
    try {
        const strix::MoeExpertExecutionCostModel unqualified;
        static_cast<void>(
            strix::select_moe_expert_execution_strategy(request, unqualified));
    } catch (const std::invalid_argument&) {
        unqualified_cost_rejected = true;
    }
    expect(unqualified_cost_rejected,
           "MoE strategy selection rejects unqualified target cost data");

    bool nonfinite_cost_rejected = false;
    try {
        auto invalid_cost = cost_model;
        invalid_cost.grouped_per_batch_us =
            std::numeric_limits<double>::quiet_NaN();
        static_cast<void>(
            strix::select_moe_expert_execution_strategy(request, invalid_cost));
    } catch (const std::invalid_argument&) {
        nonfinite_cost_rejected = true;
    }
    expect(nonfinite_cost_rejected,
           "MoE strategy selection rejects non-finite cost coefficients");

    strix::MoeLayerExecutionCostModel layer_cost;
    layer_cost.qualified = true;
    layer_cost.routed = cost_model;
    layer_cost.shared_expert_us = 20.0;
    layer_cost.combine_us = 1.5;
    layer_cost.routed_shared_parallel = true;

    request.rows = 10;
    request.experts_per_row = 2;
    request.expert_count = 11;
    request.packed_weight_bytes_per_expert = 1000;
    request.selected_experts = reuse_routes;

    const auto hidden_routed =
        strix::select_moe_layer_execution_strategy(
            request,
            layer_cost);
    expect(
        hidden_routed.strategy ==
                strix::MoeExpertExecutionStrategy::
                    DirectAssignment &&
            hidden_routed.direct_layer_estimated_us ==
                21.5 &&
            hidden_routed.grouped_layer_estimated_us ==
                21.5,
        "parallel shared expert can hide routed optimization and complete-layer tie stays direct");

    layer_cost.shared_expert_us = 1.0;
    const auto routed_dominant =
        strix::select_moe_layer_execution_strategy(
            request,
            layer_cost);
    expect(
        routed_dominant.strategy ==
                strix::MoeExpertExecutionStrategy::
                    GroupedExpertMajor &&
            routed_dominant.grouped_layer_estimated_us <
                routed_dominant.direct_layer_estimated_us,
        "complete-layer policy selects grouped routed experts only when end-to-end layer cost improves");

    layer_cost.routed_shared_parallel = false;
    const auto serial_branches =
        strix::select_moe_layer_execution_strategy(
            request,
            layer_cost);
    expect(
        serial_branches.direct_layer_estimated_us >
                serial_branches.routed.direct_estimated_us &&
            serial_branches.grouped_layer_estimated_us >
                serial_branches.routed.grouped_estimated_us,
        "non-overlapped shared expert contributes explicitly to complete-layer cost");

    bool unqualified_layer_rejected = false;
    try {
        auto invalid_layer = layer_cost;
        invalid_layer.qualified = false;
        static_cast<void>(
            strix::select_moe_layer_execution_strategy(
                request,
                invalid_layer));
    } catch (const std::invalid_argument&) {
        unqualified_layer_rejected = true;
    }
    expect(
        unqualified_layer_rejected,
        "complete MoE layer selection rejects unqualified shared/combine evidence");

    layer_cost.routed_shared_parallel = true;
    layer_cost.shared_expert_us = 1.0;
    const auto qualified_policy =
        strix::bind_moe_layer_cost_model(
            0xabcULL,
            0xdefULL,
            layer_cost);
    const auto qualified_decision =
        strix::select_qualified_moe_layer_execution_strategy(
            request,
            0xabcULL,
            qualified_policy);
    expect(
        qualified_policy.qualification_identity != 0 &&
            qualified_decision.strategy ==
                strix::MoeExpertExecutionStrategy::
                    GroupedExpertMajor,
        "MoE measured layer policy is bound to active runtime identity");

    bool foreign_runtime_rejected = false;
    try {
        static_cast<void>(
            strix::select_qualified_moe_layer_execution_strategy(
                request,
                0xabdULL,
                qualified_policy));
    } catch (const std::invalid_argument&) {
        foreign_runtime_rejected = true;
    }
    expect(
        foreign_runtime_rejected,
        "MoE layer policy cannot be reused by a different runtime identity");

    auto mutated_policy =
        qualified_policy;
    mutated_policy.cost_model.combine_us += 0.25;
    bool mutated_policy_rejected = false;
    try {
        static_cast<void>(
            strix::select_qualified_moe_layer_execution_strategy(
                request,
                0xabcULL,
                mutated_policy));
    } catch (const std::invalid_argument&) {
        mutated_policy_rejected = true;
    }
    expect(
        mutated_policy_rejected,
        "MoE qualification identity rejects post-qualification cost mutation");

    bool duplicate_rejected = false;
    try {
        const std::vector<std::uint32_t> duplicate{1, 1};
        request.rows = 1;
        request.experts_per_row = 2;
        request.expert_count = 4;
        request.selected_experts = duplicate;
        static_cast<void>(strix::build_moe_route_plan(request));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    expect(duplicate_rejected,
           "one row cannot select the same expert twice");

    bool range_rejected = false;
    try {
        const std::vector<std::uint32_t> invalid{0, 4};
        request.rows = 1;
        request.experts_per_row = 2;
        request.expert_count = 4;
        request.selected_experts = invalid;
        static_cast<void>(strix::build_moe_route_plan(request));
    } catch (const std::invalid_argument&) {
        range_rejected = true;
    }
    expect(range_rejected, "out-of-range expert is rejected");

    if (failures == 0) {
        std::cout << "MoE route plan tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
