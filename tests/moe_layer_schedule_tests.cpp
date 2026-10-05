#include "strix/moe_layer_schedule.hpp"

#include <iostream>
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
    const std::vector<std::uint32_t> experts{
        1, 2, 3, 4,
        1, 2, 5, 6,
    };

    strix::MoeRouteRequest request;
    request.rows = 2;
    request.experts_per_row = 4;
    request.expert_count = 16;
    request.packed_weight_bytes_per_expert = 1024;
    request.selected_experts = experts;

    strix::MoeLayerExecutionCostModel costs;
    costs.qualified = true;
    costs.routed.qualified = true;
    costs.routed.direct_fixed_us = 1.0;
    costs.routed.direct_per_assignment_us = 1.0;
    costs.routed.grouped_fixed_us = 1.0;
    costs.routed.grouped_per_batch_us = 0.5;
    costs.shared_expert_us = 2.0;
    costs.combine_us = 1.0;
    costs.routed_shared_parallel = true;

    const auto policy =
        strix::bind_moe_layer_cost_model(
            0x1111ULL,
            0x2222ULL,
            costs);
    const auto parallel =
        strix::build_qualified_moe_layer_schedule(
            request,
            0x1111ULL,
            policy);

    expect(parallel.schedule_identity != 0 &&
               parallel.nodes.size() == 3 &&
               parallel.routed_shared_parallel,
           "qualified MoE schedule materializes routed/shared/combine stages");
    expect(parallel.nodes[0].dependencies.empty() &&
               parallel.nodes[1].dependencies.empty() &&
               parallel.nodes[2].dependencies.size() == 2,
           "parallel MoE schedule launches routed/shared independently and joins at combine");

    auto serial_costs = costs;
    serial_costs.routed_shared_parallel = false;
    const auto serial_policy =
        strix::bind_moe_layer_cost_model(
            0x1111ULL,
            0x3333ULL,
            serial_costs);
    const auto serial =
        strix::build_qualified_moe_layer_schedule(
            request,
            0x1111ULL,
            serial_policy);

    expect(!serial.routed_shared_parallel &&
               serial.nodes[1].dependencies.size() == 1 &&
               serial.nodes[1].dependencies[0] ==
                   strix::MoeLayerScheduleStage::routed_experts,
           "serialized MoE schedule makes shared expert depend on routed branch");

    bool foreign_rejected = false;
    try {
        static_cast<void>(
            strix::build_qualified_moe_layer_schedule(
                request,
                0x9999ULL,
                policy));
    } catch (const std::invalid_argument&) {
        foreign_rejected = true;
    }
    expect(foreign_rejected,
           "MoE layer schedule rejects a policy from another runtime");

    if (failures == 0) {
        std::cout << "MoE layer schedule tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
