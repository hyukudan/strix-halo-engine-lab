#include "strix/moe_plan.hpp"

#include <cstdint>
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
    expect(plan.mean_adjacent_row_overlap == 4.0 / 9.0,
           "MoE plan measures adjacent-row route overlap");
    expect(plan.direct_assignment_weight_bytes == 12'000,
           "direct tiny-row path models one expert-weight traversal per assignment");
    expect(plan.ideal_grouped_weight_bytes == 6'000,
           "ideal grouped path models one traversal per touched expert");
    expect(plan.ideal_grouped_weight_bytes_saved == 6'000,
           "MoE plan exposes grouped weight-traffic upper bound");

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
