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
    expect(execution.batches.size() == 13,
           "grouped execution emits thirteen row-class batches");
    expect(execution.batches[0].expert == 0 &&
               execution.batches[0].logical_rows == 8 &&
               execution.batches[1].expert == 0 &&
               execution.batches[1].logical_rows == 2,
           "expert execution plan decomposes reuse into 8/4/2/1 row classes");
    expect(execution.grouped_weight_traversals == 13 &&
               execution.grouped_weight_bytes == 13'000 &&
               execution.grouped_weight_bytes_saved == 7'000,
           "execution plan models realizable grouped weight traffic without padding");

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
