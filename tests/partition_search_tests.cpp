#include "strix/partition_search.hpp"

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
    strix::PartitionSearchPolicy policy;
    policy.node0_weight_budget_bytes = 70;
    policy.node1_weight_budget_bytes = 70;
    policy.local_serial_token_us = 100.0;
    policy.minimum_latency_improvement_fraction = 0.05;

    const std::vector<strix::PartitionCandidate> candidates{
        {
            "whole-layer-capacity",
            strix::PartitionGranularity::whole_layer,
            50,
            50,
            10,
            1,
            62.0,
            55.0,
            18.0,
            5.0,
        },
        {
            "operator-latency",
            strix::PartitionGranularity::operator_level,
            60,
            40,
            20,
            3,
            40.0,
            42.0,
            7.0,
            3.0,
        },
        {
            "too-large",
            strix::PartitionGranularity::whole_layer,
            80,
            20,
            5,
            1,
            30.0,
            30.0,
            5.0,
            2.0,
        },
    };

    const auto result =
        strix::search_partition_candidates(
            candidates,
            policy);

    expect(result.whole_layer_baseline_present,
           "partition search requires and recognizes whole-layer baseline");
    expect(result.capacity_choice.has_value() &&
               *result.capacity_choice == 0,
           "capacity objective prefers the balanced feasible whole-layer split");
    expect(result.latency_choice.has_value() &&
               *result.latency_choice == 1,
           "latency objective may select finer operator partition");
    expect(result.evaluations[1].critical_compute_us == 42.0 &&
               result.evaluations[1].estimated_token_us == 52.0,
           "partition cost uses critical compute plus uncovered communication/control");
    expect(!result.evaluations[2].capacity_feasible,
           "over-budget candidate is never selected despite low latency");

    auto no_baseline = candidates;
    no_baseline[0].granularity =
        strix::PartitionGranularity::operator_level;
    no_baseline[2].granularity =
        strix::PartitionGranularity::operator_level;
    bool rejected = false;
    try {
        static_cast<void>(
            strix::search_partition_candidates(
                no_baseline,
                policy));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected,
           "partition search rejects comparisons without whole-layer baseline");

    if (failures == 0) {
        std::cout << "partition search tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
