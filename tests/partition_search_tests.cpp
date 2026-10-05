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

    {
        strix::DistributedPlacementPlan placement;
        placement.local_node = 0;
        placement.plan_identity = 0x1234ULL;
        placement.executable = true;
        placement.has_remote_work = true;
        placement.boundary_payload_bytes_per_token = 4096;
        placement.steps = {
            {0, strix::SerialOp::token_embedding, 0, false, false},
            {1, strix::SerialOp::hyper_expand, 1, true, false},
            {2, strix::SerialOp::lm_head, 0, false, false},
        };

        strix::DistributedExecutionBoundary boundary;
        boundary.boundary_id = 0x5555ULL;
        boundary.source_step = 0;
        boundary.destination_step = 1;
        boundary.source_node = 0;
        boundary.destination_node = 1;
        boundary.payload_bytes = 4096;
        placement.boundaries = {boundary};

        strix::ModelManifest manifest;
        manifest.layout_identity = 1;

        strix::ManifestTensor local;
        local.name = "local";
        local.byte_size = 60;
        local.residency =
            strix::ResidencyPolicy::resident;
        manifest.tensors.push_back(local);

        strix::ManifestTensor remote;
        remote.name = "remote";
        remote.byte_size = 40;
        remote.residency =
            strix::ResidencyPolicy::resident;
        remote.placement.kind =
            strix::PlacementKind::remote;
        remote.placement.node = 1;
        manifest.tensors.push_back(remote);

        strix::ManifestTensor sparse;
        sparse.name = "sparse";
        sparse.byte_size = 1000;
        sparse.residency =
            strix::ResidencyPolicy::
                file_backed_candidate;
        manifest.tensors.push_back(sparse);

        const std::vector<strix::PartitionStepTiming>
            step_timings{
                {0, 10.0, 20.0},
                {1, 11.0, 7.0},
                {2, 5.0, 9.0},
            };
        const std::vector<strix::PartitionBoundaryTiming>
            boundary_timings{
                {0x5555ULL, 3.0, 1.0},
            };

        const auto built =
            strix::build_partition_candidate_from_placement(
                placement,
                manifest,
                step_timings,
                boundary_timings,
                "measured-placement",
                strix::PartitionGranularity::
                    operator_level);

        expect(
            built.node0_weight_bytes == 60 &&
                built.node1_weight_bytes == 40 &&
                built.node0_compute_us == 15.0 &&
                built.node1_compute_us == 7.0 &&
                built.transfer_bytes_per_token == 4096 &&
                built.synchronization_count == 1 &&
                built.uncovered_communication_us == 3.0 &&
                built.control_us == 1.0,
            "placement candidate builder derives resident memory and measured step/boundary cost");

        strix::PartitionCandidateBuildPolicy
            all_policy;
        all_policy.weight_accounting =
            strix::PartitionWeightAccounting::
                all_manifest_tensors;
        const auto all_weights =
            strix::build_partition_candidate_from_placement(
                placement,
                manifest,
                step_timings,
                boundary_timings,
                "all-manifest",
                strix::PartitionGranularity::
                    whole_layer,
                all_policy);
        expect(
            all_weights.node0_weight_bytes == 1060 &&
                all_weights.node1_weight_bytes == 40,
            "candidate builder can explicitly include file-backed manifest tensors");

        auto missing_timings =
            step_timings;
        missing_timings.pop_back();
        bool missing_rejected = false;
        try {
            static_cast<void>(
                strix::build_partition_candidate_from_placement(
                    placement,
                    manifest,
                    missing_timings,
                    boundary_timings,
                    "missing",
                    strix::PartitionGranularity::
                        whole_layer));
        } catch (const std::invalid_argument&) {
            missing_rejected = true;
        }
        expect(
            missing_rejected,
            "candidate builder rejects incomplete step timing evidence");
    }

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
