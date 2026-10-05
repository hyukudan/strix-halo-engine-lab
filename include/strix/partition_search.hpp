#pragma once

#include "strix/distributed_placement.hpp"
#include "strix/model_manifest.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace strix {

enum class PartitionGranularity : std::uint8_t {
    whole_layer,
    operator_level,
};

struct PartitionCandidate {
    std::string name;
    PartitionGranularity granularity =
        PartitionGranularity::whole_layer;

    std::uint64_t node0_weight_bytes = 0;
    std::uint64_t node1_weight_bytes = 0;
    std::uint64_t transfer_bytes_per_token = 0;
    std::uint32_t synchronization_count = 0;

    double node0_compute_us = 0.0;
    double node1_compute_us = 0.0;
    double uncovered_communication_us = 0.0;
    double control_us = 0.0;
};

struct PartitionStepTiming {
    std::uint32_t step_id = 0;
    double node0_us = 0.0;
    double node1_us = 0.0;
};

struct PartitionBoundaryTiming {
    std::uint64_t boundary_id = 0;
    double uncovered_communication_us = 0.0;
    double control_us = 0.0;
};

enum class PartitionWeightAccounting : std::uint8_t {
    resident_only,
    all_manifest_tensors,
};

struct PartitionCandidateBuildPolicy {
    PartitionWeightAccounting weight_accounting =
        PartitionWeightAccounting::resident_only;
};

struct PartitionSearchPolicy {
    std::uint64_t node0_weight_budget_bytes = 0;
    std::uint64_t node1_weight_budget_bytes = 0;
    double local_serial_token_us = 0.0;
    double minimum_latency_improvement_fraction = 0.0;
};

struct PartitionCandidateEvaluation {
    std::size_t candidate_index = 0;
    bool capacity_feasible = false;
    bool latency_useful = false;

    std::uint64_t maximum_node_weight_bytes = 0;
    std::uint64_t minimum_node_weight_bytes = 0;
    std::uint64_t memory_imbalance_bytes = 0;

    double critical_compute_us = 0.0;
    double estimated_token_us = 0.0;
    double latency_improvement_fraction = 0.0;
};

struct PartitionSearchResult {
    std::vector<PartitionCandidateEvaluation> evaluations;
    std::optional<std::size_t> capacity_choice;
    std::optional<std::size_t> latency_choice;
    bool whole_layer_baseline_present = false;
};

[[nodiscard]] PartitionCandidate
build_partition_candidate_from_placement(
    const DistributedPlacementPlan& placement,
    const ModelManifest& manifest,
    std::span<const PartitionStepTiming> step_timings,
    std::span<const PartitionBoundaryTiming> boundary_timings,
    std::string name,
    PartitionGranularity granularity,
    PartitionCandidateBuildPolicy policy = {});

[[nodiscard]] PartitionCandidateEvaluation
evaluate_partition_candidate(
    const PartitionCandidate& candidate,
    const PartitionSearchPolicy& policy,
    std::size_t candidate_index = 0);

[[nodiscard]] PartitionSearchResult search_partition_candidates(
    std::span<const PartitionCandidate> candidates,
    const PartitionSearchPolicy& policy);

}  // namespace strix
