#include "strix/partition_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace strix {
namespace {

void validate_policy(
    const PartitionSearchPolicy& policy) {
    if (policy.node0_weight_budget_bytes == 0 ||
        policy.node1_weight_budget_bytes == 0 ||
        !std::isfinite(policy.local_serial_token_us) ||
        policy.local_serial_token_us <= 0.0 ||
        !std::isfinite(
            policy.minimum_latency_improvement_fraction) ||
        policy.minimum_latency_improvement_fraction < 0.0 ||
        policy.minimum_latency_improvement_fraction >= 1.0) {
        throw std::invalid_argument(
            "partition search policy is invalid");
    }
}

void validate_candidate(
    const PartitionCandidate& candidate) {
    if (candidate.name.empty() ||
        candidate.node0_weight_bytes == 0 ||
        candidate.node1_weight_bytes == 0 ||
        !std::isfinite(candidate.node0_compute_us) ||
        candidate.node0_compute_us < 0.0 ||
        !std::isfinite(candidate.node1_compute_us) ||
        candidate.node1_compute_us < 0.0 ||
        !std::isfinite(candidate.uncovered_communication_us) ||
        candidate.uncovered_communication_us < 0.0 ||
        !std::isfinite(candidate.control_us) ||
        candidate.control_us < 0.0) {
        throw std::invalid_argument(
            "partition candidate contains invalid cost or memory data");
    }
}

std::uint64_t checked_add_u64(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() -
            left) {
        throw std::overflow_error(
            "partition candidate byte accounting overflow");
    }
    return left + right;
}

double checked_add_double(
    double left,
    double right,
    const char* message) {
    const auto result =
        left + right;
    if (!std::isfinite(result)) {
        throw std::overflow_error(message);
    }
    return result;
}

bool count_tensor(
    const ManifestTensor& tensor,
    PartitionCandidateBuildPolicy policy) noexcept {
    if (policy.weight_accounting ==
        PartitionWeightAccounting::
            all_manifest_tensors) {
        return true;
    }
    return tensor.residency ==
        ResidencyPolicy::resident;
}

}  // namespace

PartitionCandidate
build_partition_candidate_from_placement(
    const DistributedPlacementPlan& placement,
    const ModelManifest& manifest,
    std::span<const PartitionStepTiming> step_timings,
    std::span<const PartitionBoundaryTiming> boundary_timings,
    std::string name,
    PartitionGranularity granularity,
    PartitionCandidateBuildPolicy policy) {
    if (placement.plan_identity == 0 ||
        !placement.executable ||
        name.empty() ||
        placement.steps.empty()) {
        throw std::invalid_argument(
            "partition candidate builder requires executable placement and name");
    }

    std::unordered_map<
        std::uint32_t,
        PartitionStepTiming>
        measured_steps;
    measured_steps.reserve(
        step_timings.size());
    for (const auto& timing :
         step_timings) {
        if (!std::isfinite(
                timing.node0_us) ||
            timing.node0_us < 0.0 ||
            !std::isfinite(
                timing.node1_us) ||
            timing.node1_us < 0.0 ||
            !measured_steps.emplace(
                timing.step_id,
                timing).second) {
            throw std::invalid_argument(
                "partition step timing evidence is invalid or duplicated");
        }
    }

    std::unordered_map<
        std::uint64_t,
        PartitionBoundaryTiming>
        measured_boundaries;
    measured_boundaries.reserve(
        boundary_timings.size());
    for (const auto& timing :
         boundary_timings) {
        if (timing.boundary_id == 0 ||
            !std::isfinite(
                timing.uncovered_communication_us) ||
            timing.uncovered_communication_us < 0.0 ||
            !std::isfinite(
                timing.control_us) ||
            timing.control_us < 0.0 ||
            !measured_boundaries.emplace(
                timing.boundary_id,
                timing).second) {
            throw std::invalid_argument(
                "partition boundary timing evidence is invalid or duplicated");
        }
    }

    PartitionCandidate candidate;
    candidate.name =
        std::move(name);
    candidate.granularity =
        granularity;

    for (const auto& tensor :
         manifest.tensors) {
        if (!count_tensor(
                tensor,
                policy)) {
            continue;
        }
        const auto node =
            tensor.placement.kind ==
                    PlacementKind::remote
                ? tensor.placement.node
                : placement.local_node;
        if (node > 1) {
            throw std::invalid_argument(
                "two-node partition candidate cannot account tensor on node > 1");
        }
        auto& bytes =
            node == 0
                ? candidate.node0_weight_bytes
                : candidate.node1_weight_bytes;
        bytes =
            checked_add_u64(
                bytes,
                tensor.byte_size);
    }

    for (const auto& step :
         placement.steps) {
        if (step.placement_ambiguous ||
            step.node > 1) {
            throw std::invalid_argument(
                "two-node partition candidate requires unambiguous node 0/1 steps");
        }
        const auto found =
            measured_steps.find(
                step.step_id);
        if (found ==
            measured_steps.end()) {
            throw std::invalid_argument(
                "partition candidate is missing measured step timing");
        }
        const auto value =
            step.node == 0
                ? found->second.node0_us
                : found->second.node1_us;
        auto& total =
            step.node == 0
                ? candidate.node0_compute_us
                : candidate.node1_compute_us;
        total =
            checked_add_double(
                total,
                value,
                "partition compute timing overflow");
    }

    candidate.transfer_bytes_per_token =
        placement.
            boundary_payload_bytes_per_token;
    if (placement.boundaries.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "partition synchronization count exceeds uint32");
    }
    candidate.synchronization_count =
        static_cast<std::uint32_t>(
            placement.boundaries.size());

    for (const auto& boundary :
         placement.boundaries) {
        const auto found =
            measured_boundaries.find(
                boundary.boundary_id);
        if (found ==
            measured_boundaries.end()) {
            throw std::invalid_argument(
                "partition candidate is missing measured boundary timing");
        }
        candidate.uncovered_communication_us =
            checked_add_double(
                candidate.
                    uncovered_communication_us,
                found->second.
                    uncovered_communication_us,
                "partition communication timing overflow");
        candidate.control_us =
            checked_add_double(
                candidate.control_us,
                found->second.control_us,
                "partition control timing overflow");
    }

    if (candidate.node0_weight_bytes == 0 ||
        candidate.node1_weight_bytes == 0) {
        throw std::invalid_argument(
            "two-node partition candidate requires resident weight bytes on both nodes");
    }

    return candidate;
}

PartitionCandidateEvaluation
evaluate_partition_candidate(
    const PartitionCandidate& candidate,
    const PartitionSearchPolicy& policy,
    std::size_t candidate_index) {
    validate_policy(policy);
    validate_candidate(candidate);

    PartitionCandidateEvaluation result;
    result.candidate_index =
        candidate_index;
    result.capacity_feasible =
        candidate.node0_weight_bytes <=
            policy.node0_weight_budget_bytes &&
        candidate.node1_weight_bytes <=
            policy.node1_weight_budget_bytes;
    result.maximum_node_weight_bytes =
        std::max(
            candidate.node0_weight_bytes,
            candidate.node1_weight_bytes);
    result.minimum_node_weight_bytes =
        std::min(
            candidate.node0_weight_bytes,
            candidate.node1_weight_bytes);
    result.memory_imbalance_bytes =
        result.maximum_node_weight_bytes -
        result.minimum_node_weight_bytes;

    result.critical_compute_us =
        std::max(
            candidate.node0_compute_us,
            candidate.node1_compute_us);
    result.estimated_token_us =
        result.critical_compute_us +
        candidate.uncovered_communication_us +
        candidate.control_us;
    if (!std::isfinite(result.estimated_token_us)) {
        throw std::overflow_error(
            "partition candidate token cost is not finite");
    }

    result.latency_improvement_fraction =
        (policy.local_serial_token_us -
         result.estimated_token_us) /
        policy.local_serial_token_us;
    result.latency_useful =
        result.capacity_feasible &&
        result.latency_improvement_fraction >=
            policy.minimum_latency_improvement_fraction;
    return result;
}

PartitionSearchResult search_partition_candidates(
    std::span<const PartitionCandidate> candidates,
    const PartitionSearchPolicy& policy) {
    validate_policy(policy);
    if (candidates.empty()) {
        throw std::invalid_argument(
            "partition search requires at least one candidate");
    }

    PartitionSearchResult result;
    result.evaluations.reserve(
        candidates.size());

    for (std::size_t index = 0;
         index < candidates.size();
         ++index) {
        if (candidates[index].granularity ==
            PartitionGranularity::whole_layer) {
            result.whole_layer_baseline_present =
                true;
        }
        result.evaluations.push_back(
            evaluate_partition_candidate(
                candidates[index],
                policy,
                index));
    }

    if (!result.whole_layer_baseline_present) {
        throw std::invalid_argument(
            "partition search requires a whole-layer baseline candidate");
    }

    for (const auto& evaluation :
         result.evaluations) {
        if (!evaluation.capacity_feasible) {
            continue;
        }

        if (!result.capacity_choice.has_value()) {
            result.capacity_choice =
                evaluation.candidate_index;
        } else {
            const auto& current =
                result.evaluations[
                    *result.capacity_choice];
            if (evaluation.maximum_node_weight_bytes <
                    current.maximum_node_weight_bytes ||
                (evaluation.maximum_node_weight_bytes ==
                     current.maximum_node_weight_bytes &&
                 (evaluation.memory_imbalance_bytes <
                      current.memory_imbalance_bytes ||
                  (evaluation.memory_imbalance_bytes ==
                       current.memory_imbalance_bytes &&
                   evaluation.estimated_token_us <
                       current.estimated_token_us)))) {
                result.capacity_choice =
                    evaluation.candidate_index;
            }
        }

        if (!evaluation.latency_useful) {
            continue;
        }
        if (!result.latency_choice.has_value() ||
            evaluation.estimated_token_us <
                result.evaluations[
                    *result.latency_choice].
                    estimated_token_us) {
            result.latency_choice =
                evaluation.candidate_index;
        }
    }

    return result;
}

}  // namespace strix
