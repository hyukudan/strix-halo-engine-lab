#include "strix/partition_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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

}  // namespace

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
