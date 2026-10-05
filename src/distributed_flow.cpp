#include "strix/distributed_flow.hpp"

#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) &
            0xffULL;
        hash *=
            fnv_prime;
    }
}

bool boundary_in_plan(
    const DistributedPlacementPlan& plan,
    const DistributedExecutionBoundary& boundary) noexcept {
    for (const auto& candidate :
         plan.boundaries) {
        if (candidate.boundary_id ==
                boundary.boundary_id &&
            candidate.source_step ==
                boundary.source_step &&
            candidate.destination_step ==
                boundary.destination_step &&
            candidate.source_node ==
                boundary.source_node &&
            candidate.destination_node ==
                boundary.destination_node &&
            candidate.payload_bytes ==
                boundary.payload_bytes) {
            return true;
        }
    }
    return false;
}

}  // namespace

DistributedBoundaryFlow
classify_distributed_boundary_flow(
    const DistributedPlacementPlan& plan,
    const DistributedExecutionBoundary& boundary,
    DistributedBoundaryFlowPolicy policy) {
    if (plan.plan_identity == 0 ||
        !plan.executable ||
        boundary.boundary_id == 0 ||
        boundary.payload_bytes == 0 ||
        policy.bulk_min_bytes == 0 ||
        policy.latency_critical_max_bytes >=
            policy.bulk_min_bytes ||
        !boundary_in_plan(
            plan,
            boundary)) {
        throw std::invalid_argument(
            "distributed boundary flow classification requires valid plan/boundary/policy");
    }

    const bool local_source =
        boundary.source_node ==
        plan.local_node;
    const bool local_destination =
        boundary.destination_node ==
        plan.local_node;
    if (local_source ==
        local_destination) {
        throw std::invalid_argument(
            "distributed boundary flow must cross the local node exactly once");
    }

    DistributedBoundaryFlow result;
    result.direction =
        local_source
            ? transport::LinkDirection::outbound
            : transport::LinkDirection::inbound;

    if (boundary.payload_bytes <=
        policy.latency_critical_max_bytes) {
        result.priority =
            transport::TransferPriority::
                latency_critical;
    } else if (boundary.payload_bytes >=
               policy.bulk_min_bytes) {
        result.priority =
            transport::TransferPriority::bulk;
    }

    std::uint64_t hash =
        fnv_offset;
    hash_u64(
        hash,
        plan.plan_identity);
    hash_u64(
        hash,
        boundary.boundary_id);
    hash_u64(
        hash,
        boundary.source_node);
    hash_u64(
        hash,
        boundary.destination_node);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            result.direction));
    result.flow_id =
        hash;

    if (result.flow_id == 0) {
        throw std::logic_error(
            "distributed boundary flow identity must be non-zero");
    }
    return result;
}

}  // namespace strix
