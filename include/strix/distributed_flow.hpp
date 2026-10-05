#pragma once

#include "strix/distributed_placement.hpp"
#include "strix/dual_link_runtime.hpp"

#include <cstdint>

namespace strix {

struct DistributedBoundaryFlowPolicy {
    std::uint64_t latency_critical_max_bytes =
        64ULL * 1024ULL;
    std::uint64_t bulk_min_bytes =
        1ULL << 20U;
};

struct DistributedBoundaryFlow {
    std::uint64_t flow_id = 0;
    transport::LinkDirection direction =
        transport::LinkDirection::outbound;
    transport::TransferPriority priority =
        transport::TransferPriority::normal;
};

[[nodiscard]] DistributedBoundaryFlow
classify_distributed_boundary_flow(
    const DistributedPlacementPlan& plan,
    const DistributedExecutionBoundary& boundary,
    DistributedBoundaryFlowPolicy policy = {});

}  // namespace strix
