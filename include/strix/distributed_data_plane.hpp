#pragma once

#include "strix/distributed_flow.hpp"
#include "strix/distributed_session.hpp"
#include "strix/dual_link_runtime.hpp"

#include <cstdint>

namespace strix {

struct DistributedBoundaryTransfer {
    DistributedBoundaryFlow flow;
    transport::TransferDescriptor descriptor;
    transport::TransferPlan link_plan;
};

[[nodiscard]] DistributedBoundaryTransfer
plan_distributed_boundary_transfer(
    const DistributedPlacementPlan& placement,
    const DistributedExecutionBoundary& boundary,
    transport::DualLinkRuntimeScheduler& scheduler,
    std::uint64_t connection_epoch,
    std::uint64_t request_id,
    std::uint64_t generation,
    std::uint64_t sequence,
    std::uint32_t dtype_id,
    DistributedBoundaryFlowPolicy flow_policy = {});

void validate_distributed_boundary_transfer_receive(
    DistributedRequestState& request,
    const DistributedPlacementPlan& placement,
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer,
    std::uint32_t local_node,
    DistributedBoundaryFlowPolicy flow_policy = {});

}  // namespace strix
