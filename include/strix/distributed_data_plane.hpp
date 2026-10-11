#pragma once

#include "strix/distributed_flow.hpp"
#include "strix/distributed_session.hpp"
#include "strix/dual_link_runtime.hpp"

#include <cstdint>
#include <vector>

namespace strix {

// These identities belong to physical leased links. The descriptor below
// retains the logical control epoch, request and message sequence.
struct DistributedNativeSliceIdentity {
    std::uint64_t connection_epoch = 0;
    std::uint64_t sequence = 0;
};

struct DistributedBoundaryTransfer {
    DistributedBoundaryFlow flow;
    transport::TransferDescriptor descriptor;
    transport::TransferPlan link_plan;
    // Empty before native send preparation; otherwise one identity per slice
    // in exactly the same order as link_plan.slices.
    std::vector<DistributedNativeSliceIdentity> native_slices;
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
