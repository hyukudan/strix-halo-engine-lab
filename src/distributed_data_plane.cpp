#include "strix/distributed_data_plane.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

std::uint64_t planned_bytes(
    const transport::TransferPlan& plan) {
    std::uint64_t total = 0;
    for (const auto& slice :
         plan.slices) {
        if (slice.byte_length == 0 ||
            slice.byte_offset >
                std::numeric_limits<std::uint64_t>::max() -
                    slice.byte_length) {
            throw std::invalid_argument(
                "distributed transfer plan contains invalid slice");
        }
        if (slice.byte_length >
            std::numeric_limits<std::uint64_t>::max() -
                total) {
            throw std::overflow_error(
                "distributed transfer byte count overflow");
        }
        total += slice.byte_length;
    }
    return total;
}

bool same_flow(
    const DistributedBoundaryFlow& left,
    const DistributedBoundaryFlow& right) noexcept {
    return left.flow_id ==
               right.flow_id &&
           left.direction ==
               right.direction &&
           left.priority ==
               right.priority;
}

}  // namespace

DistributedBoundaryTransfer
plan_distributed_boundary_transfer(
    const DistributedPlacementPlan& placement,
    const DistributedExecutionBoundary& boundary,
    transport::DualLinkRuntimeScheduler& scheduler,
    std::uint64_t connection_epoch,
    std::uint64_t request_id,
    std::uint64_t generation,
    std::uint64_t sequence,
    std::uint32_t dtype_id,
    DistributedBoundaryFlowPolicy flow_policy) {
    auto flow =
        classify_distributed_boundary_flow(
            placement,
            boundary,
            flow_policy);
    auto descriptor =
        make_distributed_boundary_descriptor(
            placement,
            boundary,
            connection_epoch,
            request_id,
            generation,
            sequence,
            dtype_id);

    auto link_plan =
        scheduler.plan(
            flow.flow_id,
            flow.direction,
            boundary.payload_bytes,
            flow.priority);

    if (link_plan.kind ==
            transport::PlanKind::unavailable ||
        link_plan.slices.empty() ||
        planned_bytes(link_plan) !=
            boundary.payload_bytes) {
        throw std::logic_error(
            "distributed boundary has no complete qualified link plan");
    }

    DistributedBoundaryTransfer result;
    result.flow =
        flow;
    result.descriptor =
        descriptor;
    result.link_plan =
        std::move(link_plan);
    return result;
}

void validate_distributed_boundary_transfer_receive(
    DistributedRequestState& request,
    const DistributedPlacementPlan& placement,
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer,
    std::uint32_t local_node,
    DistributedBoundaryFlowPolicy flow_policy) {
    const auto expected =
        classify_distributed_boundary_flow(
            placement,
            boundary,
            flow_policy, local_node);

    if (expected.direction !=
            transport::LinkDirection::inbound ||
        !same_flow(
            expected,
            transfer.flow) ||
        transfer.descriptor.byte_length !=
            boundary.payload_bytes ||
        transfer.link_plan.kind ==
            transport::PlanKind::unavailable ||
        transfer.link_plan.slices.empty() ||
        planned_bytes(
            transfer.link_plan) !=
            boundary.payload_bytes) {
        request.failed = true;
        throw std::invalid_argument(
            "distributed boundary transfer package is incompatible with receive side");
    }

    validate_distributed_boundary_receive(
        request,
        boundary,
        transfer.descriptor,
        local_node);
}

}  // namespace strix
