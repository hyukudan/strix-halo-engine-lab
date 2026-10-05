#include "strix/distributed_control_receiver.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {

DistributedControlReceiver::DistributedControlReceiver(
    const DistributedPlacementPlan& placement,
    DistributedRuntimeRequest request,
    std::uint32_t receiver_node,
    DistributedNativeLocalExecutor& executor,
    std::uint32_t protocol_version)
    : placement_(&placement),
      request_(request),
      receiver_node_(receiver_node),
      executor_(&executor),
      protocol_version_(protocol_version) {
    if (placement.plan_identity == 0 ||
        !placement.executable ||
        request.connection_epoch == 0 ||
        request.request_id == 0 ||
        request.generation == 0 ||
        request.dtype_id == 0 ||
        protocol_version_ == 0 ||
        receiver_node_ ==
            placement.local_node) {
        throw std::invalid_argument(
            "distributed control receiver requires global placement, request and distinct receiver node");
    }
}

void DistributedControlReceiver::require_healthy() const {
    if (failed_) {
        throw std::logic_error(
            "distributed control receiver is poisoned");
    }
}

const DistributedStepPlacement*
DistributedControlReceiver::find_step(
    std::uint32_t step_id) const noexcept {
    for (const auto& step :
         placement_->steps) {
        if (step.step_id == step_id) {
            return &step;
        }
    }
    return nullptr;
}

const DistributedExecutionBoundary*
DistributedControlReceiver::find_boundary(
    std::uint64_t boundary_id) const noexcept {
    for (const auto& boundary :
         placement_->boundaries) {
        if (boundary.boundary_id ==
            boundary_id) {
            return &boundary;
        }
    }
    return nullptr;
}

bool DistributedControlReceiver::execute_remote_step(
    const DistributedRemoteStepCommand& command) {
    require_healthy();

    const auto* step =
        find_step(
            command.step_id);
    const bool valid =
        command.protocol_version ==
            protocol_version_ &&
        command.placement_plan_identity ==
            placement_->plan_identity &&
        command.request_id ==
            request_.request_id &&
        command.generation ==
            request_.generation &&
        command.sequence ==
            next_command_sequence_ &&
        command.target_node ==
            receiver_node_ &&
        step != nullptr &&
        !step->placement_ambiguous &&
        step->node ==
            receiver_node_ &&
        step->op ==
            command.op;

    if (!valid) {
        failed_ = true;
        throw std::invalid_argument(
            "remote-step command is stale, foreign or not owned by receiver node");
    }
    if (next_command_sequence_ ==
        std::numeric_limits<std::uint64_t>::max()) {
        failed_ = true;
        throw std::overflow_error(
            "remote-step receiver sequence exhausted");
    }

    try {
        executor_->execute_local_step(
            *step);
    } catch (...) {
        failed_ = true;
        throw;
    }

    ++next_command_sequence_;
    return true;
}

void DistributedControlReceiver::receive_boundary_transfer(
    const DistributedBoundaryTransfer& transfer) {
    require_healthy();

    const auto* boundary =
        find_boundary(
            transfer.descriptor.boundary_id);
    const bool valid =
        boundary != nullptr &&
        boundary->destination_node ==
            receiver_node_ &&
        transfer.descriptor.protocol_version ==
            protocol_version_ &&
        transfer.descriptor.connection_epoch ==
            request_.connection_epoch &&
        transfer.descriptor.model_plan_id ==
            placement_->plan_identity &&
        transfer.descriptor.request_id ==
            request_.request_id &&
        transfer.descriptor.generation ==
            request_.generation &&
        transfer.descriptor.byte_length ==
            boundary->payload_bytes &&
        transfer.flow.flow_id != 0 &&
        transfer.link_plan.kind !=
            transport::PlanKind::unavailable &&
        !transfer.link_plan.slices.empty();

    if (!valid) {
        failed_ = true;
        throw std::invalid_argument(
            "boundary metadata is stale, foreign or not addressed to receiver node");
    }

    const auto inserted =
        pending_boundaries_.emplace(
            boundary->boundary_id,
            transfer);
    if (!inserted.second) {
        failed_ = true;
        throw std::logic_error(
            "duplicate pending boundary metadata for receiver");
    }
}

bool DistributedControlReceiver::has_boundary(
    std::uint64_t boundary_id) const noexcept {
    return pending_boundaries_.find(
               boundary_id) !=
           pending_boundaries_.end();
}

DistributedBoundaryTransfer
DistributedControlReceiver::take_boundary(
    std::uint64_t boundary_id) {
    require_healthy();
    const auto found =
        pending_boundaries_.find(
            boundary_id);
    if (found ==
        pending_boundaries_.end()) {
        throw std::invalid_argument(
            "requested boundary metadata is not pending");
    }

    auto result =
        std::move(
            found->second);
    pending_boundaries_.erase(
        found);
    return result;
}

}  // namespace strix
