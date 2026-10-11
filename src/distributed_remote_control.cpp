#include "strix/distributed_remote_control.hpp"
#include <limits>
#include <stdexcept>
namespace strix {
SequencedDistributedNativeControlPlane::SequencedDistributedNativeControlPlane(
    const DistributedPlacementPlan& placement,
    DistributedRuntimeRequest request,
    DistributedControlWireBackend& wire,
    std::uint32_t protocol_version)
    : placement_(&placement), request_(request), wire_(&wire),
      protocol_version_(protocol_version) {
    if (placement.plan_identity == 0 || !placement.executable ||
        request.connection_epoch == 0 || request.request_id == 0 ||
        request.generation == 0 || request.dtype_id == 0 ||
        protocol_version_ == 0) {
        throw std::invalid_argument("sequenced distributed control requires executable placement/request identities");
    }
}
void SequencedDistributedNativeControlPlane::require_healthy() const {
    if (failed_) throw std::logic_error("distributed control plane is poisoned");
}
bool SequencedDistributedNativeControlPlane::step_in_plan(
    const DistributedStepPlacement& step) const noexcept {
    for (const auto& c : placement_->steps) {
        if (c.step_id==step.step_id && c.op==step.op && c.node==step.node &&
            c.remote==step.remote && c.placement_ambiguous==step.placement_ambiguous) return true;
    }
    return false;
}
bool SequencedDistributedNativeControlPlane::boundary_in_plan(
    const DistributedExecutionBoundary& b) const noexcept {
    for (const auto& c : placement_->boundaries) {
        if (c.boundary_id==b.boundary_id && c.source_step==b.source_step &&
            c.destination_step==b.destination_step && c.source_node==b.source_node &&
            c.destination_node==b.destination_node && c.payload_bytes==b.payload_bytes) return true;
    }
    return false;
}
void SequencedDistributedNativeControlPlane::execute_remote_step(
    const DistributedStepPlacement& step) {
    require_healthy();
    if (!step.remote || step.node==placement_->local_node || step.placement_ambiguous ||
        !step_in_plan(step) || next_sequence_==std::numeric_limits<std::uint64_t>::max()) {
        failed_=true; throw std::invalid_argument("remote-step control request is not valid for placement");
    }
    DistributedRemoteStepCommand cmd;
    cmd.protocol_version=protocol_version_;
    cmd.placement_plan_identity=placement_->plan_identity;
    cmd.request_id=request_.request_id; cmd.generation=request_.generation;
    cmd.sequence=next_sequence_; cmd.step_id=step.step_id; cmd.op=step.op; cmd.target_node=step.node;
    const auto ack=wire_->execute_remote_step(cmd);
    const bool ok=
        ack.protocol_version==cmd.protocol_version &&
        ack.placement_plan_identity==cmd.placement_plan_identity &&
        ack.request_id==cmd.request_id && ack.generation==cmd.generation &&
        ack.sequence==cmd.sequence && ack.step_id==cmd.step_id &&
        ack.target_node==cmd.target_node && ack.success;
    if(!ok){failed_=true; throw std::runtime_error("remote-step ACK is stale, foreign or unsuccessful");}
    ++next_sequence_;
}
void SequencedDistributedNativeControlPlane::publish_boundary_transfer(
    const DistributedExecutionBoundary& b,
    const DistributedBoundaryTransfer& t) {
    require_healthy();
    const bool ok=boundary_in_plan(b) && b.source_node==placement_->local_node &&
        t.flow.direction==transport::LinkDirection::outbound &&
        t.descriptor.protocol_version==protocol_version_ &&
        t.descriptor.connection_epoch==request_.connection_epoch &&
        t.descriptor.model_plan_id==placement_->plan_identity &&
        t.descriptor.request_id==request_.request_id &&
        t.descriptor.generation==request_.generation &&
        t.descriptor.boundary_id==b.boundary_id &&
        t.descriptor.dtype_id==request_.dtype_id &&
        t.descriptor.sequence==next_boundary_send_sequence_ &&
        next_boundary_send_sequence_!=std::numeric_limits<std::uint64_t>::max() &&
        t.descriptor.byte_length==b.payload_bytes;
    if(!ok){failed_=true; throw std::invalid_argument("outbound boundary metadata does not match control request");}
    wire_->publish_boundary_transfer(b,t);
    ++next_boundary_send_sequence_;
}
DistributedBoundaryTransfer SequencedDistributedNativeControlPlane::await_boundary_transfer(
    const DistributedExecutionBoundary& b) {
    require_healthy();
    if(!boundary_in_plan(b) || b.destination_node!=placement_->local_node){
        failed_=true; throw std::invalid_argument("inbound boundary is not owned by local placement");
    }
    auto t=wire_->await_boundary_transfer(b);
    const bool ok=t.flow.direction==transport::LinkDirection::inbound &&
        t.descriptor.protocol_version==protocol_version_ &&
        t.descriptor.connection_epoch==request_.connection_epoch &&
        t.descriptor.model_plan_id==placement_->plan_identity &&
        t.descriptor.request_id==request_.request_id &&
        t.descriptor.generation==request_.generation &&
        t.descriptor.boundary_id==b.boundary_id &&
        t.descriptor.dtype_id==request_.dtype_id &&
        t.descriptor.sequence==next_boundary_receive_sequence_ &&
        next_boundary_receive_sequence_!=std::numeric_limits<std::uint64_t>::max() &&
        t.descriptor.byte_length==b.payload_bytes;
    if(!ok){failed_=true; throw std::runtime_error("inbound boundary metadata is stale or foreign");}
    ++next_boundary_receive_sequence_;
    return t;
}
}  // namespace strix
