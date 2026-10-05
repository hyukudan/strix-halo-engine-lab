#include "strix/distributed_session.hpp"

#include <limits>
#include <stdexcept>

namespace strix {

DistributedControlDecision
reconcile_distributed_control(
    const DistributedControlIdentity& local,
    const DistributedControlIdentity& peer,
    std::uint32_t expected_peer_node) {
    DistributedControlDecision result;

    if (local.protocol_version == 0 ||
        peer.protocol_version == 0 ||
        local.protocol_version !=
            peer.protocol_version) {
        result.blockers.push_back(
            "protocol version mismatch");
    }
    if (local.node == peer.node ||
        peer.node != expected_peer_node) {
        result.blockers.push_back(
            "peer node identity mismatch");
    }
    if (local.placement_plan_identity == 0 ||
        local.placement_plan_identity !=
            peer.placement_plan_identity) {
        result.blockers.push_back(
            "placement plan identity mismatch");
    }
    if (local.runtime_compatibility_identity == 0 ||
        local.runtime_compatibility_identity !=
            peer.runtime_compatibility_identity) {
        result.blockers.push_back(
            "runtime compatibility identity mismatch");
    }
    if (!local.model_content_identity.valid() ||
        local.model_content_identity !=
            peer.model_content_identity) {
        result.blockers.push_back(
            "model content identity mismatch");
    }

    result.compatible =
        result.blockers.empty();
    return result;
}

void validate_distributed_boundary_receive(
    DistributedRequestState& state,
    const DistributedExecutionBoundary& boundary,
    const transport::TransferDescriptor& descriptor,
    std::uint32_t local_node) {
    if (state.failed) {
        throw std::logic_error(
            "failed distributed request cannot accept boundaries");
    }
    if (state.placement_plan_identity == 0 ||
        state.request_id == 0 ||
        state.generation == 0 ||
        state.next_receive_sequence == 0) {
        throw std::invalid_argument(
            "distributed request state is not initialized");
    }

    const bool valid =
        boundary.boundary_id != 0 &&
        boundary.destination_node ==
            local_node &&
        descriptor.model_plan_id ==
            state.placement_plan_identity &&
        descriptor.request_id ==
            state.request_id &&
        descriptor.generation ==
            state.generation &&
        descriptor.boundary_id ==
            boundary.boundary_id &&
        descriptor.sequence ==
            state.next_receive_sequence &&
        descriptor.byte_length ==
            boundary.payload_bytes;

    if (!valid) {
        state.failed = true;
        throw std::invalid_argument(
            "distributed boundary descriptor does not match request/placement sequence");
    }

    if (state.next_receive_sequence ==
        std::numeric_limits<std::uint64_t>::max()) {
        state.failed = true;
        throw std::overflow_error(
            "distributed receive sequence exhausted");
    }
    ++state.next_receive_sequence;
}

}  // namespace strix
