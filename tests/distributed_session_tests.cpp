#include "strix/distributed_session.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::ModelContentIdentity content() {
    strix::ModelContentIdentity id;
    id.bytes[0] = 0x42;
    id.bytes[1] = 0x11;
    return id;
}

}  // namespace

int main() {
    strix::DistributedControlIdentity local;
    local.node = 0;
    local.placement_plan_identity = 0x1000ULL;
    local.runtime_compatibility_identity = 0x2000ULL;
    local.model_content_identity = content();

    auto peer = local;
    peer.node = 1;

    const auto compatible =
        strix::reconcile_distributed_control(
            local, peer, 1);
    expect(compatible.compatible &&
               compatible.blockers.empty(),
           "matching distributed control identities reconcile");

    auto foreign = peer;
    ++foreign.placement_plan_identity;
    const auto mismatch =
        strix::reconcile_distributed_control(
            local, foreign, 1);
    expect(!mismatch.compatible &&
               !mismatch.blockers.empty(),
           "foreign placement plan is rejected at control plane");

    strix::DistributedExecutionBoundary boundary;
    boundary.boundary_id = 0x3333ULL;
    boundary.source_node = 0;
    boundary.destination_node = 1;
    boundary.payload_bytes = 512;

    strix::DistributedRequestState state;
    state.placement_plan_identity =
        local.placement_plan_identity;
    state.request_id = 7;
    state.generation = 3;

    strix::transport::TransferDescriptor descriptor;
    descriptor.connection_epoch = 9;
    descriptor.model_plan_id =
        local.placement_plan_identity;
    descriptor.request_id = state.request_id;
    descriptor.generation = state.generation;
    descriptor.boundary_id =
        boundary.boundary_id;
    descriptor.sequence = 1;
    descriptor.shape_fingerprint = 1;
    descriptor.dtype_id = 1;
    descriptor.byte_length =
        boundary.payload_bytes;

    strix::validate_distributed_boundary_receive(
        state,
        boundary,
        descriptor,
        1);
    expect(state.next_receive_sequence == 2 &&
               !state.failed,
           "valid boundary advances exact receive sequence");

    bool stale_rejected = false;
    try {
        strix::validate_distributed_boundary_receive(
            state,
            boundary,
            descriptor,
            1);
    } catch (const std::invalid_argument&) {
        stale_rejected = true;
    }
    expect(stale_rejected &&
               state.failed,
           "duplicate/stale boundary poisons distributed request");

    if (failures == 0) {
        std::cout << "distributed session tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
