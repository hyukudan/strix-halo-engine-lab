#pragma once

#include "strix/distributed_placement.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/transport.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct DistributedControlIdentity {
    std::uint32_t protocol_version = 1;
    std::uint32_t node = 0;
    std::uint64_t placement_plan_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    ModelContentIdentity model_content_identity;
};

struct DistributedControlDecision {
    bool compatible = false;
    std::vector<const char*> blockers;
};

[[nodiscard]] DistributedControlDecision
reconcile_distributed_control(
    const DistributedControlIdentity& local,
    const DistributedControlIdentity& peer,
    std::uint32_t expected_peer_node);

struct DistributedRequestState {
    std::uint64_t placement_plan_identity = 0;
    std::uint64_t request_id = 0;
    std::uint64_t generation = 0;
    std::uint64_t next_receive_sequence = 1;
    bool failed = false;
};

void validate_distributed_boundary_receive(
    DistributedRequestState& state,
    const DistributedExecutionBoundary& boundary,
    const transport::TransferDescriptor& descriptor,
    std::uint32_t local_node);

}  // namespace strix
