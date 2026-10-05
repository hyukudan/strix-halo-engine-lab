#pragma once
#include "strix/distributed_data_plane.hpp"
#include "strix/distributed_placement.hpp"
#include "strix/distributed_session.hpp"
#include <cstdint>
namespace strix {
struct DistributedRuntimeRequest {
    std::uint64_t connection_epoch = 0;
    std::uint64_t request_id = 0;
    std::uint64_t generation = 0;
    std::uint32_t dtype_id = 0;
};
struct DistributedRuntimeState {
    DistributedRequestState receive;
    std::uint64_t next_send_sequence = 1;
    bool failed = false;
};
struct DistributedRuntimeResult {
    std::uint64_t local_steps = 0;
    std::uint64_t remote_steps = 0;
    std::uint64_t sent_boundaries = 0;
    std::uint64_t received_boundaries = 0;
};
class DistributedExecutionBackend {
public:
    virtual ~DistributedExecutionBackend() = default;
    virtual void execute_local_step(const DistributedStepPlacement& step) = 0;
    virtual void execute_remote_step(const DistributedStepPlacement& step) = 0;
    virtual void send_boundary(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;
    [[nodiscard]] virtual DistributedBoundaryTransfer
    receive_boundary(const DistributedExecutionBoundary& boundary) = 0;
};
[[nodiscard]] DistributedRuntimeState make_distributed_runtime_state(
    const DistributedPlacementPlan& plan,
    const DistributedRuntimeRequest& request);
[[nodiscard]] DistributedRuntimeResult execute_distributed_runtime(
    const DistributedPlacementPlan& plan,
    const DistributedExecutionDecision& decision,
    const DistributedRuntimeRequest& request,
    DistributedRuntimeState& state,
    transport::DualLinkRuntimeScheduler& scheduler,
    DistributedExecutionBackend& backend,
    DistributedBoundaryFlowPolicy flow_policy = {});
}  // namespace strix
