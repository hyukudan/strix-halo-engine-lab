#include "strix/distributed_runtime.hpp"
#include <limits>
#include <stdexcept>
namespace strix {
namespace {
void validate_request(
    const DistributedPlacementPlan& plan,
    const DistributedRuntimeRequest& request,
    const DistributedRuntimeState& state) {
    if (plan.plan_identity == 0 || !plan.executable ||
        request.connection_epoch == 0 || request.request_id == 0 ||
        request.generation == 0 || request.dtype_id == 0 ||
        state.failed || state.receive.failed ||
        state.receive.placement_plan_identity != plan.plan_identity ||
        state.receive.request_id != request.request_id ||
        state.receive.generation != request.generation ||
        state.next_send_sequence == 0) {
        throw std::invalid_argument("distributed runtime request/state is incompatible");
    }
}
}  // namespace
DistributedRuntimeState make_distributed_runtime_state(
    const DistributedPlacementPlan& plan,
    const DistributedRuntimeRequest& request) {
    if (plan.plan_identity == 0 || !plan.executable ||
        request.connection_epoch == 0 || request.request_id == 0 ||
        request.generation == 0 || request.dtype_id == 0) {
        throw std::invalid_argument("distributed runtime state requires executable plan and request identities");
    }
    DistributedRuntimeState state;
    state.receive.placement_plan_identity = plan.plan_identity;
    state.receive.request_id = request.request_id;
    state.receive.generation = request.generation;
    state.receive.next_receive_sequence = 1;
    state.next_send_sequence = 1;
    return state;
}
DistributedRuntimeResult execute_distributed_runtime(
    const DistributedPlacementPlan& plan,
    const DistributedExecutionDecision& decision,
    const DistributedRuntimeRequest& request,
    DistributedRuntimeState& state,
    transport::DualLinkRuntimeScheduler& scheduler,
    DistributedExecutionBackend& backend,
    DistributedBoundaryFlowPolicy flow_policy) {
    validate_request(plan, request, state);
    if (decision.mode != DistributedExecutionMode::distributed ||
        !decision.missing_remote_nodes.empty()) {
        throw std::invalid_argument("distributed runtime requires complete distributed execution decision");
    }
    DistributedRuntimeResult result;
    try {
        for (const auto& step : plan.steps) {
            if (step.placement_ambiguous) {
                throw std::logic_error("distributed runtime encountered ambiguous step placement");
            }
            for (const auto& boundary : plan.boundaries) {
                if (boundary.destination_step != step.step_id ||
                    boundary.destination_node != plan.local_node) continue;
                auto observed = backend.receive_boundary(boundary);
                validate_distributed_boundary_transfer_receive(
                    state.receive, plan, boundary, observed, plan.local_node, flow_policy);
                ++result.received_boundaries;
            }
            if (step.remote) {
                backend.execute_remote_step(step);
                ++result.remote_steps;
            } else {
                backend.execute_local_step(step);
                ++result.local_steps;
            }
            for (const auto& boundary : plan.boundaries) {
                if (boundary.source_step != step.step_id ||
                    boundary.source_node != plan.local_node) continue;
                if (state.next_send_sequence == std::numeric_limits<std::uint64_t>::max()) {
                    throw std::overflow_error("distributed send sequence exhausted");
                }
                auto transfer = plan_distributed_boundary_transfer(
                    plan, boundary, scheduler,
                    request.connection_epoch, request.request_id,
                    request.generation, state.next_send_sequence,
                    request.dtype_id, flow_policy);
                backend.send_boundary(boundary, transfer);
                ++state.next_send_sequence;
                ++result.sent_boundaries;
            }
        }
    } catch (...) {
        state.failed = true;
        state.receive.failed = true;
        throw;
    }
    return result;
}
}  // namespace strix
