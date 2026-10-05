#include "strix/distributed_control_receiver.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class Executor final
    : public strix::DistributedNativeLocalExecutor {
public:
    void execute_local_step(
        const strix::DistributedStepPlacement& step) override {
        if (fail) {
            throw std::runtime_error(
                "injected receiver execution failure");
        }
        steps.push_back(
            step.step_id);
    }

    bool fail = false;
    std::vector<std::uint32_t> steps;
};

strix::DistributedPlacementPlan placement() {
    strix::DistributedPlacementPlan plan;
    plan.local_node = 0;
    plan.plan_identity = 0xabcULL;
    plan.executable = true;
    plan.has_remote_work = true;
    plan.remote_nodes = {1};
    plan.steps = {
        {0, strix::SerialOp::token_embedding, 0, false, false},
        {1, strix::SerialOp::moe_experts, 1, true, false},
        {2, strix::SerialOp::lm_head, 0, false, false},
    };

    strix::DistributedExecutionBoundary inbound;
    inbound.boundary_id = 0x1001ULL;
    inbound.source_step = 0;
    inbound.destination_step = 1;
    inbound.source_node = 0;
    inbound.destination_node = 1;
    inbound.payload_bytes = 128;

    strix::DistributedExecutionBoundary outbound;
    outbound.boundary_id = 0x1002ULL;
    outbound.source_step = 1;
    outbound.destination_step = 2;
    outbound.source_node = 1;
    outbound.destination_node = 0;
    outbound.payload_bytes = 128;

    plan.boundaries = {
        inbound,
        outbound,
    };
    return plan;
}

strix::DistributedRuntimeRequest request() {
    return {9,11,2,7};
}

strix::DistributedRemoteStepCommand command(
    std::uint64_t sequence = 1) {
    strix::DistributedRemoteStepCommand value;
    value.protocol_version = 1;
    value.placement_plan_identity = 0xabcULL;
    value.request_id = 11;
    value.generation = 2;
    value.sequence = sequence;
    value.step_id = 1;
    value.op = strix::SerialOp::moe_experts;
    value.target_node = 1;
    return value;
}

strix::DistributedBoundaryTransfer boundary_transfer() {
    strix::DistributedBoundaryTransfer value;
    value.flow.flow_id = 0x999ULL;
    value.flow.direction =
        strix::transport::LinkDirection::outbound;
    value.flow.priority =
        strix::transport::TransferPriority::normal;
    value.descriptor.protocol_version = 1;
    value.descriptor.connection_epoch = 9;
    value.descriptor.model_plan_id = 0xabcULL;
    value.descriptor.request_id = 11;
    value.descriptor.generation = 2;
    value.descriptor.boundary_id = 0x1001ULL;
    value.descriptor.sequence = 1;
    value.descriptor.shape_fingerprint = 3;
    value.descriptor.dtype_id = 7;
    value.descriptor.byte_length = 128;
    value.link_plan.kind =
        strix::transport::PlanKind::single_link;
    value.link_plan.slices = {
        {0,0,128},
    };
    return value;
}

}  // namespace

int main() {
    {
        auto plan =
            placement();
        Executor executor;
        strix::DistributedControlReceiver receiver(
            plan,
            request(),
            1,
            executor);

        expect(
            receiver.execute_remote_step(
                command()) &&
                executor.steps ==
                    std::vector<std::uint32_t>({1}) &&
                receiver.next_command_sequence() == 2 &&
                !receiver.failed(),
            "receiver validates global placement identities and executes owned step");

        auto transfer =
            boundary_transfer();
        receiver.receive_boundary_transfer(
            transfer);
        expect(
            receiver.has_boundary(
                0x1001ULL),
            "receiver queues boundary addressed to its physical node");
        const auto taken =
            receiver.take_boundary(
                0x1001ULL);
        expect(
            taken.descriptor.boundary_id ==
                    0x1001ULL &&
                !receiver.has_boundary(
                    0x1001ULL),
            "receiver consumes pending boundary metadata exactly once");
    }

    {
        auto plan =
            placement();
        Executor executor;
        strix::DistributedControlReceiver receiver(
            plan,
            request(),
            1,
            executor);

        auto stale =
            command(2);
        bool rejected = false;
        try {
            static_cast<void>(
                receiver.execute_remote_step(
                    stale));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected &&
                receiver.failed(),
            "out-of-sequence remote command poisons receiver");
    }

    {
        auto plan =
            placement();
        Executor executor;
        executor.fail = true;
        strix::DistributedControlReceiver receiver(
            plan,
            request(),
            1,
            executor);

        bool failed = false;
        try {
            static_cast<void>(
                receiver.execute_remote_step(
                    command()));
        } catch (const std::runtime_error&) {
            failed = true;
        }
        expect(
            failed &&
                receiver.failed(),
            "local remote-step execution failure poisons receiver");
    }

    {
        auto plan =
            placement();
        Executor executor;
        strix::DistributedControlReceiver receiver(
            plan,
            request(),
            1,
            executor);

        auto transfer =
            boundary_transfer();
        receiver.receive_boundary_transfer(
            transfer);
        bool duplicate = false;
        try {
            receiver.receive_boundary_transfer(
                transfer);
        } catch (const std::logic_error&) {
            duplicate = true;
        }
        expect(
            duplicate &&
                receiver.failed(),
            "duplicate pending boundary metadata poisons receiver");
    }

    if (failures == 0) {
        std::cout
            << "distributed control receiver tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
