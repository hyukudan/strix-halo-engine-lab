#include "strix/distributed_control_receiver.hpp"
#include "strix/distributed_control_server.hpp"
#include "strix/distributed_control_wire.hpp"
#include "strix/distributed_remote_control.hpp"
#include "strix/distributed_tcp_control_stream.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
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
        steps.push_back(step.step_id);
    }

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

    strix::DistributedExecutionBoundary outbound;
    outbound.boundary_id = 0x1001ULL;
    outbound.source_step = 0;
    outbound.destination_step = 1;
    outbound.source_node = 0;
    outbound.destination_node = 1;
    outbound.payload_bytes = 128;
    plan.boundaries.push_back(outbound);
    return plan;
}

strix::DistributedRuntimeRequest request() {
    return {9, 11, 2, 7};
}

strix::DistributedBoundaryTransfer transfer() {
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
        {0, 0, 128},
    };
    return value;
}

std::array<int, 2> socket_pair() {
    std::array<int, 2> fds{-1, -1};
    if (::socketpair(
            AF_UNIX,
            SOCK_STREAM,
            0,
            fds.data()) != 0) {
        throw std::runtime_error(
            "socketpair failed");
    }
    return fds;
}

}  // namespace

int main() {
    const auto fds =
        socket_pair();

    auto client_stream =
        strix::TcpDistributedControlByteStream::
            adopt_connected(
                fds[0]);
    auto server_stream =
        strix::TcpDistributedControlByteStream::
            adopt_connected(
                fds[1]);

    auto plan =
        placement();
    const auto runtime_request =
        request();

    Executor executor;
    strix::DistributedControlReceiver receiver(
        plan,
        runtime_request,
        1,
        executor);
    strix::DistributedControlServer server(
        server_stream,
        receiver);

    std::exception_ptr server_error;
    std::thread server_thread(
        [&] {
            try {
                const auto first =
                    server.serve_one();
                const auto second =
                    server.serve_one();
                if (first !=
                        strix::DistributedControlServerEvent::
                            remote_step ||
                    second !=
                        strix::DistributedControlServerEvent::
                            boundary_transfer) {
                    throw std::runtime_error(
                        "unexpected control event order");
                }
            } catch (...) {
                server_error =
                    std::current_exception();
            }
        });

    strix::FramedDistributedControlWire
        wire(
            client_stream);
    strix::SequencedDistributedNativeControlPlane
        control(
            plan,
            runtime_request,
            wire);

    control.execute_remote_step(
        plan.steps[1]);
    control.publish_boundary_transfer(
        plan.boundaries[0],
        transfer());

    server_thread.join();
    if (server_error) {
        std::rethrow_exception(
            server_error);
    }

    expect(
        executor.steps ==
            std::vector<std::uint32_t>({1}) &&
            control.next_sequence() == 2 &&
            !control.failed() &&
            !receiver.failed(),
        "sequenced framed control executes remote step and receives exact ACK end-to-end");

    expect(
        receiver.has_boundary(
            0x1001ULL),
        "boundary metadata reaches receiver over the same framed socket session");

    const auto observed =
        receiver.take_boundary(
            0x1001ULL);
    expect(
        observed.descriptor.request_id ==
                runtime_request.request_id &&
            observed.descriptor.generation ==
                runtime_request.generation &&
            observed.descriptor.boundary_id ==
                plan.boundaries[0].
                    boundary_id &&
            observed.link_plan.slices.size() ==
                1,
        "receiver retains exact boundary request and link-plan metadata");

    if (failures == 0) {
        std::cout
            << "distributed control end-to-end tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
