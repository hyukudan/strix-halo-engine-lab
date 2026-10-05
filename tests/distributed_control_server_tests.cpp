#include "strix/distributed_control_server.hpp"

#include <deque>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class Stream final
    : public strix::DistributedControlByteStream {
public:
    void send(
        std::span<const std::byte> frame) override {
        outgoing.emplace_back(
            frame.begin(),
            frame.end());
    }

    std::vector<std::byte> receive() override {
        if (incoming.empty()) {
            throw std::runtime_error(
                "server stream has no incoming frame");
        }
        auto result =
            std::move(
                incoming.front());
        incoming.pop_front();
        return result;
    }

    std::deque<std::vector<std::byte>>
        incoming;
    std::vector<std::vector<std::byte>>
        outgoing;
};

class Handler final
    : public strix::DistributedControlServerHandler {
public:
    bool execute_remote_step(
        const strix::DistributedRemoteStepCommand& command) override {
        commands.push_back(command);
        if (throw_command) {
            throw std::runtime_error(
                "injected remote-step failure");
        }
        return command_success;
    }

    void receive_boundary_transfer(
        const strix::DistributedBoundaryTransfer& transfer) override {
        boundaries.push_back(
            transfer);
    }

    bool command_success = true;
    bool throw_command = false;
    std::vector<
        strix::DistributedRemoteStepCommand>
        commands;
    std::vector<
        strix::DistributedBoundaryTransfer>
        boundaries;
};

strix::DistributedRemoteStepCommand command() {
    strix::DistributedRemoteStepCommand value;
    value.protocol_version = 1;
    value.placement_plan_identity = 2;
    value.request_id = 3;
    value.generation = 4;
    value.sequence = 5;
    value.step_id = 6;
    value.op =
        strix::SerialOp::moe_experts;
    value.target_node = 1;
    return value;
}

strix::DistributedBoundaryTransfer transfer() {
    strix::DistributedBoundaryTransfer value;
    value.flow.flow_id = 9;
    value.flow.direction =
        strix::transport::LinkDirection::inbound;
    value.flow.priority =
        strix::transport::TransferPriority::normal;
    value.descriptor.connection_epoch = 10;
    value.descriptor.model_plan_id = 11;
    value.descriptor.request_id = 12;
    value.descriptor.generation = 13;
    value.descriptor.boundary_id = 14;
    value.descriptor.sequence = 15;
    value.descriptor.shape_fingerprint = 16;
    value.descriptor.dtype_id = 17;
    value.descriptor.byte_length = 128;
    value.link_plan.kind =
        strix::transport::PlanKind::striped;
    value.link_plan.slices = {
        {0,0,64},
        {1,64,64},
    };
    return value;
}

}  // namespace

int main() {
    {
        Stream stream;
        Handler handler;
        strix::DistributedControlServer
            server(
                stream,
                handler);

        const auto sent_command =
            command();
        stream.incoming.push_back(
            strix::encode_distributed_remote_step_command(
                sent_command));

        const auto event =
            server.serve_one();
        expect(
            event ==
                    strix::DistributedControlServerEvent::
                        remote_step &&
                handler.commands.size() == 1 &&
                stream.outgoing.size() == 1,
            "control server dispatches remote-step command and sends ACK");

        const auto ack =
            strix::decode_distributed_remote_step_ack(
                stream.outgoing.front());
        expect(
            ack.success &&
                ack.sequence ==
                    sent_command.sequence &&
                ack.step_id ==
                    sent_command.step_id &&
                ack.target_node ==
                    sent_command.target_node,
            "control server ACK mirrors exact command identities");
    }

    {
        Stream stream;
        Handler handler;
        strix::DistributedControlServer
            server(
                stream,
                handler);

        const auto sent_transfer =
            transfer();
        stream.incoming.push_back(
            strix::encode_distributed_boundary_transfer(
                sent_transfer));

        const auto event =
            server.serve_one();
        expect(
            event ==
                    strix::DistributedControlServerEvent::
                        boundary_transfer &&
                handler.boundaries.size() == 1 &&
                handler.boundaries.front().
                    link_plan.slices.size() == 2 &&
                stream.outgoing.empty(),
            "control server delivers boundary metadata without spurious ACK");
    }

    {
        Stream stream;
        Handler handler;
        handler.command_success = false;
        strix::DistributedControlServer
            server(
                stream,
                handler);

        stream.incoming.push_back(
            strix::encode_distributed_remote_step_command(
                command()));
        static_cast<void>(
            server.serve_one());
        const auto ack =
            strix::decode_distributed_remote_step_ack(
                stream.outgoing.front());
        expect(
            !ack.success,
            "remote handler negative result is explicit negative ACK");
    }

    {
        Stream stream;
        Handler handler;
        handler.throw_command = true;
        strix::DistributedControlServer
            server(
                stream,
                handler);

        stream.incoming.push_back(
            strix::encode_distributed_remote_step_command(
                command()));
        bool threw = false;
        try {
            static_cast<void>(
                server.serve_one());
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expect(
            threw &&
                stream.outgoing.size() == 1 &&
                !strix::decode_distributed_remote_step_ack(
                     stream.outgoing.front()).
                     success,
            "remote execution exception sends negative ACK before surfacing locally");
    }

    if (failures == 0) {
        std::cout
            << "distributed control server tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
