#include "strix/distributed_control_wire.hpp"

#include <deque>
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

class Loopback final
    : public strix::DistributedControlByteStream {
public:
    void send(
        std::span<const std::byte> frame) override {
        sent.emplace_back(
            frame.begin(),
            frame.end());
        if (auto_ack) {
            try {
                const auto command =
                    strix::decode_distributed_remote_step_command(
                        frame);
                strix::DistributedRemoteStepAck ack;
                ack.protocol_version =
                    command.protocol_version;
                ack.placement_plan_identity =
                    command.placement_plan_identity;
                ack.request_id =
                    command.request_id;
                ack.generation =
                    command.generation;
                ack.sequence =
                    command.sequence;
                ack.step_id =
                    command.step_id;
                ack.target_node =
                    command.target_node;
                ack.success = true;
                incoming.push_back(
                    strix::encode_distributed_remote_step_ack(
                        ack));
            } catch (const std::invalid_argument&) {
            }
        }
    }

    std::vector<std::byte> receive() override {
        if (incoming.empty()) {
            throw std::runtime_error(
                "loopback receive queue empty");
        }
        auto value =
            std::move(
                incoming.front());
        incoming.pop_front();
        return value;
    }

    bool auto_ack = true;
    std::vector<std::vector<std::byte>> sent;
    std::deque<std::vector<std::byte>> incoming;
};

strix::DistributedBoundaryTransfer striped_transfer() {
    strix::DistributedBoundaryTransfer value;
    value.flow.flow_id = 0x1111ULL;
    value.flow.direction =
        strix::transport::LinkDirection::outbound;
    value.flow.priority =
        strix::transport::TransferPriority::bulk;
    value.descriptor.protocol_version = 1;
    value.descriptor.connection_epoch = 9;
    value.descriptor.model_plan_id = 10;
    value.descriptor.request_id = 11;
    value.descriptor.generation = 12;
    value.descriptor.boundary_id = 13;
    value.descriptor.sequence = 14;
    value.descriptor.shape_fingerprint = 15;
    value.descriptor.dtype_id = 16;
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
    strix::DistributedRemoteStepCommand command;
    command.protocol_version = 1;
    command.placement_plan_identity = 2;
    command.request_id = 3;
    command.generation = 4;
    command.sequence = 5;
    command.step_id = 6;
    command.op = strix::SerialOp::moe_experts;
    command.target_node = 1;

    const auto encoded_command =
        strix::encode_distributed_remote_step_command(
            command);
    const auto decoded_command =
        strix::decode_distributed_remote_step_command(
            encoded_command);
    expect(
        decoded_command.sequence ==
                command.sequence &&
            decoded_command.step_id ==
                command.step_id &&
            decoded_command.op ==
                command.op,
        "remote-step command round-trips through explicit wire format");

    strix::DistributedRemoteStepAck ack;
    ack.protocol_version = 1;
    ack.placement_plan_identity = 2;
    ack.request_id = 3;
    ack.generation = 4;
    ack.sequence = 5;
    ack.step_id = 6;
    ack.target_node = 1;
    ack.success = true;
    const auto encoded_ack =
        strix::encode_distributed_remote_step_ack(
            ack);
    const auto decoded_ack =
        strix::decode_distributed_remote_step_ack(
            encoded_ack);
    expect(
        decoded_ack.success &&
            decoded_ack.sequence == 5 &&
            decoded_ack.target_node == 1,
        "remote-step ACK round-trips through explicit wire format");

    const auto transfer =
        striped_transfer();
    const auto decoded_transfer =
        strix::decode_distributed_boundary_transfer(
            strix::encode_distributed_boundary_transfer(
                transfer));
    expect(
        decoded_transfer.flow.flow_id ==
                transfer.flow.flow_id &&
            decoded_transfer.descriptor.boundary_id ==
                transfer.descriptor.boundary_id &&
            decoded_transfer.link_plan.kind ==
                strix::transport::PlanKind::striped &&
            decoded_transfer.link_plan.slices.size() == 2 &&
            decoded_transfer.link_plan.slices[1].byte_offset == 64,
        "striped boundary metadata round-trips with exact slices");

    Loopback stream;
    strix::FramedDistributedControlWire wire(
        stream);
    const auto wire_ack =
        wire.execute_remote_step(
            command);
    expect(
        wire_ack.success &&
            wire_ack.sequence ==
                command.sequence &&
            stream.sent.size() == 1,
        "framed wire exchanges command and ACK through byte-stream seam");

    stream.auto_ack = false;
    wire.publish_boundary_transfer(
        strix::DistributedExecutionBoundary{},
        transfer);
    expect(
        stream.sent.size() == 2,
        "framed wire publishes boundary metadata frame");
    stream.incoming.push_back(
        stream.sent.back());
    const auto received =
        wire.await_boundary_transfer(
            strix::DistributedExecutionBoundary{});
    expect(
        received.link_plan.slices.size() == 2 &&
            received.descriptor.byte_length == 128,
        "framed wire receives boundary metadata frame");

    auto corrupt =
        encoded_command;
    corrupt[0] =
        static_cast<std::byte>(0);
    bool corrupt_rejected = false;
    try {
        static_cast<void>(
            strix::decode_distributed_remote_step_command(
                corrupt));
    } catch (const std::invalid_argument&) {
        corrupt_rejected = true;
    }
    expect(
        corrupt_rejected,
        "control wire rejects invalid frame magic");

    auto trailing =
        encoded_ack;
    trailing.push_back(
        static_cast<std::byte>(0));
    bool trailing_rejected = false;
    try {
        static_cast<void>(
            strix::decode_distributed_remote_step_ack(
                trailing));
    } catch (const std::invalid_argument&) {
        trailing_rejected = true;
    }
    expect(
        trailing_rejected,
        "control wire rejects frame length/trailing-byte mismatch");

    if (failures == 0) {
        std::cout
            << "distributed control wire tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
