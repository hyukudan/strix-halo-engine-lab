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
template<class F> bool rejected(F function) { try { function(); } catch (const std::invalid_argument&) { return true; } return false; }

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
    value.native_slices = {{21, 101}, {39, 203}};
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
    std::vector<std::byte> golden_ack(64);
    golden_ack[0] = std::byte{0x57}; golden_ack[1] = std::byte{0x43}; golden_ack[2] = std::byte{0x54}; golden_ack[3] = std::byte{0x53};
    golden_ack[4] = std::byte{2}; golden_ack[8] = std::byte{2}; golden_ack[12] = std::byte{48}; golden_ack[16] = std::byte{1};
    golden_ack[20] = std::byte{2}; golden_ack[28] = std::byte{3}; golden_ack[36] = std::byte{4}; golden_ack[44] = std::byte{5};
    golden_ack[52] = std::byte{6}; golden_ack[56] = std::byte{1}; golden_ack[60] = std::byte{1};
    expect(encoded_ack == golden_ack, "native extension preserves exact historical v2 ACK bytes");
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
            decoded_transfer.link_plan.slices[1].byte_offset == 64 &&
            decoded_transfer.descriptor.connection_epoch == 9 &&
            decoded_transfer.descriptor.sequence == 14 &&
            decoded_transfer.native_slices.size() == 2 &&
            decoded_transfer.native_slices[0].connection_epoch == 21 &&
            decoded_transfer.native_slices[0].sequence == 101 &&
            decoded_transfer.native_slices[1].connection_epoch == 39 &&
            decoded_transfer.native_slices[1].sequence == 203,
        "striped boundary metadata round-trips with exact slices");

    auto legacy_frame = strix::encode_distributed_boundary_transfer(transfer);
    legacy_frame[4] = static_cast<std::byte>(1);
    bool legacy_rejected = false;
    try {
        static_cast<void>(strix::decode_distributed_boundary_transfer(legacy_frame));
    } catch (const std::invalid_argument&) { legacy_rejected = true; }
    expect(legacy_rejected, "wire v2 rejects v1 peers without interpreting old slice framing");

    bool downgrade_rejected = false;
    auto legacy_limits = strix::DistributedControlWireLimits{};
    legacy_limits.protocol_version = 1;
    try {
        static_cast<void>(strix::encode_distributed_boundary_transfer(transfer, legacy_limits));
    } catch (const std::invalid_argument&) { downgrade_rejected = true; }
    expect(downgrade_rejected, "wire v2 does not silently downgrade physical identity metadata");

    auto bad_native_count = transfer;
    bad_native_count.native_slices.pop_back();
    bool count_rejected = false;
    try {
        static_cast<void>(strix::encode_distributed_boundary_transfer(bad_native_count));
    } catch (const std::invalid_argument&) { count_rejected = true; }
    expect(count_rejected, "wire rejects incomplete physical slice identity coverage");

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

    strix::DistributedDirectGpuControlFrame native;
    native.physical_link_id = "synthetic-physical-link"; native.local_endpoint_identity = "synthetic-a"; native.peer_endpoint_identity = "synthetic-b";
    native.descriptor = transfer.descriptor;
    native.commit = {1, strix::transport::gpu_commit_word, 9, 14, 128, 1, 2};
    const auto native_bytes = strix::encode_distributed_direct_gpu_control(native);
    const auto native_decoded = strix::decode_distributed_direct_gpu_control(native_bytes);
    expect(strix::inspect_distributed_control_frame_kind(native_bytes) == strix::DistributedControlFrameKind::direct_gpu_stream &&
           native_decoded.descriptor.request_id == 11 && native_decoded.descriptor.model_plan_id == 10 && native_decoded.descriptor.generation == 12 &&
           native_decoded.descriptor.sequence == 14 && native_decoded.descriptor.byte_length == 128 && native_decoded.commit.stripe_index == 1,
           "explicit native kind4/v1 payload preserves full execution descriptor and terminal stripe record");
    expect(rejected([&] { static_cast<void>(strix::decode_distributed_boundary_transfer(native_bytes)); }), "legacy boundary decoder rejects unsupported native extension without downgrade");
    for (const auto kind : {strix::DistributedDirectGpuControlKind::receive_ready, strix::DistributedDirectGpuControlKind::received, strix::DistributedDirectGpuControlKind::consumed}) {
        auto message = native; message.kind = kind; message.commit = {};
        expect(strix::decode_distributed_direct_gpu_control(strix::encode_distributed_direct_gpu_control(message)).kind == kind, "native ready/received/consumer credit remain distinct canonical kinds");
        message.commit = native.commit;
        expect(rejected([&] { static_cast<void>(strix::encode_distributed_direct_gpu_control(message)); }), "native ACK cannot carry forged producer commit");
    }
    for (const auto index : {4U, 10U, 16U, 20U, 28U}) {
        auto message = native_bytes; message[index] = index == 28 ? std::byte{0} : std::byte{99};
        expect(rejected([&] { static_cast<void>(strix::decode_distributed_direct_gpu_control(message)); }), "native control rejects outer/inner version, reserved, unsupported kind or noncanonical identity");
    }
    auto invalid_native = native; invalid_native.contract_version = 2;
    expect(rejected([&] { static_cast<void>(strix::encode_distributed_direct_gpu_control(invalid_native)); }), "unsupported native payload version cannot be emitted");
    invalid_native = native; invalid_native.physical_link_id.assign(257, 'x');
    expect(rejected([&] { static_cast<void>(strix::encode_distributed_direct_gpu_control(invalid_native)); }), "native identity length is bounded independently of total frame cap");
    invalid_native = native; ++invalid_native.commit.sequence;
    expect(rejected([&] { static_cast<void>(strix::encode_distributed_direct_gpu_control(invalid_native)); }), "stale terminal record sequence cannot be published as native producer release");
    expect(rejected([&] { static_cast<void>(strix::encode_distributed_direct_gpu_control(native, {1, 2048, 1})); }) &&
           rejected([&] { static_cast<void>(strix::encode_distributed_direct_gpu_control(native, {2, 16, 1})); }), "native extension rejects wire downgrade and insufficient bounded frame admission");

    if (failures == 0) {
        std::cout
            << "distributed control wire tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
