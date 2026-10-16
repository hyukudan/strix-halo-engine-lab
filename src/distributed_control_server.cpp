#include "strix/distributed_control_server.hpp"

#include <stdexcept>

namespace strix {

DistributedControlServer::DistributedControlServer(
    DistributedControlByteStream& stream,
    DistributedControlServerHandler& handler,
    DistributedControlWireLimits limits)
    : stream_(&stream),
      handler_(&handler),
      limits_(limits) {
    if (limits_.protocol_version == 0 ||
        limits_.maximum_frame_bytes < 16 ||
        limits_.maximum_slices == 0) {
        throw std::invalid_argument(
            "distributed control server limits are invalid");
    }
}

DistributedControlServerEvent
DistributedControlServer::serve_one() {
    const auto bytes =
        stream_->receive();
    const auto kind =
        inspect_distributed_control_frame_kind(
            bytes,
            limits_);

    switch (kind) {
    case DistributedControlFrameKind::
            remote_step_command: {
        const auto command =
            decode_distributed_remote_step_command(
                bytes,
                limits_);

        DistributedRemoteStepAck ack;
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

        try {
            ack.success =
                handler_->execute_remote_step(
                    command);
        } catch (...) {
            ack.success = false;
            const auto encoded =
                encode_distributed_remote_step_ack(
                    ack,
                    limits_);
            stream_->send(encoded);
            throw;
        }

        const auto encoded =
            encode_distributed_remote_step_ack(
                ack,
                limits_);
        stream_->send(encoded);
        return DistributedControlServerEvent::
            remote_step;
    }

    case DistributedControlFrameKind::
            boundary_transfer: {
        const auto transfer =
            decode_distributed_boundary_transfer(
                bytes,
                limits_);
        handler_->receive_boundary_transfer(
            transfer);
        return DistributedControlServerEvent::
            boundary_transfer;
    }

    case DistributedControlFrameKind::
            remote_step_ack:
        throw std::invalid_argument(
            "distributed control server cannot consume unsolicited ACK frame");
    case DistributedControlFrameKind::token_transaction:
        throw std::invalid_argument("token transactions require their dedicated control channel");
    case DistributedControlFrameKind::direct_gpu_stream:
        throw std::invalid_argument(
            "GPU transport control requires its dedicated experimental channel");
    }

    throw std::logic_error(
        "distributed control server frame dispatch fell through");
}

}  // namespace strix
