#include "strix/distributed_control_wire.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t magic =
    0x53544357U;
constexpr std::size_t header_bytes =
    16;

class Writer {
public:
    void u16(std::uint16_t value) {
        bytes_.push_back(
            static_cast<std::byte>(
                value & 0xffU));
        bytes_.push_back(
            static_cast<std::byte>(
                (value >> 8U) & 0xffU));
    }

    void u32(std::uint32_t value) {
        for (unsigned shift = 0;
             shift < 32;
             shift += 8) {
            bytes_.push_back(
                static_cast<std::byte>(
                    (value >> shift) &
                    0xffU));
        }
    }

    void u64(std::uint64_t value) {
        for (unsigned shift = 0;
             shift < 64;
             shift += 8) {
            bytes_.push_back(
                static_cast<std::byte>(
                    (value >> shift) &
                    0xffULL));
        }
    }

    void boolean(bool value) {
        u32(value ? 1U : 0U);
    }

    [[nodiscard]] std::vector<std::byte>
    take() {
        return std::move(bytes_);
    }

private:
    std::vector<std::byte> bytes_;
};

class Reader {
public:
    explicit Reader(
        std::span<const std::byte> bytes)
        : bytes_(bytes) {}

    [[nodiscard]] std::uint16_t u16() {
        require(2);
        const auto low =
            static_cast<std::uint32_t>(
                std::to_integer<std::uint8_t>(
                    bytes_[offset_]));
        const auto high =
            static_cast<std::uint32_t>(
                std::to_integer<std::uint8_t>(
                    bytes_[offset_ + 1U]));
        const auto value =
            static_cast<std::uint16_t>(
                low |
                (high << 8U));
        offset_ += 2;
        return value;
    }

    [[nodiscard]] std::uint32_t u32() {
        require(4);
        std::uint32_t value = 0;
        for (unsigned shift = 0;
             shift < 32;
             shift += 8) {
            value |=
                static_cast<std::uint32_t>(
                    std::to_integer<std::uint8_t>(
                        bytes_[offset_++])) <<
                shift;
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        require(8);
        std::uint64_t value = 0;
        for (unsigned shift = 0;
             shift < 64;
             shift += 8) {
            value |=
                static_cast<std::uint64_t>(
                    std::to_integer<std::uint8_t>(
                        bytes_[offset_++])) <<
                shift;
        }
        return value;
    }

    [[nodiscard]] bool boolean() {
        const auto value =
            u32();
        if (value > 1U) {
            throw std::invalid_argument(
                "control wire boolean is not canonical");
        }
        return value != 0;
    }

    [[nodiscard]] bool empty() const noexcept {
        return offset_ ==
            bytes_.size();
    }

private:
    void require(
        std::size_t bytes) {
        if (bytes >
            bytes_.size() -
                offset_) {
            throw std::invalid_argument(
                "control wire frame is truncated");
        }
    }

    std::span<const std::byte> bytes_;
    std::size_t offset_ = 0;
};

void validate_limits(
    const DistributedControlWireLimits& limits) {
    if (limits.protocol_version == 0 ||
        limits.maximum_frame_bytes <
            header_bytes ||
        limits.maximum_slices == 0) {
        throw std::invalid_argument(
            "distributed control wire limits are invalid");
    }
}

std::vector<std::byte> frame(
    DistributedControlFrameKind kind,
    std::vector<std::byte> payload,
    DistributedControlWireLimits limits) {
    validate_limits(limits);
    if (payload.size() >
        limits.maximum_frame_bytes -
            header_bytes ||
        payload.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "distributed control payload exceeds frame limit");
    }

    Writer writer;
    writer.u32(magic);
    writer.u32(
        limits.protocol_version);
    writer.u16(
        static_cast<std::uint16_t>(
            kind));
    writer.u16(0);
    writer.u32(
        static_cast<std::uint32_t>(
            payload.size()));
    auto result =
        writer.take();
    result.insert(
        result.end(),
        payload.begin(),
        payload.end());
    return result;
}

std::span<const std::byte> payload_of(
    std::span<const std::byte> bytes,
    DistributedControlFrameKind expected,
    DistributedControlWireLimits limits) {
    validate_limits(limits);
    if (bytes.size() <
            header_bytes ||
        bytes.size() >
            limits.maximum_frame_bytes) {
        throw std::invalid_argument(
            "distributed control frame size is invalid");
    }

    Reader reader(bytes);
    if (reader.u32() != magic ||
        reader.u32() !=
            limits.protocol_version ||
        reader.u16() !=
            static_cast<std::uint16_t>(
                expected)) {
        throw std::invalid_argument(
            "distributed control frame header mismatch");
    }
    if (reader.u16() != 0) {
        throw std::invalid_argument(
            "distributed control frame reserved bits are non-zero");
    }
    const auto payload_bytes =
        reader.u32();
    if (payload_bytes !=
        bytes.size() -
            header_bytes) {
        throw std::invalid_argument(
            "distributed control frame payload length mismatch");
    }
    return bytes.subspan(
        header_bytes);
}

void require_consumed(
    const Reader& reader) {
    if (!reader.empty()) {
        throw std::invalid_argument(
            "distributed control frame has trailing bytes");
    }
}

void encode_descriptor(
    Writer& writer,
    const transport::TransferDescriptor& d) {
    writer.u32(d.protocol_version);
    writer.u64(d.connection_epoch);
    writer.u64(d.model_plan_id);
    writer.u64(d.request_id);
    writer.u64(d.generation);
    writer.u64(d.boundary_id);
    writer.u64(d.sequence);
    writer.u64(d.shape_fingerprint);
    writer.u32(d.dtype_id);
    writer.u64(d.byte_length);
}

transport::TransferDescriptor
decode_descriptor(
    Reader& reader) {
    transport::TransferDescriptor d;
    d.protocol_version = reader.u32();
    d.connection_epoch = reader.u64();
    d.model_plan_id = reader.u64();
    d.request_id = reader.u64();
    d.generation = reader.u64();
    d.boundary_id = reader.u64();
    d.sequence = reader.u64();
    d.shape_fingerprint = reader.u64();
    d.dtype_id = reader.u32();
    d.byte_length = reader.u64();
    return d;
}

}  // namespace

DistributedControlFrameKind
inspect_distributed_control_frame_kind(
    std::span<const std::byte> bytes,
    DistributedControlWireLimits limits) {
    validate_limits(limits);
    if (bytes.size() <
            header_bytes ||
        bytes.size() >
            limits.maximum_frame_bytes) {
        throw std::invalid_argument(
            "distributed control frame size is invalid");
    }

    Reader reader(bytes);
    if (reader.u32() != magic ||
        reader.u32() !=
            limits.protocol_version) {
        throw std::invalid_argument(
            "distributed control frame header mismatch");
    }
    const auto raw_kind =
        reader.u16();
    static_cast<void>(
        reader.u16());
    const auto payload_bytes =
        reader.u32();
    if (payload_bytes !=
        bytes.size() -
            header_bytes) {
        throw std::invalid_argument(
            "distributed control frame payload length mismatch");
    }

    switch (
        static_cast<
            DistributedControlFrameKind>(
                raw_kind)) {
    case DistributedControlFrameKind::
            remote_step_command:
    case DistributedControlFrameKind::
            remote_step_ack:
    case DistributedControlFrameKind::
            boundary_transfer:
        return static_cast<
            DistributedControlFrameKind>(
                raw_kind);
    }
    throw std::invalid_argument(
        "distributed control frame kind is unknown");
}

std::vector<std::byte>
encode_distributed_remote_step_command(
    const DistributedRemoteStepCommand& command,
    DistributedControlWireLimits limits) {
    Writer payload;
    payload.u32(command.protocol_version);
    payload.u64(command.placement_plan_identity);
    payload.u64(command.request_id);
    payload.u64(command.generation);
    payload.u64(command.sequence);
    payload.u32(command.step_id);
    payload.u32(
        static_cast<std::uint32_t>(
            command.op));
    payload.u32(command.target_node);
    return frame(
        DistributedControlFrameKind::
            remote_step_command,
        payload.take(),
        limits);
}

DistributedRemoteStepCommand
decode_distributed_remote_step_command(
    std::span<const std::byte> bytes,
    DistributedControlWireLimits limits) {
    Reader reader(
        payload_of(
            bytes,
            DistributedControlFrameKind::
                remote_step_command,
            limits));
    DistributedRemoteStepCommand command;
    command.protocol_version = reader.u32();
    command.placement_plan_identity = reader.u64();
    command.request_id = reader.u64();
    command.generation = reader.u64();
    command.sequence = reader.u64();
    command.step_id = reader.u32();
    command.op =
        static_cast<SerialOp>(
            reader.u32());
    command.target_node = reader.u32();
    require_consumed(reader);
    return command;
}

std::vector<std::byte>
encode_distributed_remote_step_ack(
    const DistributedRemoteStepAck& ack,
    DistributedControlWireLimits limits) {
    Writer payload;
    payload.u32(ack.protocol_version);
    payload.u64(ack.placement_plan_identity);
    payload.u64(ack.request_id);
    payload.u64(ack.generation);
    payload.u64(ack.sequence);
    payload.u32(ack.step_id);
    payload.u32(ack.target_node);
    payload.boolean(ack.success);
    return frame(
        DistributedControlFrameKind::
            remote_step_ack,
        payload.take(),
        limits);
}

DistributedRemoteStepAck
decode_distributed_remote_step_ack(
    std::span<const std::byte> bytes,
    DistributedControlWireLimits limits) {
    Reader reader(
        payload_of(
            bytes,
            DistributedControlFrameKind::
                remote_step_ack,
            limits));
    DistributedRemoteStepAck ack;
    ack.protocol_version = reader.u32();
    ack.placement_plan_identity = reader.u64();
    ack.request_id = reader.u64();
    ack.generation = reader.u64();
    ack.sequence = reader.u64();
    ack.step_id = reader.u32();
    ack.target_node = reader.u32();
    ack.success = reader.boolean();
    require_consumed(reader);
    return ack;
}

std::vector<std::byte>
encode_distributed_boundary_transfer(
    const DistributedBoundaryTransfer& transfer,
    DistributedControlWireLimits limits) {
    validate_limits(limits);
    if (transfer.flow.flow_id == 0 ||
        transfer.descriptor.byte_length == 0 ||
        transfer.link_plan.slices.empty() ||
        transfer.link_plan.slices.size() >
            limits.maximum_slices) {
        throw std::invalid_argument(
            "distributed boundary transfer metadata is invalid");
    }

    Writer payload;
    payload.u64(transfer.flow.flow_id);
    payload.u32(
        static_cast<std::uint32_t>(
            transfer.flow.direction));
    payload.u32(
        static_cast<std::uint32_t>(
            transfer.flow.priority));
    encode_descriptor(
        payload,
        transfer.descriptor);
    payload.u32(
        static_cast<std::uint32_t>(
            transfer.link_plan.kind));
    payload.u32(
        static_cast<std::uint32_t>(
            transfer.link_plan.slices.size()));
    for (const auto& slice :
         transfer.link_plan.slices) {
        if (slice.link_index >
            std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "distributed control link index exceeds uint32");
        }
        payload.u32(
            static_cast<std::uint32_t>(
                slice.link_index));
        payload.u64(slice.byte_offset);
        payload.u64(slice.byte_length);
    }
    return frame(
        DistributedControlFrameKind::
            boundary_transfer,
        payload.take(),
        limits);
}

DistributedBoundaryTransfer
decode_distributed_boundary_transfer(
    std::span<const std::byte> bytes,
    DistributedControlWireLimits limits) {
    Reader reader(
        payload_of(
            bytes,
            DistributedControlFrameKind::
                boundary_transfer,
            limits));
    DistributedBoundaryTransfer transfer;
    transfer.flow.flow_id = reader.u64();
    transfer.flow.direction =
        static_cast<transport::LinkDirection>(
            reader.u32());
    transfer.flow.priority =
        static_cast<transport::TransferPriority>(
            reader.u32());
    transfer.descriptor =
        decode_descriptor(reader);
    transfer.link_plan.kind =
        static_cast<transport::PlanKind>(
            reader.u32());
    const auto slices =
        reader.u32();
    if (slices == 0 ||
        slices >
            limits.maximum_slices) {
        throw std::invalid_argument(
            "distributed control slice count is invalid");
    }
    transfer.link_plan.slices.reserve(
        slices);
    for (std::uint32_t index = 0;
         index < slices;
         ++index) {
        transport::LinkSlice slice;
        slice.link_index =
            reader.u32();
        slice.byte_offset =
            reader.u64();
        slice.byte_length =
            reader.u64();
        transfer.link_plan.slices.push_back(
            slice);
    }
    require_consumed(reader);
    return transfer;
}

FramedDistributedControlWire::FramedDistributedControlWire(
    DistributedControlByteStream& stream,
    DistributedControlWireLimits limits)
    : stream_(&stream),
      limits_(limits) {
    validate_limits(limits_);
}

DistributedRemoteStepAck
FramedDistributedControlWire::execute_remote_step(
    const DistributedRemoteStepCommand& command) {
    const auto encoded =
        encode_distributed_remote_step_command(
            command,
            limits_);
    stream_->send(encoded);
    const auto response =
        stream_->receive();
    return decode_distributed_remote_step_ack(
        response,
        limits_);
}

void FramedDistributedControlWire::publish_boundary_transfer(
    const DistributedExecutionBoundary&,
    const DistributedBoundaryTransfer& transfer) {
    const auto encoded =
        encode_distributed_boundary_transfer(
            transfer,
            limits_);
    stream_->send(encoded);
}

DistributedBoundaryTransfer
FramedDistributedControlWire::await_boundary_transfer(
    const DistributedExecutionBoundary&) {
    const auto frame_bytes =
        stream_->receive();
    return decode_distributed_boundary_transfer(
        frame_bytes,
        limits_);
}

}  // namespace strix
