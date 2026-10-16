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
    void text(const std::string& value) {
        if (value.empty() || value.size() > 256) throw std::invalid_argument("direct GPU control identity is empty or excessive");
        u32(static_cast<std::uint32_t>(value.size()));
        for (unsigned char character : value) {
            if (character < 32 || character > 126) throw std::invalid_argument("direct GPU control identity is not canonical ASCII");
            bytes_.push_back(static_cast<std::byte>(character));
        }
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
    [[nodiscard]] std::string text() {
        const auto count = u32();
        if (!count || count > 256) throw std::invalid_argument("direct GPU control identity is empty or excessive");
        require(count); std::string value; value.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto character = std::to_integer<unsigned char>(bytes_[offset_++]);
            if (character < 32 || character > 126) throw std::invalid_argument("direct GPU control identity is not canonical ASCII");
            value.push_back(static_cast<char>(character));
        }
        return value;
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
    if (limits.protocol_version != 2 ||
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
void validate_direct_gpu_control(const DistributedDirectGpuControlFrame& value) {
    const auto& descriptor = value.descriptor;
    if (value.contract_version != 1 || value.local_endpoint_identity == value.peer_endpoint_identity ||
        !transport::validate_descriptor(descriptor, {1, descriptor.connection_epoch, std::numeric_limits<std::uint32_t>::max()}) ||
        !descriptor.sequence || !descriptor.generation) throw std::invalid_argument("direct GPU control contract/descriptor is incompatible");
    switch (value.kind) {
    case DistributedDirectGpuControlKind::producer_released:
        if (!transport::validate_gpu_commit_record(value.commit, descriptor) || value.commit.payload_bytes != descriptor.byte_length) {
            throw std::invalid_argument("direct GPU control producer record is foreign");
        }
        break;
    case DistributedDirectGpuControlKind::receive_ready:
    case DistributedDirectGpuControlKind::received:
    case DistributedDirectGpuControlKind::consumed:
        if (value.commit.protocol_version != 1 || value.commit.commit_word || value.commit.connection_epoch || value.commit.sequence ||
            value.commit.payload_bytes || value.commit.stripe_index || value.commit.stripe_count != 1) throw std::invalid_argument("direct GPU control ACK cannot forge a commit record");
        break;
    default: throw std::invalid_argument("direct GPU control kind is unsupported");
    }
}

void token_state_write(Writer& w,const PersistentStateGenerations& state) {
    w.u32(static_cast<std::uint32_t>(state.valid));for(const auto value:state.generations) w.u64(value);
}
PersistentStateGenerations token_state_read(Reader& r) {
    PersistentStateGenerations state;const auto flags=r.u32();
    if(flags & ~63U) throw std::invalid_argument("distributed token state has unknown flags");
    state.valid=static_cast<PersistentState>(flags);for(auto& value:state.generations) value=r.u64();return state;
}
void token_shape(const DistributedTokenMessage& m) {
    const auto kind=static_cast<std::uint32_t>(m.kind);
    if(kind<1U || kind>6U || m.sender_node>1U || !m.sequence || m.sequence==std::numeric_limits<std::uint64_t>::max() || m.binding.contract_version!=1U ||
        !m.binding.model_content_identity.valid() || !m.binding.request_id || !m.binding.connection_epoch || !m.binding.generation || !m.binding.boundary_dtype_id ||
        !m.binding_identity || distributed_token_binding_identity(m.binding)!=m.binding_identity)
        throw std::invalid_argument("distributed token wire message has invalid kind, identity or sequence");
    const auto flags_valid=[](const PersistentStateGenerations& state){return !(static_cast<std::uint32_t>(state.valid)&~63U);};
    if(!flags_valid(m.binding.initial_global_state) || !flags_valid(m.receipt.local_state))
        throw std::invalid_argument("distributed token wire state has unknown flags");
    for(std::uint32_t node=0;node<2U;++node)
        if(m.binding.peers[node].node!=node || !flags_valid(m.binding.peers[node].initial_state))
            throw std::invalid_argument("distributed token wire peer table is invalid");
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
    if (reader.u16() != 0) throw std::invalid_argument("distributed control frame reserved bits are non-zero");
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
    case DistributedControlFrameKind::direct_gpu_stream:
    case DistributedControlFrameKind::token_transaction:
        return static_cast<
            DistributedControlFrameKind>(
                raw_kind);
    }
    throw std::invalid_argument(
        "distributed control frame kind is unknown");
}

std::vector<std::byte> encode_distributed_direct_gpu_control(const DistributedDirectGpuControlFrame& value, DistributedControlWireLimits limits) {
    validate_direct_gpu_control(value); Writer payload;
    payload.u32(value.contract_version); payload.u32(static_cast<std::uint32_t>(value.kind));
    payload.text(value.physical_link_id); payload.text(value.local_endpoint_identity); payload.text(value.peer_endpoint_identity);
    encode_descriptor(payload, value.descriptor);
    payload.u32(value.commit.protocol_version); payload.u32(value.commit.commit_word); payload.u64(value.commit.connection_epoch);
    payload.u64(value.commit.sequence); payload.u32(value.commit.payload_bytes); payload.u16(value.commit.stripe_index); payload.u16(value.commit.stripe_count);
    return frame(DistributedControlFrameKind::direct_gpu_stream, payload.take(), limits);
}
DistributedDirectGpuControlFrame decode_distributed_direct_gpu_control(std::span<const std::byte> bytes, DistributedControlWireLimits limits) {
    Reader reader(payload_of(bytes, DistributedControlFrameKind::direct_gpu_stream, limits)); DistributedDirectGpuControlFrame value;
    value.contract_version = reader.u32(); value.kind = static_cast<DistributedDirectGpuControlKind>(reader.u32());
    value.physical_link_id = reader.text(); value.local_endpoint_identity = reader.text(); value.peer_endpoint_identity = reader.text();
    value.descriptor = decode_descriptor(reader);
    value.commit.protocol_version = reader.u32(); value.commit.commit_word = reader.u32(); value.commit.connection_epoch = reader.u64();
    value.commit.sequence = reader.u64(); value.commit.payload_bytes = reader.u32(); value.commit.stripe_index = reader.u16(); value.commit.stripe_count = reader.u16();
    require_consumed(reader); validate_direct_gpu_control(value); return value;
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
        (!transfer.native_slices.empty() &&
         transfer.native_slices.size() != transfer.link_plan.slices.size()) ||
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
    payload.u32(static_cast<std::uint32_t>(transfer.native_slices.size()));
    for (const auto& identity : transfer.native_slices) {
        if (identity.connection_epoch == 0 || identity.sequence == 0) {
            throw std::invalid_argument("native slice identity is uninitialized");
        }
        payload.u64(identity.connection_epoch);
        payload.u64(identity.sequence);
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
    const auto native_count = reader.u32();
    if (native_count != 0 && native_count != slices) {
        throw std::invalid_argument("native slice identity count does not match slices");
    }
    transfer.native_slices.reserve(native_count);
    for (std::uint32_t index = 0; index < native_count; ++index) {
        DistributedNativeSliceIdentity identity;
        identity.connection_epoch = reader.u64();
        identity.sequence = reader.u64();
        if (identity.connection_epoch == 0 || identity.sequence == 0) {
            throw std::invalid_argument("native slice identity is uninitialized");
        }
        transfer.native_slices.push_back(identity);
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

std::vector<std::byte> encode_distributed_token_message(const DistributedTokenMessage& m,DistributedControlWireLimits limits) {
    token_shape(m);Writer w;const auto& b=m.binding;
    w.u32(b.contract_version);w.u64(b.serial_execution_identity);w.u64(b.kernel_plan_identity);w.u64(b.placement_plan_identity);
    for(const auto byte:b.model_content_identity.bytes) w.u16(byte);
    w.u64(b.request_id);w.u64(b.connection_epoch);w.u64(b.generation);w.u64(b.position);
    w.u64(b.committed_tokens);w.u64(b.previous_history_identity);w.u32(b.input_token);w.u32(b.boundary_dtype_id);token_state_write(w,b.initial_global_state);
    for(const auto& p:b.peers) {
        w.u32(p.node);w.u32(p.first_step);w.u32(p.last_step);w.u64(p.runtime_compatibility_identity);
        w.u64(p.state_layout_identity);w.u64(p.context_capacity);token_state_write(w,p.initial_state);
    }
    w.u64(m.binding_identity);w.u64(m.sequence);w.u64(m.decision_identity);w.u32(m.sender_node);w.u32(static_cast<std::uint32_t>(m.kind));
    w.u64(m.receipt.snapshot_plan_identity);w.u64(m.receipt.snapshot_generation);token_state_write(w,m.receipt.local_state);w.u32(m.receipt.completed_steps);
    return frame(DistributedControlFrameKind::token_transaction,w.take(),limits);
}
DistributedTokenMessage decode_distributed_token_message(std::span<const std::byte> bytes,DistributedControlWireLimits limits) {
    Reader r(payload_of(bytes,DistributedControlFrameKind::token_transaction,limits));DistributedTokenMessage m;auto& b=m.binding;
    b.contract_version=r.u32();b.serial_execution_identity=r.u64();b.kernel_plan_identity=r.u64();b.placement_plan_identity=r.u64();
    for(auto& byte:b.model_content_identity.bytes) {const auto value=r.u16();if(value>255U) throw std::invalid_argument("distributed model digest byte is noncanonical");byte=static_cast<std::uint8_t>(value);}
    b.request_id=r.u64();b.connection_epoch=r.u64();b.generation=r.u64();b.position=r.u64();
    b.committed_tokens=r.u64();b.previous_history_identity=r.u64();b.input_token=r.u32();b.boundary_dtype_id=r.u32();b.initial_global_state=token_state_read(r);
    for(auto& p:b.peers) {
        p.node=r.u32();p.first_step=r.u32();p.last_step=r.u32();p.runtime_compatibility_identity=r.u64();
        p.state_layout_identity=r.u64();p.context_capacity=r.u64();p.initial_state=token_state_read(r);
    }
    m.binding_identity=r.u64();m.sequence=r.u64();m.decision_identity=r.u64();m.sender_node=r.u32();m.kind=static_cast<DistributedTokenMessageKind>(r.u32());
    m.receipt.snapshot_plan_identity=r.u64();m.receipt.snapshot_generation=r.u64();m.receipt.local_state=token_state_read(r);m.receipt.completed_steps=r.u32();
    require_consumed(r);token_shape(m);return m;
}
void FramedDistributedTokenWire::send(const DistributedTokenMessage& m) {stream_->send(encode_distributed_token_message(m,limits_));}
DistributedTokenMessage FramedDistributedTokenWire::receive() {return decode_distributed_token_message(stream_->receive(),limits_);}

}  // namespace strix
