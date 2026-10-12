#include "strix/copied_stream_transport.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix::transport {
namespace {
constexpr std::uint32_t copied_magic = 0x53544350U;
constexpr std::uint32_t copied_version = 1;
enum class PacketKind : std::uint16_t { payload = 1, received = 2, consumed = 3 };

bool same_buffer(const TransportBufferView& a, const TransportBufferView& b) noexcept {
    return a.address == b.address && a.bytes == b.bytes && a.allocation_generation == b.allocation_generation &&
           a.domain == b.domain && a.access == b.access;
}
bool same_descriptor(const TransferDescriptor& a, const TransferDescriptor& b) noexcept {
    return a.protocol_version == b.protocol_version && a.connection_epoch == b.connection_epoch &&
           a.model_plan_id == b.model_plan_id && a.request_id == b.request_id && a.generation == b.generation &&
           a.boundary_id == b.boundary_id && a.sequence == b.sequence && a.shape_fingerprint == b.shape_fingerprint &&
           a.dtype_id == b.dtype_id && a.byte_length == b.byte_length;
}
bool same_binding(const CopiedStreamCompletionBinding& a, const CopiedStreamCompletionBinding& b) noexcept {
    return same_descriptor(a.descriptor, b.descriptor) && same_buffer(a.buffer, b.buffer) && a.scope == b.scope;
}
CopiedStreamCompletionBinding completion_binding(const DistributedBoundaryTransfer& transfer,
    const CopiedStreamOwnedBuffer& buffer, bool receive) {
    return {transfer.descriptor, buffer.view, receive ? CopiedStreamCompletionScope::consumer_complete
                                                    : CopiedStreamCompletionScope::producer_released};
}
void validate_owned(const CopiedStreamOwnedBuffer& buffer, std::uint64_t required, std::uint64_t maximum, bool receive,
                    bool exact = true) {
    if (!buffer.allocation || buffer.allocation->allocation_generation() == 0 ||
        buffer.view.allocation_generation != buffer.allocation->allocation_generation() ||
        (exact && buffer.view.bytes != required) ||
        (buffer.view.domain != TransportMemoryDomain::host_pageable && buffer.view.domain != TransportMemoryDomain::host_registered) ||
        (receive && buffer.view.access != TransportBufferAccess::read_write) ||
        !(receive ? validate_transport_receive_buffer(buffer.view, required) : validate_transport_send_buffer(buffer.view, required))) {
        throw std::invalid_argument("copied stream requires an owned readable/writable host allocation");
    }
    const auto bytes = buffer.allocation->bytes();
    const auto base = reinterpret_cast<std::uintptr_t>(bytes.data());
    if (base == 0 || bytes.empty() || bytes.size() > maximum ||
        bytes.size() > std::numeric_limits<std::uintptr_t>::max() - base ||
        buffer.allocation->writable_bytes().data() != bytes.data() ||
        buffer.allocation->writable_bytes().size() != bytes.size() || buffer.view.address < base ||
        buffer.view.address - base > bytes.size() || buffer.view.bytes > bytes.size() - (buffer.view.address - base)) {
        throw std::invalid_argument("copied stream view is outside its bounded owned allocation");
    }
}
std::vector<std::byte> metadata(const DistributedBoundaryTransfer& transfer, const CopiedStreamConnection& connection,
                              const CopiedStreamTransportPolicy& policy) {
    const auto& d = transfer.descriptor;
    if (!validate_descriptor(d, {1, d.connection_epoch, policy.maximum_payload_bytes}) || d.sequence == 0 || d.generation == 0 ||
        transfer.flow.flow_id == 0 || transfer.flow.direction != LinkDirection::outbound ||
        (transfer.flow.priority != TransferPriority::normal && transfer.flow.priority != TransferPriority::latency_critical &&
         transfer.flow.priority != TransferPriority::bulk) || transfer.link_plan.kind != PlanKind::single_link ||
        transfer.link_plan.slices.size() != 1 || !transfer.native_slices.empty()) {
        throw std::invalid_argument("copied stream accepts only exact v1 logical, unstriped host-copy metadata");
    }
    const auto& slice = transfer.link_plan.slices[0];
    if (slice.link_index != connection.link_index || slice.byte_offset != 0 || slice.byte_length != d.byte_length) {
        throw std::invalid_argument("copied stream plan does not match its dedicated physical route");
    }
    return encode_distributed_boundary_transfer(transfer, {2, policy.maximum_frame_bytes, 1});
}
TransferHandle next_handle() {
    static std::atomic<TransferHandle> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;) {
        if (value == std::numeric_limits<TransferHandle>::max()) throw std::overflow_error("copied stream handles exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
    }
}
class Writer {
public:
    void integer(std::uint64_t value, unsigned bytes) {
        for (unsigned index = 0; index < bytes; ++index) data.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
    }
    void append(std::span<const std::byte> bytes) { data.insert(data.end(), bytes.begin(), bytes.end()); }
    std::vector<std::byte> data;
};
class Reader {
public:
    explicit Reader(std::span<const std::byte> input) : input_(input) {}
    std::span<const std::byte> take(std::size_t count) {
        if (count > input_.size() - offset_) throw std::invalid_argument("copied stream packet is truncated");
        const auto result = input_.subspan(offset_, count); offset_ += count; return result;
    }
    std::uint64_t integer(unsigned bytes) {
        const auto data = take(bytes); std::uint64_t value = 0;
        for (unsigned index = 0; index < bytes; ++index) value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(data[index])) << (index * 8U);
        return value;
    }
    bool empty() const noexcept { return offset_ == input_.size(); }
private:
    std::span<const std::byte> input_;
    std::size_t offset_ = 0;
};
class SynchronousProof final : public CopiedStreamCompletionProof {
public:
    explicit SynchronousProof(CopiedStreamCompletionBinding value) : binding_(std::move(value)) {}
    const CopiedStreamCompletionBinding& binding() const noexcept override { return binding_; }
    CopiedStreamCompletionStatus poll() override { return CopiedStreamCompletionStatus::complete; }
private:
    CopiedStreamCompletionBinding binding_;
};
class InboundProof final : public CopiedStreamCompletionProof {
public:
    InboundProof(CopiedStreamCompletionBinding value, std::unique_ptr<DistributedInboundCompletion> completion)
        : binding_(std::move(value)), completion_(std::move(completion)) {
        if (!completion_ || binding_.scope != CopiedStreamCompletionScope::consumer_complete) {
            throw std::invalid_argument("copied stream inbound completion requires a consumer-scoped proof");
        }
    }
    const CopiedStreamCompletionBinding& binding() const noexcept override { return binding_; }
    CopiedStreamCompletionStatus poll() override {
        switch (completion_->poll()) {
        case DistributedInboundCompletionStatus::pending: return CopiedStreamCompletionStatus::pending;
        case DistributedInboundCompletionStatus::complete: return CopiedStreamCompletionStatus::complete;
        case DistributedInboundCompletionStatus::failed: return CopiedStreamCompletionStatus::failed_retained;
        }
        return CopiedStreamCompletionStatus::failed_retained;
    }
private:
    CopiedStreamCompletionBinding binding_;
    std::unique_ptr<DistributedInboundCompletion> completion_;
};
}  // namespace

CopiedStreamConnectionState::CopiedStreamConnectionState(CopiedStreamConnection connection) : connection_(std::move(connection)) {
    if (connection_.physical_link_id.empty() || connection_.physical_link_id.size() > 256 || connection_.physical_epoch == 0 ||
        connection_.local_node == connection_.peer_node) throw std::invalid_argument("copied stream requires a reconciled physical connection");
}
CopiedStreamHostBuffer::CopiedStreamHostBuffer(std::uint64_t bytes, AllocationGenerationSource& generations)
    : generation_(generations.next()) {
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("copied host buffer size is invalid");
    storage_.resize(static_cast<std::size_t>(bytes));
}
std::span<std::byte> CopiedStreamHostBuffer::writable_bytes() noexcept { return storage_; }
std::span<const std::byte> CopiedStreamHostBuffer::bytes() const noexcept { return storage_; }
std::uint64_t CopiedStreamHostBuffer::allocation_generation() const noexcept { return generation_; }
TransportBufferView CopiedStreamHostBuffer::view() const noexcept {
    return {reinterpret_cast<std::uintptr_t>(storage_.data()), storage_.size(), generation_,
            TransportMemoryDomain::host_pageable, TransportBufferAccess::read_write};
}
std::unique_ptr<CopiedStreamCompletionProof> make_copied_stream_synchronous_proof(CopiedStreamCompletionBinding binding) {
    return std::make_unique<SynchronousProof>(std::move(binding));
}
std::unique_ptr<CopiedStreamCompletionProof> bind_copied_stream_inbound_proof(CopiedStreamCompletionBinding binding,
    std::unique_ptr<DistributedInboundCompletion> completion) {
    return std::make_unique<InboundProof>(std::move(binding), std::move(completion));
}

struct CopiedStreamTransportBackend::Impl {
    struct Active {
        TransferHandle handle = 0;
        DistributedBoundaryTransfer transfer;
        CopiedStreamOwnedBuffer buffer;
        std::unique_ptr<CopiedStreamCompletionProof> proof;
        std::unique_ptr<CopiedStreamCompletionProof> previous_proof;
        std::vector<std::byte> metadata;
        std::uint64_t physical_sequence = 0;
        bool receive = false, submitted = false, exposed = false, received_ack = false;
    };
    DirectStreamControlByteStream* stream;
    CopiedStreamConnectionState* state;
    CopiedStreamTransportPolicy policy;
    std::unique_ptr<Active> active;
    std::deque<Completion> completions;
    bool failed = false;

    void fail() noexcept { failed = true; state->failed_ = true; }
    void healthy() const { if (failed || state->failed_) throw std::logic_error("copied stream connection is failed"); }
    Active& require(TransferHandle handle, bool check_health = true) {
        if (check_health) healthy();
        if (!active || active->handle != handle) throw std::invalid_argument("copied stream handle is stale or foreign");
        return *active;
    }
    CopiedStreamCompletionStatus progress(Active& operation, bool cleanup = false) {
        if (!operation.proof) return CopiedStreamCompletionStatus::pending;
        const auto expected = completion_binding(operation.transfer, operation.buffer, operation.receive);
        const auto required = cleanup ? 1 : operation.transfer.descriptor.byte_length;
        const auto maximum = cleanup ? std::numeric_limits<std::uint64_t>::max() : policy.maximum_payload_bytes;
        validate_owned(operation.buffer, required, maximum, operation.receive, !cleanup);
        if (!same_binding(operation.proof->binding(), expected)) throw std::invalid_argument("copied completion proof is foreign");
        const auto status = operation.proof->poll();
        validate_owned(operation.buffer, required, maximum, operation.receive, !cleanup);
        if (!same_binding(operation.proof->binding(), expected)) throw std::invalid_argument("copied completion binding changed during callback");
        return status;
    }
    std::vector<std::byte> packet(PacketKind kind, const Active& operation, std::span<const std::byte> payload = {}) {
        const auto& connection = state->connection_;
        Writer writer;
        writer.data.reserve(static_cast<std::size_t>(48 + connection.physical_link_id.size() + operation.metadata.size() + payload.size()));
        writer.integer(copied_magic, 4); writer.integer(copied_version, 4);
        writer.integer(static_cast<std::uint16_t>(kind), 2); writer.integer(0, 2);
        writer.integer(connection.local_node, 4); writer.integer(connection.peer_node, 4);
        writer.integer(connection.physical_epoch, 8); writer.integer(operation.physical_sequence, 8);
        writer.integer(connection.physical_link_id.size(), 4); writer.integer(operation.metadata.size(), 4); writer.integer(payload.size(), 4);
        for (const auto character : connection.physical_link_id) writer.integer(static_cast<unsigned char>(character), 1);
        writer.append(operation.metadata); writer.append(payload);
        if (writer.data.size() > policy.maximum_frame_bytes) throw std::invalid_argument("copied packet exceeds frame budget");
        return std::move(writer.data);
    }
    std::span<const std::byte> unpack(std::span<const std::byte> packet_bytes, Active& operation, PacketKind kind) {
        if (packet_bytes.size() > policy.maximum_frame_bytes) throw std::invalid_argument("copied received packet exceeds frame budget");
        Reader reader(packet_bytes); const auto& connection = state->connection_;
        if (reader.integer(4) != copied_magic || reader.integer(4) != copied_version ||
            reader.integer(2) != static_cast<std::uint16_t>(kind) || reader.integer(2) != 0 ||
            reader.integer(4) != connection.peer_node || reader.integer(4) != connection.local_node ||
            reader.integer(8) != connection.physical_epoch) throw std::invalid_argument("copied packet version/kind/node/epoch is foreign");
        const auto physical_sequence = reader.integer(8);
        const auto name_bytes = reader.integer(4), metadata_bytes = reader.integer(4), payload_bytes = reader.integer(4);
        if (name_bytes != connection.physical_link_id.size() || metadata_bytes != operation.metadata.size() ||
            payload_bytes != (kind == PacketKind::payload ? operation.transfer.descriptor.byte_length : 0)) {
            throw std::invalid_argument("copied packet has impossible route/descriptor/payload lengths");
        }
        const auto name = reader.take(static_cast<std::size_t>(name_bytes));
        for (std::size_t index = 0; index < name.size(); ++index) {
            if (std::to_integer<unsigned char>(name[index]) != static_cast<unsigned char>(connection.physical_link_id[index])) {
                throw std::invalid_argument("copied packet physical route is foreign");
            }
        }
        const auto meta = reader.take(static_cast<std::size_t>(metadata_bytes));
        static_cast<void>(decode_distributed_boundary_transfer(meta, {2, policy.maximum_frame_bytes, 1}));
        if (!std::equal(meta.begin(), meta.end(), operation.metadata.begin())) throw std::invalid_argument("copied packet logical descriptor/plan differs from queued metadata");
        const auto payload = reader.take(static_cast<std::size_t>(payload_bytes));
        if (!reader.empty()) throw std::invalid_argument("copied packet contains trailing bytes");
        if (kind == PacketKind::payload) {
            if (state->last_receive_ == std::numeric_limits<std::uint64_t>::max() || physical_sequence != state->last_receive_ + 1) {
                throw std::invalid_argument("copied packet physical sequence is stale or out of order");
            }
            operation.physical_sequence = physical_sequence; state->last_receive_ = physical_sequence;
        } else if (physical_sequence != operation.physical_sequence) throw std::invalid_argument("copied ACK physical sequence is stale");
        return payload;
    }
};

CopiedStreamTransportBackend::CopiedStreamTransportBackend(DirectStreamControlByteStream& stream,
    CopiedStreamConnectionState& state, CopiedStreamTransportPolicy policy) : impl_(std::make_unique<Impl>()) {
    if (policy.maximum_payload_bytes == 0 || policy.maximum_frame_bytes > std::numeric_limits<std::uint32_t>::max() ||
        policy.maximum_frame_bytes < 512 || policy.maximum_payload_bytes > policy.maximum_frame_bytes - 512 ||
        policy.maximum_frame_bytes > stream.maximum_frame_bytes() ||
        policy.maximum_completion_polls == 0) throw std::invalid_argument("copied stream requires bounded payload/frame/progress budgets");
    impl_->stream = &stream; impl_->state = &state; impl_->policy = policy; impl_->healthy();
    if (state.attached_) throw std::logic_error("copied physical connection already has an owner");
    state.attached_ = true;
}
CopiedStreamTransportBackend::~CopiedStreamTransportBackend() {
    if (impl_->active && drain_retained() != CopiedStreamCleanupStatus::released) {
        // One bounded host allocation and its proof remain owned until process
        // teardown. A failed epoch may not recreate transports automatically.
        static_cast<void>(impl_.release());
        return;
    }
    impl_->state->attached_ = false;
}
BackendCapabilities CopiedStreamTransportBackend::capabilities() const {
    BackendCapabilities result; result.name = "host-copy-framed-stream-v1";
    result.maximum_payload_bytes = impl_->policy.maximum_payload_bytes;
    result.buffer_mode = TransportBufferMode::copy_stream; result.progress_mode = TransportProgressMode::blocking;
    return result;
}
TransferHandle CopiedStreamTransportBackend::prepare_send(const TransferDescriptor&, const TransportBufferView&) {
    throw std::logic_error("copied stream send requires owned metadata and explicit producer proof");
}
TransferHandle CopiedStreamTransportBackend::prepare_receive(const TransferDescriptor&, const TransportBufferView&) {
    throw std::logic_error("copied stream receive requires owned metadata and host allocation");
}
TransferHandle CopiedStreamTransportBackend::prepare_send(const DistributedBoundaryTransfer& transfer, CopiedStreamPreparedSource source) {
    impl_->healthy();
    if (impl_->active || !impl_->completions.empty()) throw std::logic_error("copied stream permits one in-flight operation");
    auto operation = std::make_unique<Impl::Active>(); operation->handle = next_handle(); operation->transfer = transfer;
    operation->buffer = std::move(source.buffer); operation->proof = std::move(source.producer);
    impl_->active = std::move(operation);
    try {
        auto& active = *impl_->active;
        validate_owned(active.buffer, transfer.descriptor.byte_length, impl_->policy.maximum_payload_bytes, false);
        if (!active.proof || !same_binding(active.proof->binding(), completion_binding(transfer, active.buffer, false))) {
            throw std::invalid_argument("copied stream producer proof is missing or foreign");
        }
        active.metadata = metadata(transfer, impl_->state->connection_, impl_->policy);
        if (impl_->state->next_send_ == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("copied physical sequence exhausted");
        active.physical_sequence = impl_->state->next_send_++;
        return active.handle;
    } catch (...) { impl_->fail(); throw; }
}
TransferHandle CopiedStreamTransportBackend::prepare_receive(const DistributedBoundaryTransfer& transfer, CopiedStreamOwnedBuffer buffer) {
    impl_->healthy();
    if (impl_->active || !impl_->completions.empty()) throw std::logic_error("copied stream permits one in-flight operation");
    auto operation = std::make_unique<Impl::Active>(); operation->handle = next_handle(); operation->transfer = transfer;
    operation->buffer = std::move(buffer); operation->receive = true; impl_->active = std::move(operation);
    try {
        validate_owned(impl_->active->buffer, transfer.descriptor.byte_length, impl_->policy.maximum_payload_bytes, true);
        impl_->active->metadata = metadata(transfer, impl_->state->connection_, impl_->policy);
        return impl_->active->handle;
    } catch (...) { impl_->fail(); throw; }
}
void CopiedStreamTransportBackend::submit(TransferHandle handle) {
    auto& operation = impl_->require(handle);
    if (operation.submitted) throw std::logic_error("copied operation cannot be submitted twice");
    try {
        if (operation.receive) {
            const auto frame = impl_->stream->receive();
            const auto payload = impl_->unpack(frame, operation, PacketKind::payload);
            validate_owned(operation.buffer, operation.transfer.descriptor.byte_length, impl_->policy.maximum_payload_bytes, true);
            // Network bytes are bounded staging only until exact expected
            // descriptor, physical identity and sequence were all validated.
            std::memcpy(reinterpret_cast<void*>(operation.buffer.view.address), payload.data(), payload.size());
            impl_->stream->send(impl_->packet(PacketKind::received, operation));
            operation.submitted = true;
            impl_->completions.push_back({handle, CompletionKind::receive_complete, operation.transfer.descriptor.byte_length});
        } else {
            bool ready = false;
            for (std::uint64_t count = 0; count < impl_->policy.maximum_completion_polls; ++count) {
                const auto status = impl_->progress(operation);
                if (status == CopiedStreamCompletionStatus::complete) { ready = true; break; }
                if (status != CopiedStreamCompletionStatus::pending) throw std::runtime_error("copied producer failed with source retained");
            }
            if (!ready) throw std::runtime_error("copied producer completion budget exhausted; source retained");
            const auto* payload = reinterpret_cast<const std::byte*>(operation.buffer.view.address);
            const auto frame = impl_->packet(PacketKind::payload, operation,
                {payload, static_cast<std::size_t>(operation.transfer.descriptor.byte_length)});
            operation.submitted = true;
            impl_->stream->send(frame); // FD boundary retains no bytes after return.
            impl_->completions.push_back({handle, CompletionKind::local_buffer_reusable, operation.transfer.descriptor.byte_length});
        }
    } catch (...) { impl_->fail(); throw; }
}
std::optional<Completion> CopiedStreamTransportBackend::poll() {
    impl_->healthy();
    if (!impl_->completions.empty()) { const auto result = impl_->completions.front(); impl_->completions.pop_front(); return result; }
    if (!impl_->active || !impl_->active->submitted) return std::nullopt;
    auto& operation = *impl_->active;
    try {
        if (operation.receive) {
            if (!operation.proof) return std::nullopt;
            const auto status = impl_->progress(operation);
            if (status == CopiedStreamCompletionStatus::pending) return std::nullopt;
            if (status != CopiedStreamCompletionStatus::complete) throw std::runtime_error("copied consumer failed with source retained");
            impl_->stream->send(impl_->packet(PacketKind::consumed, operation));
            const Completion result{operation.handle, CompletionKind::local_buffer_reusable, operation.transfer.descriptor.byte_length};
            impl_->active.reset(); return result;
        }
        const auto frame = impl_->stream->receive();
        static_cast<void>(impl_->unpack(frame, operation, operation.received_ack ? PacketKind::consumed : PacketKind::received));
        const Completion result{operation.handle, operation.received_ack ? CompletionKind::remote_consumer_complete
                                                                      : CompletionKind::remote_transport_ack,
                                operation.transfer.descriptor.byte_length};
        if (operation.received_ack) impl_->active.reset(); else operation.received_ack = true;
        return result;
    } catch (...) { impl_->fail(); throw; }
}
void CopiedStreamTransportBackend::cancel(TransferHandle handle) {
    static_cast<void>(impl_->require(handle, false)); impl_->fail();
    // Cancellation invalidates this local epoch, not remote model state. No
    // buffer or pending producer/consumer proof is retired by callback return.
}
TransportBufferView CopiedStreamTransportBackend::receive_buffer(TransferHandle handle) {
    auto& operation = impl_->require(handle);
    if (!operation.receive || !operation.submitted) throw std::logic_error("copied RX is not validated and ready");
    validate_owned(operation.buffer, operation.transfer.descriptor.byte_length, impl_->policy.maximum_payload_bytes, true);
    operation.exposed = true; return operation.buffer.view;
}
void CopiedStreamTransportBackend::finish_consumer(TransferHandle handle, std::unique_ptr<CopiedStreamCompletionProof> proof) {
    auto& operation = impl_->require(handle);
    if (!operation.receive || !operation.submitted || operation.proof) {
        throw std::logic_error("copied consumer proof requires one validated unregistered RX");
    }
    attach_cleanup_proof(handle, std::move(proof));
}
void CopiedStreamTransportBackend::attach_cleanup_proof(TransferHandle handle, std::unique_ptr<CopiedStreamCompletionProof> proof) {
    auto& operation = impl_->require(handle, false);
    if (!proof || !same_binding(proof->binding(), completion_binding(operation.transfer, operation.buffer, operation.receive))) {
        impl_->fail(); throw std::invalid_argument("copied completion/cleanup proof is missing or foreign");
    }
    if (operation.proof && !impl_->failed) throw std::logic_error("copied consumer completion already registered");
    if (operation.proof) {
        if (operation.previous_proof) throw std::logic_error("copied cleanup proof replacement budget exhausted");
        operation.previous_proof = std::move(operation.proof);
    }
    operation.proof = std::move(proof);
}
CopiedStreamCleanupStatus CopiedStreamTransportBackend::drain_retained() {
    if (!impl_->active) return CopiedStreamCleanupStatus::released;
    impl_->fail(); // Cleanup never makes a partially accounted wire epoch reusable.
    auto& operation = *impl_->active;
    try {
        if (!operation.proof && (!operation.receive || operation.exposed)) return CopiedStreamCleanupStatus::pending;
        for (std::uint64_t count = 0; count < impl_->policy.maximum_completion_polls; ++count) {
            const auto status = operation.proof ? impl_->progress(operation, true) : CopiedStreamCompletionStatus::complete;
            if (status == CopiedStreamCompletionStatus::complete) { impl_->active.reset(); impl_->completions.clear(); return CopiedStreamCleanupStatus::released; }
            if (status != CopiedStreamCompletionStatus::pending) { impl_->fail(); return CopiedStreamCleanupStatus::failed_retained; }
        }
        impl_->fail(); return CopiedStreamCleanupStatus::pending;
    } catch (...) { impl_->fail(); return CopiedStreamCleanupStatus::failed_retained; }
}
bool CopiedStreamTransportBackend::failed() const noexcept { return impl_->failed || impl_->state->failed_; }
bool CopiedStreamTransportBackend::retains_buffer() const noexcept { return static_cast<bool>(impl_->active); }
std::optional<TransferHandle> CopiedStreamTransportBackend::retained_handle() const noexcept {
    return impl_->active ? std::optional<TransferHandle>(impl_->active->handle) : std::nullopt;
}
const CopiedStreamConnection& CopiedStreamTransportBackend::connection() const noexcept { return impl_->state->connection_; }

}  // namespace strix::transport

namespace strix {
namespace {
bool same_boundary(const DistributedExecutionBoundary& a, const DistributedExecutionBoundary& b) {
    return a.boundary_id == b.boundary_id && a.source_step == b.source_step && a.destination_step == b.destination_step &&
           a.source_node == b.source_node && a.destination_node == b.destination_node && a.payload_bytes == b.payload_bytes &&
           a.payloads == b.payloads;
}
bool same_step(const DistributedStepPlacement& a, const DistributedStepPlacement& b) {
    return a.step_id == b.step_id && a.op == b.op && a.node == b.node && a.remote == b.remote &&
           a.placement_ambiguous == b.placement_ambiguous;
}
}  // namespace
CopiedStreamDistributedBackend::CopiedStreamDistributedBackend(transport::CopiedStreamTransportBackend& backend,
    const DistributedPlacementPlan& placement, DistributedRuntimeRequest request, DistributedNativeControlPlane& control,
    DistributedControlReceiver& receiver, DistributedNativeLocalExecutor& local, CopiedStreamBoundarySourceProvider& source,
    DistributedBoundaryStagingProvider& consumer, AllocationGenerationSource& generations, std::uint64_t maximum_polls)
    : transport_(&backend), placement_(&placement), request_(request), control_(&control), receiver_(&receiver), local_(&local),
      source_(&source), consumer_(&consumer), generations_(&generations), maximum_polls_(maximum_polls) {
    if (!placement.executable || placement.plan_identity == 0 || placement.local_node != backend.connection().local_node ||
        request.connection_epoch == 0 || request.request_id == 0 || request.generation == 0 || request.dtype_id == 0 ||
        maximum_polls == 0) throw std::invalid_argument("copied distributed backend requires exact executable request/route");
}
void CopiedStreamDistributedBackend::require_healthy() const {
    if (failed_ || transport_->failed() || receiver_->failed()) throw std::logic_error("copied distributed backend is failed");
}
void CopiedStreamDistributedBackend::execute_local_step(const DistributedStepPlacement& step) {
    require_healthy();
    if (step.remote || step.placement_ambiguous || step.node != placement_->local_node ||
        std::none_of(placement_->steps.begin(), placement_->steps.end(), [&](const auto& value) { return same_step(value, step); })) {
        failed_ = true; throw std::invalid_argument("copied local step is outside placement authority");
    }
    try { local_->execute_local_step(step); } catch (...) { failed_ = true; throw; }
}
void CopiedStreamDistributedBackend::execute_remote_step(const DistributedStepPlacement& step) {
    require_healthy();
    try { control_->execute_remote_step(step); } catch (...) { failed_ = true; throw; }
}
void CopiedStreamDistributedBackend::validate_boundary(const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer, bool receive, std::uint64_t sequence) const {
    if (std::none_of(placement_->boundaries.begin(), placement_->boundaries.end(),
            [&](const auto& value) { return same_boundary(value, boundary); }) ||
        (receive ? boundary.destination_node : boundary.source_node) != placement_->local_node ||
        (receive ? boundary.source_node : boundary.destination_node) != transport_->connection().peer_node ||
        boundary.payload_bytes == 0 || boundary.payload_bytes > transport_->capabilities().maximum_payload_bytes ||
        sequence == std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument("copied boundary is outside exact placement/peer authority");
    }
    const auto expected = make_distributed_boundary_descriptor(*placement_, boundary, request_.connection_epoch,
        request_.request_id, request_.generation, sequence, request_.dtype_id);
    if (!transport::same_descriptor(transfer.descriptor, expected)) throw std::invalid_argument("copied boundary descriptor is stale or foreign");
    auto source_placement = *placement_; source_placement.local_node = boundary.source_node;
    const auto expected_flow = classify_distributed_boundary_flow(source_placement, boundary);
    if (transfer.flow.flow_id != expected_flow.flow_id || transfer.flow.direction != expected_flow.direction ||
        transfer.flow.priority != expected_flow.priority) throw std::invalid_argument("copied sender flow metadata is foreign");
}
void CopiedStreamDistributedBackend::send_boundary(const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer) {
    require_healthy(); std::optional<transport::TransferHandle> handle;
    try {
        validate_boundary(boundary, transfer, false, next_send_);
        auto source = source_->stage_outbound_owned(boundary, transfer);
        handle = transport_->prepare_send(transfer, std::move(source));
        control_->publish_boundary_transfer(boundary, transfer);
        transport_->submit(*handle);
        bool local = false, ack = false, consumed = false;
        for (std::uint64_t count = 0; count < maximum_polls_; ++count) {
            const auto completion = transport_->poll();
            if (!completion) continue;
            if (completion->handle != *handle || completion->bytes != boundary.payload_bytes) throw std::runtime_error("copied send completion is foreign");
            if (completion->kind == transport::CompletionKind::local_buffer_reusable) local = true;
            else if (completion->kind == transport::CompletionKind::remote_transport_ack) ack = true;
            else if (completion->kind == transport::CompletionKind::remote_consumer_complete) consumed = true;
            else throw std::runtime_error("copied send completion kind is invalid");
            if (local && ack && consumed) { ++next_send_; return; }
        }
        throw std::runtime_error("copied send progress budget exhausted");
    } catch (...) {
        failed_ = true;
        if (handle && transport_->retains_buffer()) { try { transport_->cancel(*handle); } catch (...) {} }
        throw;
    }
}
DistributedBoundaryTransfer CopiedStreamDistributedBackend::receive_boundary(const DistributedExecutionBoundary& boundary) {
    require_healthy(); std::optional<transport::TransferHandle> handle;
    try {
        // The v2 receiver queue, not arrival order of payload bytes, identifies
        // the next expected request/boundary. Revalidate before staging/consume.
        auto transfer = receiver_->take_boundary(boundary.boundary_id);
        validate_boundary(boundary, transfer, true, next_receive_);
        auto allocation = std::make_shared<transport::CopiedStreamHostBuffer>(boundary.payload_bytes, *generations_);
        const transport::CopiedStreamOwnedBuffer staged{allocation, allocation->view()};
        handle = transport_->prepare_receive(transfer, staged);
        transport_->submit(*handle);
        const auto completion = transport_->poll();
        if (!completion || completion->handle != *handle || completion->kind != transport::CompletionKind::receive_complete ||
            completion->bytes != boundary.payload_bytes) throw std::runtime_error("copied receive completion is invalid");
        const auto view = transport_->receive_buffer(*handle);
        auto inbound_transfer = transfer;
        inbound_transfer.flow = classify_distributed_boundary_flow(*placement_, boundary);
        consumer_->consume_inbound(boundary, view);
        auto proof = consumer_->finish_inbound(boundary, inbound_transfer);
        transport_->finish_consumer(*handle, transport::bind_copied_stream_inbound_proof(
            {transfer.descriptor, view, transport::CopiedStreamCompletionScope::consumer_complete}, std::move(proof)));
        for (std::uint64_t count = 0; count < maximum_polls_; ++count) {
            const auto ready = transport_->poll();
            if (!ready) continue;
            if (ready->handle != *handle || ready->kind != transport::CompletionKind::local_buffer_reusable ||
                ready->bytes != boundary.payload_bytes) throw std::runtime_error("copied receive retirement is invalid");
            ++next_receive_; return inbound_transfer;
        }
        throw std::runtime_error("copied consumer progress budget exhausted; host buffer retained");
    } catch (...) {
        failed_ = true;
        if (handle && transport_->retains_buffer()) { try { transport_->cancel(*handle); } catch (...) {} }
        throw;
    }
}
}  // namespace strix
