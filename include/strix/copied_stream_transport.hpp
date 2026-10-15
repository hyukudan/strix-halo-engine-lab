#pragma once

#include "strix/direct_stream_control.hpp"
#include "strix/distributed_control_receiver.hpp"
#include "strix/host_upload.hpp"
#include "strix/tcp_copied_payload_stream.hpp"
#include "strix/ssh_bound_copied_stream.hpp"

#include <memory>

namespace strix::transport {

struct CopiedStreamConnection {
    std::string physical_link_id;
    std::uint64_t physical_epoch = 0;
    std::uint32_t local_node = 0;
    std::uint32_t peer_node = 0;
    std::uint32_t link_index = 0;
    bool operator==(const CopiedStreamConnection&) const = default;
};

// One serialized owner. This state outlives the physical connection epoch and
// every backend using it; replay/failure history is never reset on reopening.
// Replacement requires drained producers/consumers and fresh peer-agreed epoch.
class CopiedStreamConnectionState {
public:
    explicit CopiedStreamConnectionState(CopiedStreamConnection);
    CopiedStreamConnectionState(const CopiedStreamConnectionState&) = delete;
    CopiedStreamConnectionState& operator=(const CopiedStreamConnectionState&) = delete;
    [[nodiscard]] const CopiedStreamConnection& connection() const noexcept { return connection_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }
private:
    friend class CopiedStreamTransportBackend;
    CopiedStreamConnection connection_;
    std::uint64_t next_send_ = 1;
    std::uint64_t last_receive_ = 0;
    bool failed_ = false;
    bool attached_ = false;
};

class CopiedStreamHostBuffer final : public HostUploadSource {
public:
    CopiedStreamHostBuffer(std::uint64_t bytes, AllocationGenerationSource&);
    [[nodiscard]] std::span<std::byte> writable_bytes() noexcept override;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept override;
    [[nodiscard]] std::uint64_t allocation_generation() const noexcept override;
    [[nodiscard]] TransportBufferView view() const noexcept;
private:
    std::vector<std::byte> storage_;
    std::uint64_t generation_;
};

enum class CopiedStreamCompletionScope { producer_released, consumer_complete };
enum class CopiedStreamCompletionStatus { pending, complete, failed_retained };
struct CopiedStreamCompletionBinding {
    TransferDescriptor descriptor;
    TransportBufferView buffer;
    CopiedStreamCompletionScope scope = CopiedStreamCompletionScope::producer_released;
};

// Trusted operation-bound completion declaration. Complete proves all work
// covered by this scope stopped accessing the owned source. CPU callback
// return, cancellation and receive completion do not supply that proof.
// Providers retain launched work/resources until the backend adopts a proof;
// proof destruction must itself drain or safely retain any outstanding work.
class CopiedStreamCompletionProof {
public:
    virtual ~CopiedStreamCompletionProof() = default;
    [[nodiscard]] virtual const CopiedStreamCompletionBinding& binding() const noexcept = 0;
    [[nodiscard]] virtual CopiedStreamCompletionStatus poll() = 0;
};
[[nodiscard]] std::unique_ptr<CopiedStreamCompletionProof> make_copied_stream_synchronous_proof(
    CopiedStreamCompletionBinding);
[[nodiscard]] std::unique_ptr<CopiedStreamCompletionProof> bind_copied_stream_inbound_proof(
    CopiedStreamCompletionBinding, std::unique_ptr<DistributedInboundCompletion>);

struct CopiedStreamOwnedBuffer {
    std::shared_ptr<HostUploadSource> allocation;
    TransportBufferView view;
};
struct CopiedStreamPreparedSource {
    CopiedStreamOwnedBuffer buffer;
    std::unique_ptr<CopiedStreamCompletionProof> producer;
};
struct CopiedStreamTransportPolicy {
    std::uint64_t maximum_payload_bytes = 32ULL * 1024ULL;
    std::uint64_t maximum_frame_bytes = 64ULL * 1024ULL;
    std::uint64_t maximum_completion_polls = 4096;
};
enum class CopiedStreamCleanupStatus { released, pending, failed_retained };

// Dedicated payload byte stream, independent of the control/command stream.
// Copied-payload envelope v1 nests the unchanged v2 boundary metadata codec.
// It has one in-flight operation and never supports GPU-address dereference,
// GPU DMA, striping or native leases. No numeric conversion is performed.
class CopiedStreamTransportBackend final : public TransportBackend {
public:
    CopiedStreamTransportBackend(DirectStreamControlByteStream&, CopiedStreamConnectionState&,
                                CopiedStreamTransportPolicy policy = {});
    // The TCP backend owns its stream and requires matching explicit ceilings.
    // Construction failure leaves the caller's stream unconsumed.
    CopiedStreamTransportBackend(TcpCopiedPayloadByteStream&&, CopiedStreamConnectionState&,
                                CopiedStreamTransportPolicy);
    CopiedStreamTransportBackend(SshBoundCopiedByteStream&&, CopiedStreamConnectionState&,
                                CopiedStreamTransportPolicy);
    ~CopiedStreamTransportBackend() override;
    CopiedStreamTransportBackend(const CopiedStreamTransportBackend&) = delete;
    CopiedStreamTransportBackend& operator=(const CopiedStreamTransportBackend&) = delete;
    [[nodiscard]] BackendCapabilities capabilities() const override;

    // Raw views cannot prove source ownership or producer/consumer completion.
    [[nodiscard]] TransferHandle prepare_send(const TransferDescriptor&, const TransportBufferView&) override;
    [[nodiscard]] TransferHandle prepare_receive(const TransferDescriptor&, const TransportBufferView&) override;
    [[nodiscard]] TransferHandle prepare_send(const DistributedBoundaryTransfer&, CopiedStreamPreparedSource);
    [[nodiscard]] TransferHandle prepare_receive(const DistributedBoundaryTransfer&, CopiedStreamOwnedBuffer);
    void submit(TransferHandle) override;
    [[nodiscard]] std::optional<Completion> poll() override;
    void cancel(TransferHandle) override;

    // Exposing RX memory retains it until an exact completed-consumer proof.
    [[nodiscard]] TransportBufferView receive_buffer(TransferHandle);
    void finish_consumer(TransferHandle, std::unique_ptr<CopiedStreamCompletionProof>);
    // Error cleanup can attach a later quiescence proof without ACK/recovery.
    void attach_cleanup_proof(TransferHandle, std::unique_ptr<CopiedStreamCompletionProof>);
    [[nodiscard]] CopiedStreamCleanupStatus drain_retained();
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] bool retains_buffer() const noexcept;
    [[nodiscard]] std::optional<TransferHandle> retained_handle() const noexcept;
    [[nodiscard]] const CopiedStreamConnection& connection() const noexcept;
private:
    CopiedStreamTransportBackend(DistributedControlByteStream&, std::uint64_t stream_frame_limit,
                                CopiedStreamConnectionState&, CopiedStreamTransportPolicy);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::transport

namespace strix {

// Source provider owns any preparation launched before a callback error.
// GPU-to-host staging must supply its own qualified event/visibility proof;
// returning a host view alone is insufficient.
class CopiedStreamBoundarySourceProvider {
public:
    virtual ~CopiedStreamBoundarySourceProvider() = default;
    [[nodiscard]] virtual transport::CopiedStreamPreparedSource stage_outbound_owned(
        const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) = 0;
};

// Bounded C1 bridge for existing distributed boundary execution. Inbound
// control metadata must already be queued by the existing v2 control server.
// Independent peer progress is required while send/receive waits. Transport
// ACK is not semantic model commit; request/state/rollback authority remains
// with the existing runtime and local executor.
class CopiedStreamDistributedBackend final : public DistributedExecutionBackend {
public:
    CopiedStreamDistributedBackend(transport::CopiedStreamTransportBackend&,
        const DistributedPlacementPlan&, DistributedRuntimeRequest,
        DistributedNativeControlPlane&, DistributedControlReceiver&,
        DistributedNativeLocalExecutor&, CopiedStreamBoundarySourceProvider&,
        DistributedBoundaryStagingProvider&, AllocationGenerationSource&,
        std::uint64_t maximum_progress_polls = 4096);
    void execute_local_step(const DistributedStepPlacement&) override;
    void execute_remote_step(const DistributedStepPlacement&) override;
    void send_boundary(const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) override;
    [[nodiscard]] DistributedBoundaryTransfer receive_boundary(const DistributedExecutionBoundary&) override;
    [[nodiscard]] bool failed() const noexcept { return failed_; }
private:
    void require_healthy() const;
    void validate_boundary(const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&,
                           bool receive, std::uint64_t sequence) const;
    transport::CopiedStreamTransportBackend* transport_;
    const DistributedPlacementPlan* placement_;
    DistributedRuntimeRequest request_;
    DistributedNativeControlPlane* control_;
    DistributedControlReceiver* receiver_;
    DistributedNativeLocalExecutor* local_;
    CopiedStreamBoundarySourceProvider* source_;
    DistributedBoundaryStagingProvider* consumer_;
    AllocationGenerationSource* generations_;
    std::uint64_t maximum_polls_;
    std::uint64_t next_send_ = 1, next_receive_ = 1;
    bool failed_ = false;
};

}  // namespace strix
