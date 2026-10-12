#pragma once

#include "strix/direct_transport_native_backend.hpp"

#include <memory>

namespace strix::transport {

// These are host-owned adapter records, not a kernel UAPI or wire layout.
// A supported driver implementation must translate its documented ABI into
// these records without deriving positive proof from sysfs/module discovery.
struct DirectDriverBinding {
    std::uint32_t adapter_contract_version = 1;
    std::string device_identity;
    std::uint64_t device_instance_generation = 0;
    DirectNativeInterfaceIdentity interface;
    bool operator==(const DirectDriverBinding&) const = default;
};

struct DirectDriverQuery {
    DirectDriverBinding binding;
    BackendCapabilities backend;
    DirectNativeCapabilities native;
};

struct DirectDriverImportBinding {
    DirectDriverBinding device;
    DirectTransportLease lease;
    DirectTransportLinkPoolBinding pool;
    std::uint64_t import_generation = 0;
};

// The concrete token owns the native exclusive-open/import/export references
// and pins the underlying allocations. Pending DMA must survive destruction
// of a session or its caller's allocation-owner references.
class DirectDriverImport {
public:
    virtual ~DirectDriverImport() = default;
    [[nodiscard]] virtual const DirectDriverImportBinding& binding() const noexcept = 0;
};

struct DirectDriverOperation {
    DirectDriverImportBinding imported;
    TransferHandle handle = 0;
    TransferDescriptor descriptor;
    TransportBufferView buffer;
    bool receive_path = false;
};

enum class DirectDriverVisibility {
    unavailable,
    producer_release_to_external_dma,
    consumer_acquire_from_external_dma,
    consumer_release_complete,
};

struct DirectDriverVisibilityProof {
    DirectDriverOperation operation;
    DirectDriverVisibility visibility = DirectDriverVisibility::unavailable;
};

struct DirectDriverCommitRead {
    GpuCommitRecord record;
    DirectDriverVisibilityProof acquire;
};

struct DirectDriverEvent {
    DirectDriverOperation operation;
    DirectNativeEvent event;
};

enum class DirectDriverQuiescenceStatus { pending, quiescent, failed_retained };
// Quiescent means neither DMA nor any GPU producer/consumer can access this
// entire import. It is not merely an empty completion queue or cancel return.
struct DirectDriverQuiescence {
    DirectDriverImportBinding imported;
    DirectDriverQuiescenceStatus status = DirectDriverQuiescenceStatus::pending;
};

// Trusted normalized syscall/GPU boundary. Its default implementation is
// unavailable and performs no guessed ioctl, export, import or DMA operation.
// The provider outlives the adapter and invalidates its cached binding on any
// device replacement, driver/ABI/peer/epoch change or lost qualification.
// Each operation must be atomically fenced by the supplied binding at the
// real driver boundary; the host adapter cannot prove that property itself.
class DirectTransportDriverIo {
public:
    virtual ~DirectTransportDriverIo() = default;
    [[nodiscard]] virtual DirectDriverQuery query();
    // Cheap cached inspection; it must not issue an I/O query per progress call.
    [[nodiscard]] virtual std::optional<DirectDriverBinding> current_binding() const;
    // Prepare owns/pins resources but may not arm or enqueue DMA. Activation
    // may partially succeed, so the adapter retains the token before calling it.
    [[nodiscard]] virtual std::shared_ptr<DirectDriverImport> prepare_import(
        const DirectDriverBinding&, const DirectTransportLease&,
        const DirectTransportLinkPoolBinding&);
    virtual void activate_import(const std::shared_ptr<DirectDriverImport>&);
    virtual void prepare(const DirectDriverOperation&);
    virtual void submit(const DirectDriverOperation&);
    virtual void cancel(const DirectDriverOperation&);
    [[nodiscard]] virtual DirectDriverVisibilityProof publish_gpu_commit(
        const DirectDriverOperation&, const GpuCommitRecord&);
    [[nodiscard]] virtual DirectDriverCommitRead read_gpu_commit(const DirectDriverOperation&);
    // Only a completed consumer release event can produce this proof. Return
    // from a CPU callback or RX DMA completion alone cannot authorize credit.
    [[nodiscard]] virtual DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation&);
    virtual void publish_consumer_credit(const DirectDriverOperation&, std::uint64_t);
    virtual void repost_receive(const DirectDriverOperation&);
    [[nodiscard]] virtual std::optional<DirectDriverEvent> poll_event();
    virtual void retire(const DirectDriverOperation&);
    virtual void request_import_release(const std::shared_ptr<DirectDriverImport>&);
    [[nodiscard]] virtual DirectDriverQuiescence poll_import_quiescence(
        const std::shared_ptr<DirectDriverImport>&);
    virtual void release_import(const std::shared_ptr<DirectDriverImport>&);
};

struct DirectDriverPolicy {
    std::uint64_t maximum_payload_bytes = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t maximum_imported_bytes = 128ULL * 1024ULL * 1024ULL;
    std::uint32_t maximum_operations = 256;
    std::uint64_t release_polls = 4096;
};

// One physical device/connection and one serialized progress owner. This
// implements the existing native-session interface without qualifying native
// hardware. Errors are sticky; a pending import cannot be silently replaced.
// Directional sequence history survives same-epoch import reopen. The session
// lease registry must outlive connection epochs; replacing a backend/runtime
// requires drained consumers/imports and fresh peer-agreed epochs. A destructor
// with unproven quiescence retains bounded resources until process teardown;
// it cannot serve as normal cleanup or automatic recovery.
class DirectTransportDriverBackend final : public DirectTransportNativeBackend {
public:
    DirectTransportDriverBackend(DirectTransportDriverIo&, DirectDriverBinding expected,
                                 DirectDriverPolicy policy = {});
    ~DirectTransportDriverBackend() override;
    DirectTransportDriverBackend(const DirectTransportDriverBackend&) = delete;
    DirectTransportDriverBackend& operator=(const DirectTransportDriverBackend&) = delete;

    [[nodiscard]] BackendCapabilities capabilities() const override;
    [[nodiscard]] DirectNativeCapabilities direct_capabilities() const override;
    void import_link_pool(const DirectTransportLease&, const DirectTransportLinkPoolBinding&) override;
    DirectPoolReleaseStatus release_link_pool(const DirectTransportLease&) noexcept override;
    [[nodiscard]] TransferHandle prepare_send(const TransferDescriptor&, const TransportBufferView&) override;
    [[nodiscard]] TransferHandle prepare_receive(const TransferDescriptor&, const TransportBufferView&) override;
    void submit(TransferHandle) override;
    void cancel(TransferHandle) override;
    void publish_gpu_commit(TransferHandle, const GpuCommitRecord&) override;
    [[nodiscard]] GpuCommitRecord read_gpu_commit(TransferHandle) override;
    void publish_consumer_credit(TransferHandle, std::uint64_t) override;
    void repost_receive(TransferHandle) override;
    [[nodiscard]] std::optional<DirectNativeEvent> poll_direct_event() override;

    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] bool retains_import() const noexcept;
    [[nodiscard]] std::uint32_t active_operations() const noexcept;

private:
    struct Impl;
    [[nodiscard]] TransferHandle prepare(const TransferDescriptor&, const TransportBufferView&, bool receive);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::transport
