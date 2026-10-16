#pragma once

#include "strix/direct_transport_native_backend.hpp"
#include "strix/dual_link_telemetry.hpp"

#include <cstdint>

namespace strix::transport {

// One serialized progress owner per physical session. These stateful APIs
// are not thread-safe; adapters sharing a session must serialize reservations,
// preparation, control publication and completion progress in link order.
class DirectTransportNativeLinkSession {
public:
    DirectTransportNativeLinkSession() = default;
    DirectTransportNativeLinkSession(
        const DirectTransportNativeLinkSession&) = delete;
    DirectTransportNativeLinkSession& operator=(
        const DirectTransportNativeLinkSession&) = delete;

    DirectTransportNativeLinkSession(
        DirectTransportNativeLinkSession&& other) noexcept;
    DirectTransportNativeLinkSession& operator=(
        DirectTransportNativeLinkSession&& other) noexcept;

    ~DirectTransportNativeLinkSession();

    [[nodiscard]] static DirectTransportNativeLinkSession open(
        DirectTransportNativeBackend& backend,
        DirectTransportLeaseRegistry& leases,
        const DirectLinkLifecycle& link,
        const DirectTransportPoolMemoryBinding& pools,
        std::uint32_t link_index);

    [[nodiscard]] bool active() const noexcept {
        return backend_ != nullptr;
    }

    [[nodiscard]] const DirectTransportLease&
    lease() const noexcept {
        return lease_;
    }

    // Sequences are unique within this physical epoch and direction, across
    // logical requests and same-epoch lease reopen. Failed preparation burns
    // a reservation deliberately; the lease registry retains the history.
    [[nodiscard]] std::uint64_t reserve_send_sequence();
    void reserve_receive_sequence(std::uint64_t sequence);

    // Fail closed after an unretired receive. This lease's RX pool cannot be
    // prepared, released or reposted again by another distributed adapter.
    void quarantine_receive(TransferHandle handle);
    [[nodiscard]] bool receive_quarantined() const noexcept {
        return receive_quarantined_;
    }

    [[nodiscard]] TransferHandle prepare_receive(
        const TransferDescriptor& descriptor,
        const TransportBufferView& destination);

    [[nodiscard]] TransferHandle prepare_send(
        const TransferDescriptor& descriptor,
        const TransportBufferView& payload);

    void submit(
        TransferHandle handle);

    void bind_telemetry(
        DualLinkCompletionTelemetry& telemetry,
        DualLinkRuntimeScheduler& scheduler);

    void submit_tracked(
        TransferHandle handle,
        LinkDirection direction,
        double submit_seconds,
        double one_way_latency_seconds,
        std::uint64_t in_flight_bytes);

    void cancel(
        TransferHandle handle);

    void register_receive(
        TransferHandle handle,
        const TransferDescriptor& descriptor);

    void register_send(
        TransferHandle handle,
        const TransferDescriptor& descriptor);

    void acquire_receive(
        TransferHandle handle);

    void release_receive(
        TransferHandle handle);

    void publish_receive_credit(
        TransferHandle handle);

    void repost_receive(
        TransferHandle handle);

    void publish_commit(
        TransferHandle handle,
        const TransferDescriptor& descriptor,
        const GpuCommitRecord& record);

    [[nodiscard]] GpuCommitRecord read_commit(
        TransferHandle handle,
        const TransferDescriptor& descriptor);

    [[nodiscard]] bool poll_event();

    [[nodiscard]] const DirectReceiveCompletionState&
    receive_state(
        TransferHandle handle) const;

    [[nodiscard]] const DirectSendCompletionState&
    send_state(
        TransferHandle handle) const;

    // Revalidate actual current lease/interface/pools and native readiness before
    // reading operation completion. A copied ACK or old completion map is not
    // sufficient. No polling, mutation, device sync or new allocation here in
    // this class; provider capability queries retain their own API contract.
    void require_current_retirement_lease(const DirectTransportLease& expected) const;
    void require_retired_send(TransferHandle, const DirectTransportLease& expected,
                              std::uint64_t expected_bytes) const;
    void require_retired_receive(TransferHandle, const DirectTransportLease& expected,
                                 const TransferDescriptor& expected_descriptor) const;

    // On unproved backend release, throws while retaining the active host lease
    // and fencing new work. Retry close only to obtain actual release proof.
    void close();

private:
    void require_active() const;
    void require_current_lease() const;
    void release_noexcept() noexcept;

    DirectTransportNativeBackend* backend_ = nullptr;
    DirectTransportLeaseRegistry* leases_ = nullptr;
    const DirectLinkLifecycle* link_ = nullptr;
    const DirectTransportPoolMemoryBinding* pools_ = nullptr;
    DirectTransportLease lease_;
    DirectNativeInterfaceIdentity native_interface_;
    DirectReceiveCompletionBridge receives_;
    DirectSendCompletionBridge sends_;
    DualLinkCompletionTelemetry* telemetry_ = nullptr;
    DualLinkRuntimeScheduler* scheduler_ = nullptr;
    bool receive_quarantined_ = false;
    bool closing_ = false;
};

}  // namespace strix::transport
