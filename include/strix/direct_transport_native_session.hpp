#pragma once

#include "strix/direct_transport_native_backend.hpp"

#include <cstdint>

namespace strix::transport {

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

    [[nodiscard]] TransferHandle prepare_receive(
        const TransferDescriptor& descriptor,
        const TransportBufferView& destination);

    [[nodiscard]] TransferHandle prepare_send(
        const TransferDescriptor& descriptor,
        const TransportBufferView& payload);

    void submit(
        TransferHandle handle);

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

    [[nodiscard]] bool poll_event();

    [[nodiscard]] const DirectReceiveCompletionState&
    receive_state(
        TransferHandle handle) const;

    [[nodiscard]] const DirectSendCompletionState&
    send_state(
        TransferHandle handle) const;

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
    DirectReceiveCompletionBridge receives_;
    DirectSendCompletionBridge sends_;
};

}  // namespace strix::transport
