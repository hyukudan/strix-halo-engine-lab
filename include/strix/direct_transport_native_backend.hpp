#pragma once

#include "strix/direct_transport_completion.hpp"
#include "strix/direct_transport_lease.hpp"
#include "strix/direct_transport_pool_memory.hpp"
#include "strix/transport.hpp"

#include <optional>
#include <string>
#include <vector>

namespace strix::transport {

struct DirectNativeCapabilities {
    bool dma_buf_export = false;
    bool dma_buf_import = false;
    bool uncached_gpu_pool = false;
    bool completion_event_source = false;
    bool system_scope_gpu_commit = false;
    bool remote_consumer_credit = false;
    bool receive_repost = false;
};

struct DirectNativeReadiness {
    bool ready = false;
    std::vector<std::string> blockers;
};

struct DirectNativeEvent {
    Completion completion;
    bool receive_path = false;
    double monotonic_seconds = 0.0;
    std::uint64_t in_flight_bytes = 0;
};

class DirectTransportNativeBackend {
public:
    virtual ~DirectTransportNativeBackend() = default;

    [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;
    [[nodiscard]] virtual DirectNativeCapabilities direct_capabilities() const = 0;

    virtual void import_link_pool(
        const DirectTransportLease& lease,
        const DirectTransportLinkPoolBinding& pool) = 0;

    virtual void release_link_pool(
        const DirectTransportLease& lease) noexcept = 0;

    [[nodiscard]] virtual TransferHandle prepare_send(
        const TransferDescriptor& descriptor,
        const TransportBufferView& payload);

    [[nodiscard]] virtual TransferHandle prepare_receive(
        const TransferDescriptor& descriptor,
        const TransportBufferView& destination);

    virtual void submit(
        TransferHandle handle);

    virtual void cancel(
        TransferHandle handle);

    virtual void publish_gpu_commit(
        TransferHandle handle,
        const GpuCommitRecord& record) = 0;

    virtual void publish_consumer_credit(
        TransferHandle handle,
        std::uint64_t bytes) = 0;

    virtual void repost_receive(
        TransferHandle handle) = 0;

    [[nodiscard]] virtual std::optional<DirectNativeEvent>
    poll_direct_event() = 0;
};

[[nodiscard]] DirectNativeReadiness assess_direct_native_readiness(
    const BackendCapabilities& backend,
    const DirectNativeCapabilities& native);

void prepare_direct_native_link(
    DirectTransportNativeBackend& backend,
    const DirectTransportLeaseRegistry& leases,
    const DirectLinkLifecycle& link,
    const DirectTransportPoolMemoryBinding& pools,
    const DirectTransportLease& lease);

void publish_direct_native_commit(
    DirectTransportNativeBackend& backend,
    TransferHandle handle,
    const TransferDescriptor& descriptor,
    const GpuCommitRecord& record);

void publish_direct_native_consumer_credit(
    DirectTransportNativeBackend& backend,
    DirectReceiveCompletionBridge& receives,
    TransferHandle handle);

void repost_direct_native_receive(
    DirectTransportNativeBackend& backend,
    DirectReceiveCompletionBridge& receives,
    TransferHandle handle);

bool observe_direct_native_event(
    DirectTransportNativeBackend& backend,
    DirectReceiveCompletionBridge& receives,
    DirectSendCompletionBridge& sends);

}  // namespace strix::transport
