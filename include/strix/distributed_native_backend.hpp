#pragma once

#include "strix/direct_transport_native_session.hpp"
#include "strix/distributed_runtime.hpp"

#include <cstdint>
#include <span>

namespace strix {

class DistributedNativeControlPlane {
public:
    virtual ~DistributedNativeControlPlane() = default;

    virtual void execute_remote_step(
        const DistributedStepPlacement& step) = 0;

    virtual void publish_boundary_transfer(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;

    [[nodiscard]] virtual DistributedBoundaryTransfer
    await_boundary_transfer(
        const DistributedExecutionBoundary& boundary) = 0;
};

class DistributedNativeLocalExecutor {
public:
    virtual ~DistributedNativeLocalExecutor() = default;

    virtual void execute_local_step(
        const DistributedStepPlacement& step) = 0;
};

class DistributedBoundaryStagingProvider {
public:
    virtual ~DistributedBoundaryStagingProvider() = default;

    [[nodiscard]] virtual transport::TransportBufferView
    stage_outbound(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;

    [[nodiscard]] virtual transport::TransportBufferView
    stage_inbound(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;

    virtual void consume_inbound(
        const DistributedExecutionBoundary& boundary,
        const transport::TransportBufferView& staged) = 0;

    [[nodiscard]] virtual transport::TransportBufferView
    stage_outbound_slice(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer,
        const transport::LinkSlice& slice,
        std::uint16_t stripe_index,
        std::uint16_t stripe_count);

    [[nodiscard]] virtual transport::TransportBufferView
    stage_inbound_slice(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer,
        const transport::LinkSlice& slice,
        std::uint16_t stripe_index,
        std::uint16_t stripe_count);

    virtual void consume_inbound_slice(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer,
        const transport::LinkSlice& slice,
        std::uint16_t stripe_index,
        std::uint16_t stripe_count,
        const transport::TransportBufferView& staged);

    virtual void finish_inbound(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer);
};

struct DistributedNativeBackendPolicy {
    std::uint64_t maximum_event_polls = 4096;
};

class NativeDistributedBackend final
    : public DistributedExecutionBackend {
public:
    NativeDistributedBackend(
        std::span<transport::DirectTransportNativeLinkSession*> sessions,
        DistributedNativeControlPlane& control,
        DistributedNativeLocalExecutor& local,
        DistributedBoundaryStagingProvider& staging,
        DistributedNativeBackendPolicy policy = {});

    void execute_local_step(
        const DistributedStepPlacement& step) override;

    void execute_remote_step(
        const DistributedStepPlacement& step) override;

    void send_boundary(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) override;

    [[nodiscard]] DistributedBoundaryTransfer
    receive_boundary(
        const DistributedExecutionBoundary& boundary) override;

private:
    [[nodiscard]] transport::DirectTransportNativeLinkSession&
    session_for_link(
        std::size_t link_index) const;

    void wait_send_reusable(
        transport::DirectTransportNativeLinkSession& session,
        transport::TransferHandle handle) const;

    void wait_receive_complete(
        transport::DirectTransportNativeLinkSession& session,
        transport::TransferHandle handle) const;

    void wait_receive_reusable(
        transport::DirectTransportNativeLinkSession& session,
        transport::TransferHandle handle) const;

    std::span<transport::DirectTransportNativeLinkSession*> sessions_;
    DistributedNativeControlPlane* control_ = nullptr;
    DistributedNativeLocalExecutor* local_ = nullptr;
    DistributedBoundaryStagingProvider* staging_ = nullptr;
    DistributedNativeBackendPolicy policy_{};
};

using SingleLinkNativeDistributedBackend =
    NativeDistributedBackend;

}  // namespace strix
