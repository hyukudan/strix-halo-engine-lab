#include "strix/direct_transport_native_session.hpp"

#include <stdexcept>
#include <utility>

namespace strix::transport {

DirectTransportNativeLinkSession::
DirectTransportNativeLinkSession(
    DirectTransportNativeLinkSession&& other) noexcept
    : backend_(other.backend_),
      leases_(other.leases_),
      link_(other.link_),
      pools_(other.pools_),
      lease_(std::move(other.lease_)),
      receives_(std::move(other.receives_)),
      sends_(std::move(other.sends_)),
      telemetry_(other.telemetry_),
      scheduler_(other.scheduler_) {
    other.backend_ = nullptr;
    other.leases_ = nullptr;
    other.link_ = nullptr;
    other.pools_ = nullptr;
    other.lease_ = {};
    other.telemetry_ = nullptr;
    other.scheduler_ = nullptr;
}

DirectTransportNativeLinkSession&
DirectTransportNativeLinkSession::operator=(
    DirectTransportNativeLinkSession&& other) noexcept {
    if (this != &other) {
        release_noexcept();
        backend_ = other.backend_;
        leases_ = other.leases_;
        link_ = other.link_;
        pools_ = other.pools_;
        lease_ = std::move(other.lease_);
        receives_ = std::move(other.receives_);
        sends_ = std::move(other.sends_);
        telemetry_ = other.telemetry_;
        scheduler_ = other.scheduler_;

        other.backend_ = nullptr;
        other.leases_ = nullptr;
        other.link_ = nullptr;
        other.pools_ = nullptr;
        other.lease_ = {};
        other.telemetry_ = nullptr;
        other.scheduler_ = nullptr;
    }
    return *this;
}

DirectTransportNativeLinkSession::
~DirectTransportNativeLinkSession() {
    release_noexcept();
}

DirectTransportNativeLinkSession
DirectTransportNativeLinkSession::open(
    DirectTransportNativeBackend& backend,
    DirectTransportLeaseRegistry& leases,
    const DirectLinkLifecycle& link,
    const DirectTransportPoolMemoryBinding& pools,
    std::uint32_t link_index) {
    const auto readiness =
        assess_direct_native_readiness(
            backend.capabilities(),
            backend.direct_capabilities());
    if (!readiness.ready) {
        throw std::logic_error(
            "native direct link session requires capability-complete backend");
    }

    auto lease =
        leases.acquire(
            link,
            pools,
            link_index);

    try {
        prepare_direct_native_link(
            backend,
            leases,
            link,
            pools,
            lease);
    } catch (...) {
        leases.release(lease);
        throw;
    }

    DirectTransportNativeLinkSession session;
    session.backend_ = &backend;
    session.leases_ = &leases;
    session.link_ = &link;
    session.pools_ = &pools;
    session.lease_ =
        std::move(lease);
    return session;
}

void DirectTransportNativeLinkSession::require_active() const {
    if (!active() ||
        leases_ == nullptr ||
        link_ == nullptr ||
        pools_ == nullptr) {
        throw std::logic_error(
            "native direct link session is not active");
    }
}

void DirectTransportNativeLinkSession::require_current_lease() const {
    require_active();
    if (!leases_->validate(
            lease_,
            *link_,
            *pools_)) {
        throw std::logic_error(
            "native direct link session lease is stale");
    }
}

namespace {

bool within_pool(
    const TransportBufferView& buffer,
    const TransportBufferView& pool,
    std::uint64_t required_bytes,
    bool receive) noexcept {
    if (pool.address == 0 ||
        pool.bytes == 0 ||
        buffer.address < pool.address ||
        buffer.allocation_generation == 0 ||
        buffer.allocation_generation !=
            pool.allocation_generation ||
        buffer.domain !=
            TransportMemoryDomain::gpu_transport_pool) {
        return false;
    }

    const auto offset =
        static_cast<std::uint64_t>(
            buffer.address - pool.address);
    if (offset > pool.bytes ||
        buffer.bytes > pool.bytes - offset ||
        required_bytes > buffer.bytes) {
        return false;
    }

    return receive
        ? validate_transport_receive_buffer(
              buffer,
              required_bytes)
        : validate_transport_send_buffer(
              buffer,
              required_bytes);
}

}  // namespace

TransferHandle DirectTransportNativeLinkSession::prepare_receive(
    const TransferDescriptor& descriptor,
    const TransportBufferView& destination) {
    require_current_lease();
    if (descriptor.connection_epoch !=
            lease_.connection_epoch ||
        lease_.link_index >=
            pools_->links.size() ||
        !within_pool(
            destination,
            pools_->links[lease_.link_index].rx,
            descriptor.byte_length,
            true)) {
        throw std::invalid_argument(
            "native receive buffer/descriptor is outside leased RX pool");
    }

    const auto handle =
        backend_->prepare_receive(
            descriptor,
            destination);
    register_receive(
        handle,
        descriptor);
    return handle;
}

TransferHandle DirectTransportNativeLinkSession::prepare_send(
    const TransferDescriptor& descriptor,
    const TransportBufferView& payload) {
    require_current_lease();
    if (descriptor.connection_epoch !=
            lease_.connection_epoch ||
        lease_.link_index >=
            pools_->links.size() ||
        !within_pool(
            payload,
            pools_->links[lease_.link_index].tx,
            descriptor.byte_length,
            false)) {
        throw std::invalid_argument(
            "native send buffer/descriptor is outside leased TX pool");
    }

    const auto handle =
        backend_->prepare_send(
            descriptor,
            payload);
    register_send(
        handle,
        descriptor);
    return handle;
}

void DirectTransportNativeLinkSession::submit(
    TransferHandle handle) {
    require_current_lease();
    if (handle == 0) {
        throw std::invalid_argument(
            "native submit requires non-zero handle");
    }
    backend_->submit(handle);
}

void DirectTransportNativeLinkSession::bind_telemetry(
    DualLinkCompletionTelemetry& telemetry,
    DualLinkRuntimeScheduler& scheduler) {
    require_current_lease();
    telemetry_ = &telemetry;
    scheduler_ = &scheduler;
}

void DirectTransportNativeLinkSession::submit_tracked(
    TransferHandle handle,
    LinkDirection direction,
    double submit_seconds,
    double one_way_latency_seconds,
    std::uint64_t in_flight_bytes) {
    require_current_lease();
    if (telemetry_ == nullptr ||
        scheduler_ == nullptr) {
        throw std::logic_error(
            "native tracked submit requires bound telemetry");
    }

    std::uint64_t bytes = 0;
    if (direction ==
        LinkDirection::outbound) {
        bytes =
            sends_.state(handle).
                expected_bytes;
    } else {
        bytes =
            receives_.state(handle).
                descriptor.byte_length;
    }

    telemetry_->register_transfer(
        DirectTelemetryStart{
            handle,
            lease_.link_index,
            direction,
            bytes,
            submit_seconds,
            one_way_latency_seconds,
            in_flight_bytes,
        });
    submit(handle);
}

void DirectTransportNativeLinkSession::cancel(
    TransferHandle handle) {
    require_current_lease();
    if (handle == 0) {
        throw std::invalid_argument(
            "native cancel requires non-zero handle");
    }
    backend_->cancel(handle);
}

void DirectTransportNativeLinkSession::register_receive(
    TransferHandle handle,
    const TransferDescriptor& descriptor) {
    require_current_lease();
    if (descriptor.connection_epoch !=
        lease_.connection_epoch) {
        throw std::invalid_argument(
            "native receive descriptor epoch does not match leased link");
    }
    receives_.register_receive(
        handle,
        descriptor);
}

void DirectTransportNativeLinkSession::register_send(
    TransferHandle handle,
    const TransferDescriptor& descriptor) {
    require_current_lease();
    if (descriptor.connection_epoch !=
            lease_.connection_epoch ||
        descriptor.byte_length == 0) {
        throw std::invalid_argument(
            "native send descriptor does not match leased link");
    }
    sends_.register_send(
        handle,
        descriptor.byte_length);
}

void DirectTransportNativeLinkSession::acquire_receive(
    TransferHandle handle) {
    require_current_lease();
    receives_.acquire_consumer(handle);
}

void DirectTransportNativeLinkSession::release_receive(
    TransferHandle handle) {
    require_current_lease();
    receives_.release_consumer(handle);
}

void DirectTransportNativeLinkSession::publish_receive_credit(
    TransferHandle handle) {
    require_current_lease();
    publish_direct_native_consumer_credit(
        *backend_,
        receives_,
        handle);
}

void DirectTransportNativeLinkSession::repost_receive(
    TransferHandle handle) {
    require_current_lease();
    repost_direct_native_receive(
        *backend_,
        receives_,
        handle);
}

void DirectTransportNativeLinkSession::publish_commit(
    TransferHandle handle,
    const TransferDescriptor& descriptor,
    const GpuCommitRecord& record) {
    require_current_lease();
    if (descriptor.connection_epoch !=
        lease_.connection_epoch) {
        throw std::invalid_argument(
            "native commit descriptor epoch does not match leased link");
    }
    publish_direct_native_commit(
        *backend_,
        handle,
        descriptor,
        record);
}

bool DirectTransportNativeLinkSession::poll_event() {
    require_current_lease();

    const auto event =
        backend_->poll_direct_event();
    if (!event.has_value()) {
        return false;
    }
    if (event->completion.handle == 0) {
        throw std::invalid_argument(
            "native direct transport event has zero handle");
    }

    if (event->receive_path) {
        receives_.observe(
            event->completion);
    } else {
        sends_.observe(
            event->completion);
    }

    if (telemetry_ != nullptr &&
        scheduler_ != nullptr &&
        telemetry_->tracked(
            event->completion.handle)) {
        if (event->monotonic_seconds <= 0.0) {
            throw std::invalid_argument(
                "native telemetry event requires monotonic timestamp");
        }
        telemetry_->observe_completion(
            event->completion,
            event->monotonic_seconds,
            event->in_flight_bytes,
            *scheduler_);
    }
    return true;
}

const DirectReceiveCompletionState&
DirectTransportNativeLinkSession::receive_state(
    TransferHandle handle) const {
    require_active();
    return receives_.state(handle);
}

const DirectSendCompletionState&
DirectTransportNativeLinkSession::send_state(
    TransferHandle handle) const {
    require_active();
    return sends_.state(handle);
}

void DirectTransportNativeLinkSession::close() {
    require_active();

    backend_->release_link_pool(
        lease_);
    leases_->release(
        lease_);

    backend_ = nullptr;
    leases_ = nullptr;
    link_ = nullptr;
    pools_ = nullptr;
    lease_ = {};
    telemetry_ = nullptr;
    scheduler_ = nullptr;
}

void DirectTransportNativeLinkSession::release_noexcept() noexcept {
    if (!active()) {
        return;
    }

    backend_->release_link_pool(
        lease_);
    if (leases_ != nullptr) {
        try {
            leases_->release(
                lease_);
        } catch (...) {
        }
    }

    backend_ = nullptr;
    leases_ = nullptr;
    link_ = nullptr;
    pools_ = nullptr;
    lease_ = {};
    telemetry_ = nullptr;
    scheduler_ = nullptr;
}

}  // namespace strix::transport
