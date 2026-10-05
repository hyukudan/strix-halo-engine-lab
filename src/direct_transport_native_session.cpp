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
      sends_(std::move(other.sends_)) {
    other.backend_ = nullptr;
    other.leases_ = nullptr;
    other.link_ = nullptr;
    other.pools_ = nullptr;
    other.lease_ = {};
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

        other.backend_ = nullptr;
        other.leases_ = nullptr;
        other.link_ = nullptr;
        other.pools_ = nullptr;
        other.lease_ = {};
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
    return observe_direct_native_event(
        *backend_,
        receives_,
        sends_);
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
}

}  // namespace strix::transport
