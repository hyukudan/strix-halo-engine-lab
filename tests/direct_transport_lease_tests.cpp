#include "strix/direct_transport_lease.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::transport::DirectLinkLifecycle link() {
    strix::transport::DirectLinkLifecycle value;
    value.physical_link_id = "usb4-a";
    value.state =
        strix::transport::DirectLinkLifecycleState::
            direct_ready;
    value.connection_epoch = 17;
    value.control_link_ready = true;
    value.peer_reconciled = true;
    value.receive_ring_primed = true;
    value.exclusive_lease_available = true;
    value.completion_rearm_qualified = true;
    return value;
}

strix::transport::DirectTransportPoolMemoryBinding pools() {
    strix::transport::DirectTransportPoolMemoryBinding value;
    value.total_gpu_bytes = 8192;
    value.binding_identity = 0x1234ULL;
    value.links.resize(2);
    value.links[0].tx.address =
        static_cast<std::uintptr_t>(0x1000ULL);
    value.links[0].rx.address =
        static_cast<std::uintptr_t>(0x2000ULL);
    value.links[1].tx.address =
        static_cast<std::uintptr_t>(0x3000ULL);
    value.links[1].rx.address =
        static_cast<std::uintptr_t>(0x4000ULL);
    return value;
}

}  // namespace

int main() {
    auto lifecycle = link();
    const auto memory = pools();
    strix::transport::DirectTransportLeaseRegistry registry;

    const auto lease =
        registry.acquire(
            lifecycle,
            memory,
            0);
    expect(
        lease.lease_generation != 0 &&
            registry.leased(
                lifecycle.physical_link_id) &&
            registry.validate(
                lease,
                lifecycle,
                memory),
        "direct transport lease binds link epoch and physical pool identity");

    bool duplicate_rejected = false;
    try {
        static_cast<void>(
            registry.acquire(
                lifecycle,
                memory,
                1));
    } catch (const std::logic_error&) {
        duplicate_rejected = true;
    }
    expect(
        duplicate_rejected,
        "one physical link cannot hold two simultaneous direct leases");

    auto stale_epoch = lifecycle;
    ++stale_epoch.connection_epoch;
    expect(
        !registry.validate(
            lease,
            stale_epoch,
            memory),
        "connection epoch change invalidates direct transport lease");

    auto foreign_memory = memory;
    ++foreign_memory.binding_identity;
    expect(
        !registry.validate(
            lease,
            lifecycle,
            foreign_memory),
        "pool relocation/identity change invalidates direct transport lease");

    auto mutated = lease;
    ++mutated.lease_generation;
    bool mutated_release_rejected = false;
    try {
        registry.release(mutated);
    } catch (const std::invalid_argument&) {
        mutated_release_rejected = true;
    }
    expect(
        mutated_release_rejected &&
            registry.leased(
                lifecycle.physical_link_id),
        "mutated lease token cannot release the active physical link");

    registry.release(lease);
    expect(
        !registry.leased(
            lifecycle.physical_link_id) &&
            !registry.validate(
                lease,
                lifecycle,
                memory),
        "released direct transport lease cannot be reused");

    const auto replacement =
        registry.acquire(
            lifecycle,
            memory,
            1);
    expect(
        replacement.lease_generation !=
            lease.lease_generation &&
            replacement.link_index == 1,
        "new acquisition receives a fresh lease generation");

    auto unusable = lifecycle;
    unusable.receive_ring_primed = false;
    strix::transport::DirectTransportLeaseRegistry second;
    bool unusable_rejected = false;
    try {
        static_cast<void>(
            second.acquire(
                unusable,
                memory,
                0));
    } catch (const std::invalid_argument&) {
        unusable_rejected = true;
    }
    expect(
        unusable_rejected,
        "direct lease cannot be acquired before link becomes usable");

    if (failures == 0) {
        std::cout
            << "direct transport lease tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
