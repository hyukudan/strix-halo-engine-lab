#include "strix/direct_transport_lease.hpp"

#include <limits>
#include <stdexcept>

namespace strix::transport {

DirectTransportLease
DirectTransportLeaseRegistry::acquire(
    const DirectLinkLifecycle& link,
    const DirectTransportPoolMemoryBinding& pools,
    std::uint32_t link_index) {
    if (link.physical_link_id.empty() ||
        !direct_link_usable(link) ||
        link.connection_epoch == 0 ||
        pools.binding_identity == 0 ||
        link_index >= pools.links.size()) {
        throw std::invalid_argument(
            "direct transport lease requires usable link and valid pool binding");
    }
    if (leased(link.physical_link_id)) {
        throw std::logic_error(
            "direct transport physical link already has an active lease");
    }
    if (next_generation_ ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "direct transport lease generation exhausted");
    }

    DirectTransportLease lease;
    lease.physical_link_id =
        link.physical_link_id;
    lease.connection_epoch =
        link.connection_epoch;
    lease.pool_binding_identity =
        pools.binding_identity;
    lease.link_index =
        link_index;
    lease.lease_generation =
        next_generation_++;

    active_.emplace(
        lease.physical_link_id,
        lease);
    return lease;
}

void DirectTransportLeaseRegistry::release(
    const DirectTransportLease& lease) {
    const auto found =
        active_.find(
            lease.physical_link_id);
    if (found == active_.end()) {
        throw std::invalid_argument(
            "direct transport lease is not active");
    }
    const auto& active =
        found->second;
    if (active.connection_epoch !=
            lease.connection_epoch ||
        active.pool_binding_identity !=
            lease.pool_binding_identity ||
        active.link_index !=
            lease.link_index ||
        active.lease_generation !=
            lease.lease_generation) {
        throw std::invalid_argument(
            "direct transport lease token is stale or mutated");
    }
    active_.erase(found);
}

bool DirectTransportLeaseRegistry::validate(
    const DirectTransportLease& lease,
    const DirectLinkLifecycle& link,
    const DirectTransportPoolMemoryBinding& pools) const noexcept {
    if (lease.physical_link_id.empty() ||
        lease.physical_link_id !=
            link.physical_link_id ||
        lease.connection_epoch == 0 ||
        lease.connection_epoch !=
            link.connection_epoch ||
        lease.pool_binding_identity == 0 ||
        lease.pool_binding_identity !=
            pools.binding_identity ||
        lease.link_index >=
            pools.links.size() ||
        lease.lease_generation == 0 ||
        !direct_link_usable(link)) {
        return false;
    }

    const auto found =
        active_.find(
            lease.physical_link_id);
    if (found == active_.end()) {
        return false;
    }
    const auto& active =
        found->second;
    return active.connection_epoch ==
               lease.connection_epoch &&
           active.pool_binding_identity ==
               lease.pool_binding_identity &&
           active.link_index ==
               lease.link_index &&
           active.lease_generation ==
               lease.lease_generation;
}

bool DirectTransportLeaseRegistry::leased(
    const std::string& physical_link_id) const noexcept {
    return active_.find(
               physical_link_id) !=
           active_.end();
}

void DirectTransportLeaseRegistry::require_current_token(
    const DirectTransportLease& lease) const {
    const auto found = active_.find(lease.physical_link_id);
    if (found == active_.end() ||
        found->second.connection_epoch != lease.connection_epoch ||
        found->second.pool_binding_identity != lease.pool_binding_identity ||
        found->second.link_index != lease.link_index ||
        found->second.lease_generation != lease.lease_generation) {
        throw std::invalid_argument("physical sequence reservation requires current lease token");
    }
}

std::uint64_t DirectTransportLeaseRegistry::reserve_send_sequence(
    const DirectTransportLease& lease) {
    require_current_token(lease);
    auto& state = sequence_history_[lease.physical_link_id][lease.connection_epoch];
    if (state.next_send == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("native physical send sequence exhausted");
    }
    return state.next_send++;
}

void DirectTransportLeaseRegistry::reserve_receive_sequence(
    const DirectTransportLease& lease, std::uint64_t sequence) {
    require_current_token(lease);
    auto& state = sequence_history_[lease.physical_link_id][lease.connection_epoch];
    if (state.receive_quarantined) {
        throw std::logic_error("native RX epoch is quarantined");
    }
    if (sequence == 0 || sequence <= state.last_receive) {
        throw std::invalid_argument("native physical receive sequence is stale or duplicate");
    }
    state.last_receive = sequence;
}

void DirectTransportLeaseRegistry::quarantine_receive(
    const DirectTransportLease& lease) {
    require_current_token(lease);
    sequence_history_[lease.physical_link_id][lease.connection_epoch].receive_quarantined = true;
}

bool DirectTransportLeaseRegistry::receive_quarantined(
    const DirectTransportLease& lease) const {
    require_current_token(lease);
    const auto physical = sequence_history_.find(lease.physical_link_id);
    if (physical == sequence_history_.end()) return false;
    const auto epoch = physical->second.find(lease.connection_epoch);
    return epoch != physical->second.end() && epoch->second.receive_quarantined;
}

}  // namespace strix::transport
