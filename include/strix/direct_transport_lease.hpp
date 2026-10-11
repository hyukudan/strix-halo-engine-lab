#pragma once

#include "strix/direct_transport_pool_memory.hpp"
#include "strix/transport.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace strix::transport {

struct DirectTransportLease {
    std::string physical_link_id;
    std::uint64_t connection_epoch = 0;
    std::uint64_t pool_binding_identity = 0;
    std::uint32_t link_index = 0;
    std::uint64_t lease_generation = 0;
};

// One serialized owner also governs every session using this registry. Epoch
// sequence/replay state survives lease release and same-epoch reacquisition.
// The registry must outlive its physical connection epochs. Replacing it or
// restarting its owner requires drained backends/consumers and fresh,
// peer-agreed epochs before any native traffic is admitted.
class DirectTransportLeaseRegistry {
public:
    [[nodiscard]] DirectTransportLease acquire(
        const DirectLinkLifecycle& link,
        const DirectTransportPoolMemoryBinding& pools,
        std::uint32_t link_index);

    void release(
        const DirectTransportLease& lease);

    [[nodiscard]] bool validate(
        const DirectTransportLease& lease,
        const DirectLinkLifecycle& link,
        const DirectTransportPoolMemoryBinding& pools) const noexcept;

    [[nodiscard]] bool leased(
        const std::string& physical_link_id) const noexcept;

    [[nodiscard]] std::uint64_t reserve_send_sequence(
        const DirectTransportLease& lease);
    void reserve_receive_sequence(
        const DirectTransportLease& lease, std::uint64_t sequence);
    void quarantine_receive(const DirectTransportLease& lease);
    [[nodiscard]] bool receive_quarantined(
        const DirectTransportLease& lease) const;

private:
    void require_current_token(const DirectTransportLease& lease) const;
    struct EpochSequences {
        std::uint64_t next_send = 1;
        std::uint64_t last_receive = 0;
        bool receive_quarantined = false;
    };
    std::uint64_t next_generation_ = 1;
    std::unordered_map<std::string, DirectTransportLease>
        active_;
    std::unordered_map<std::string,
        std::unordered_map<std::uint64_t, EpochSequences>> sequence_history_;
};

}  // namespace strix::transport
