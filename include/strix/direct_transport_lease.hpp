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

private:
    std::uint64_t next_generation_ = 1;
    std::unordered_map<std::string, DirectTransportLease>
        active_;
};

}  // namespace strix::transport
