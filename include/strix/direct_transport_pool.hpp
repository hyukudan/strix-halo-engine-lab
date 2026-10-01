#pragma once

#include "strix/direct_transport_frames.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix::transport {

struct DirectTransportPoolPlan {
    DirectFrameGeometry geometry;
    std::uint64_t tx_pool_bytes = 0;
    std::uint64_t rx_pool_bytes = 0;
    std::uint64_t total_gpu_bytes = 0;
    std::uint64_t pool_alignment = 0;
    std::uint64_t maximum_single_message_payload_bytes = 0;
};

struct MultiLinkTransportPoolPlan {
    std::vector<DirectTransportPoolPlan> links;
    std::uint64_t total_gpu_bytes = 0;
};

[[nodiscard]] DirectTransportPoolPlan
plan_direct_transport_pool(
    const DirectFrameGeometry& geometry);

[[nodiscard]] MultiLinkTransportPoolPlan
plan_multi_link_transport_pools(
    std::span<const DirectFrameGeometry> links);

}  // namespace strix::transport
