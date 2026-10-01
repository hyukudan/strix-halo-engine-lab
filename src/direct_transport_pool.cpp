#include "strix/direct_transport_pool.hpp"

#include <limits>
#include <stdexcept>

namespace strix::transport {
namespace {

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() /
                left) {
        throw std::overflow_error(
            "direct transport pool multiplication overflow");
    }
    return left * right;
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() -
            left) {
        throw std::overflow_error(
            "direct transport pool addition overflow");
    }
    return left + right;
}

}  // namespace

DirectTransportPoolPlan
plan_direct_transport_pool(
    const DirectFrameGeometry& geometry) {
    if (geometry.frame_bytes == 0 ||
        geometry.ring_frames < 2) {
        throw std::invalid_argument(
            "direct transport pool requires at least two non-zero frames");
    }

    DirectTransportPoolPlan plan;
    plan.geometry = geometry;
    plan.pool_alignment =
        geometry.frame_bytes;

    const auto pool_bytes =
        checked_mul(
            geometry.frame_bytes,
            geometry.ring_frames);
    plan.tx_pool_bytes = pool_bytes;
    plan.rx_pool_bytes = pool_bytes;
    plan.total_gpu_bytes =
        checked_add(
            plan.tx_pool_bytes,
            plan.rx_pool_bytes);

    const auto terminal_frames =
        geometry.dedicated_terminal_frame
            ? 1ULL
            : 0ULL;
    if (terminal_frames >=
        geometry.ring_frames) {
        throw std::invalid_argument(
            "direct transport ring leaves no payload frame");
    }
    plan.maximum_single_message_payload_bytes =
        checked_mul(
            static_cast<std::uint64_t>(
                geometry.ring_frames) -
                terminal_frames,
            geometry.frame_bytes);

    return plan;
}

MultiLinkTransportPoolPlan
plan_multi_link_transport_pools(
    std::span<const DirectFrameGeometry> links) {
    if (links.empty()) {
        throw std::invalid_argument(
            "multi-link pool plan requires at least one physical link");
    }

    MultiLinkTransportPoolPlan plan;
    plan.links.reserve(
        links.size());

    for (const auto& geometry : links) {
        auto link =
            plan_direct_transport_pool(
                geometry);
        plan.total_gpu_bytes =
            checked_add(
                plan.total_gpu_bytes,
                link.total_gpu_bytes);
        plan.links.push_back(
            link);
    }

    return plan;
}

}  // namespace strix::transport
