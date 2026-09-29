#include "strix/transport.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::transport {
namespace {

bool available(const LinkProfile& link) noexcept {
    return link.health == LinkHealth::qualified &&
           std::isfinite(link.payload_bytes_per_second) &&
           link.payload_bytes_per_second > 0.0 &&
           std::isfinite(link.one_way_latency_seconds) &&
           link.one_way_latency_seconds >= 0.0;
}

TransferPlan single_plan(std::span<const LinkProfile> links, std::uint64_t bytes) {
    TransferPlan plan;
    double best = std::numeric_limits<double>::infinity();
    std::size_t best_index = 0;
    bool found = false;

    for (std::size_t index = 0; index < links.size(); ++index) {
        if (!available(links[index])) {
            continue;
        }
        const auto estimate = estimate_single_link_seconds(links[index], bytes);
        if (!found || estimate < best ||
            (estimate == best && links[index].id < links[best_index].id)) {
            found = true;
            best = estimate;
            best_index = index;
        }
    }

    if (!found) {
        return plan;
    }

    plan.kind = PlanKind::single_link;
    plan.estimated_seconds = best;
    plan.slices.push_back(LinkSlice{best_index, 0, bytes});
    return plan;
}

TransferPlan striped_pair(std::span<const LinkProfile> links,
                          std::uint64_t bytes,
                          std::size_t left,
                          std::size_t right,
                          double reassembly_seconds) {
    const auto& a = links[left];
    const auto& b = links[right];
    const auto bandwidth_sum = a.payload_bytes_per_second + b.payload_bytes_per_second;
    if (!(bandwidth_sum > 0.0) || !std::isfinite(bandwidth_sum)) {
        return {};
    }

    const long double fraction =
        static_cast<long double>(a.payload_bytes_per_second) /
        static_cast<long double>(bandwidth_sum);
    auto first_bytes = static_cast<std::uint64_t>(
        static_cast<long double>(bytes) * fraction);
    first_bytes = std::min(first_bytes, bytes);
    const auto second_bytes = bytes - first_bytes;

    if (first_bytes == 0 || second_bytes == 0) {
        return {};
    }

    const auto first_seconds = estimate_single_link_seconds(a, first_bytes);
    const auto second_seconds = estimate_single_link_seconds(b, second_bytes);
    if (!std::isfinite(first_seconds) || !std::isfinite(second_seconds)) {
        return {};
    }

    TransferPlan plan;
    plan.kind = PlanKind::striped;
    plan.estimated_seconds = std::max(first_seconds, second_seconds) +
                             std::max(0.0, reassembly_seconds);
    plan.slices.push_back(LinkSlice{left, 0, first_bytes});
    plan.slices.push_back(LinkSlice{right, first_bytes, second_bytes});
    return plan;
}

}  // namespace

bool can_transition(BufferOwnership from, BufferOwnership to) noexcept {
    switch (from) {
    case BufferOwnership::producer_owned:
        return to == BufferOwnership::producer_released;
    case BufferOwnership::producer_released:
        return to == BufferOwnership::transport_owned;
    case BufferOwnership::transport_owned:
        return to == BufferOwnership::receive_complete;
    case BufferOwnership::receive_complete:
        return to == BufferOwnership::consumer_acquired;
    case BufferOwnership::consumer_acquired:
        return to == BufferOwnership::consumer_owned;
    case BufferOwnership::consumer_owned:
        return to == BufferOwnership::reusable;
    case BufferOwnership::reusable:
        return to == BufferOwnership::producer_owned;
    }
    return false;
}

bool validate_descriptor(const TransferDescriptor& descriptor,
                         const DescriptorLimits& limits) noexcept {
    if (descriptor.protocol_version != limits.protocol_version ||
        descriptor.connection_epoch == 0 ||
        descriptor.request_id == 0 ||
        descriptor.byte_length == 0 ||
        limits.maximum_bytes == 0 ||
        descriptor.byte_length > limits.maximum_bytes) {
        return false;
    }
    return true;
}

double estimate_single_link_seconds(const LinkProfile& link,
                                    std::uint64_t bytes) noexcept {
    if (!available(link)) {
        return std::numeric_limits<double>::infinity();
    }
    const auto queued =
        static_cast<long double>(link.in_flight_bytes) /
        static_cast<long double>(link.payload_bytes_per_second);
    const auto payload =
        static_cast<long double>(bytes) /
        static_cast<long double>(link.payload_bytes_per_second);
    const auto estimate =
        static_cast<long double>(link.one_way_latency_seconds) + queued + payload;
    if (!std::isfinite(static_cast<double>(estimate))) {
        return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(estimate);
}

TransferPlan plan_transfer(std::span<const LinkProfile> links,
                           std::uint64_t bytes,
                           TransferPriority priority,
                           const SchedulerPolicy& policy) {
    if (bytes == 0) {
        throw std::invalid_argument("transport plan requires a non-zero payload");
    }

    const auto single = single_plan(links, bytes);
    if (single.kind == PlanKind::unavailable ||
        priority == TransferPriority::latency_critical ||
        bytes < policy.minimum_stripe_bytes) {
        return single;
    }

    TransferPlan best_striped;
    best_striped.estimated_seconds = std::numeric_limits<double>::infinity();

    for (std::size_t left = 0; left < links.size(); ++left) {
        if (!available(links[left]) || !links[left].independent_for_striping) {
            continue;
        }
        for (std::size_t right = left + 1; right < links.size(); ++right) {
            if (!available(links[right]) || !links[right].independent_for_striping ||
                links[left].peer_node != links[right].peer_node) {
                continue;
            }
            const auto candidate = striped_pair(
                links, bytes, left, right, policy.stripe_reassembly_seconds);
            if (candidate.kind == PlanKind::striped &&
                candidate.estimated_seconds < best_striped.estimated_seconds) {
                best_striped = candidate;
            }
        }
    }

    if (best_striped.kind == PlanKind::striped &&
        best_striped.estimated_seconds < single.estimated_seconds) {
        return best_striped;
    }
    return single;
}

}  // namespace strix::transport
