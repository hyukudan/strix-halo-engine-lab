#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strix::transport {

enum class BufferOwnership {
    producer_owned,
    producer_released,
    transport_owned,
    receive_complete,
    consumer_acquired,
    consumer_owned,
    reusable,
};

[[nodiscard]] bool can_transition(BufferOwnership from, BufferOwnership to) noexcept;

enum class LinkHealth {
    absent,
    present,
    usable,
    qualified,
    failed,
};

enum class TransferPriority {
    latency_critical,
    normal,
    bulk,
};

enum class PlanKind {
    unavailable,
    single_link,
    striped,
};

struct TransferDescriptor {
    std::uint32_t protocol_version = 1;
    std::uint64_t connection_epoch = 0;
    std::uint64_t request_id = 0;
    std::uint64_t generation = 0;
    std::uint64_t sequence = 0;
    std::uint64_t byte_length = 0;
};

struct DescriptorLimits {
    std::uint32_t protocol_version = 1;
    std::uint64_t maximum_bytes = 0;
};

[[nodiscard]] bool validate_descriptor(const TransferDescriptor& descriptor,
                                       const DescriptorLimits& limits) noexcept;

struct LinkProfile {
    std::string id;
    std::uint32_t peer_node = 0;
    LinkHealth health = LinkHealth::absent;
    double payload_bytes_per_second = 0.0;
    double one_way_latency_seconds = 0.0;
    std::uint64_t in_flight_bytes = 0;
    bool independent_for_striping = false;
};

struct SchedulerPolicy {
    std::uint64_t minimum_stripe_bytes = 1ULL << 20U;
    double stripe_reassembly_seconds = 0.0;
};

struct LinkSlice {
    std::size_t link_index = 0;
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_length = 0;
};

struct TransferPlan {
    PlanKind kind = PlanKind::unavailable;
    std::vector<LinkSlice> slices;
    double estimated_seconds = 0.0;
};

[[nodiscard]] double estimate_single_link_seconds(const LinkProfile& link,
                                                  std::uint64_t bytes) noexcept;

[[nodiscard]] TransferPlan plan_transfer(std::span<const LinkProfile> links,
                                         std::uint64_t bytes,
                                         TransferPriority priority,
                                         const SchedulerPolicy& policy = {});

}  // namespace strix::transport
