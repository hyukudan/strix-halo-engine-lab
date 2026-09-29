#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
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
    std::uint64_t connection_epoch = 0;
    std::uint64_t maximum_bytes = 0;
};

[[nodiscard]] bool validate_descriptor(const TransferDescriptor& descriptor,
                                       const DescriptorLimits& limits) noexcept;


using TransferHandle = std::uint64_t;

enum class CompletionKind {
    local_buffer_reusable,
    receive_complete,
    remote_transport_ack,
    failed,
};

struct Completion {
    TransferHandle handle = 0;
    CompletionKind kind = CompletionKind::failed;
    std::uint64_t bytes = 0;
};

struct BackendCapabilities {
    std::string name;
    std::uint64_t maximum_payload_bytes = 0;
    bool full_duplex = false;
    bool busy_poll = false;
    bool mapped_host_buffers = false;
    bool gpu_visible_buffers = false;
    bool external_dma = false;
};

class TransportBackend {
public:
    virtual ~TransportBackend() = default;

    [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;

    [[nodiscard]] virtual TransferHandle prepare_send(
        const TransferDescriptor& descriptor,
        std::span<const std::byte> payload) = 0;

    [[nodiscard]] virtual TransferHandle prepare_receive(
        const TransferDescriptor& descriptor,
        std::span<std::byte> destination) = 0;

    virtual void submit(TransferHandle handle) = 0;

    [[nodiscard]] virtual std::optional<Completion> poll() = 0;

    virtual void cancel(TransferHandle handle) = 0;
};

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
