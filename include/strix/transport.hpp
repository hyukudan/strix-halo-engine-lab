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
    consumer_released,
    repost_pending,
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
    std::uint64_t model_plan_id = 0;
    std::uint64_t request_id = 0;
    std::uint64_t generation = 0;
    std::uint64_t boundary_id = 0;
    std::uint64_t sequence = 0;
    std::uint64_t shape_fingerprint = 0;
    std::uint32_t dtype_id = 0;
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
    remote_consumer_complete,
    receive_reposted,
    failed,
};

struct Completion {
    TransferHandle handle = 0;
    CompletionKind kind = CompletionKind::failed;
    std::uint64_t bytes = 0;
};

enum class TransportProgressMode {
    blocking,
    interrupt_driven,
    busy_poll,
    dedicated_poll,
};

enum class TransportBufferMode {
    copy_stream,
    mapped_host,
    gpu_visible_staging,
    imported_gpu_dma_buf,
};

struct BackendCapabilities {
    std::string name;
    std::uint64_t maximum_payload_bytes = 0;
    bool full_duplex = false;
    bool busy_poll = false;
    bool mapped_host_buffers = false;
    bool gpu_visible_buffers = false;
    bool external_dma = false;
    bool two_sided = true;
    bool requires_exclusive_lease = false;
    bool explicit_receive_repost = false;
    bool gpu_commit_record = false;
    TransportProgressMode progress_mode =
        TransportProgressMode::blocking;
    TransportBufferMode buffer_mode =
        TransportBufferMode::copy_stream;
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

inline constexpr std::uint32_t gpu_commit_word =
    0x53545843U;

struct GpuCommitRecord {
    std::uint32_t protocol_version = 1;
    std::uint32_t commit_word = 0;
    std::uint64_t connection_epoch = 0;
    std::uint64_t sequence = 0;
    std::uint32_t payload_bytes = 0;
    std::uint16_t stripe_index = 0;
    std::uint16_t stripe_count = 1;
};

static_assert(sizeof(GpuCommitRecord) == 32);

[[nodiscard]] bool validate_gpu_commit_record(
    const GpuCommitRecord& record,
    const TransferDescriptor& descriptor) noexcept;

class StripedMessageTracker {
public:
    explicit StripedMessageTracker(
        TransferDescriptor descriptor,
        std::uint16_t stripe_count);

    [[nodiscard]] bool observe_commit(
        const GpuCommitRecord& record);

    [[nodiscard]] bool complete() const noexcept {
        return complete_;
    }

    [[nodiscard]] bool invalid() const noexcept {
        return invalid_;
    }

    [[nodiscard]] std::uint64_t committed_bytes() const noexcept {
        return committed_bytes_;
    }

    [[nodiscard]] std::uint16_t observed_stripes() const noexcept {
        return observed_stripes_;
    }

private:
    TransferDescriptor descriptor_;
    std::uint16_t stripe_count_ = 0;
    std::uint16_t observed_stripes_ = 0;
    std::uint64_t committed_bytes_ = 0;
    bool complete_ = false;
    bool invalid_ = false;
    std::vector<std::uint32_t> stripe_bytes_;
};

enum class DirectLinkLifecycleState {
    portable_only,
    arm_allowed,
    preparing,
    direct_ready,
    leased,
    draining,
    partial,
    failed,
};

struct DirectLinkLifecycle {
    std::string physical_link_id;
    DirectLinkLifecycleState state =
        DirectLinkLifecycleState::portable_only;
    std::uint64_t connection_epoch = 0;
    std::uint32_t ring_frames = 0;
    std::uint64_t interrupt_throttling_ns = 0;
    std::int32_t in_hopid = -1;
    std::int32_t out_hopid = -1;
    bool control_link_ready = false;
    bool peer_reconciled = false;
    bool receive_ring_primed = false;
    bool exclusive_lease_available = false;
    bool dma_buf_import_supported = false;
    bool completion_rearm_qualified = false;
};

[[nodiscard]] bool direct_link_armable(
    const DirectLinkLifecycle& link) noexcept;

[[nodiscard]] bool direct_link_usable(
    const DirectLinkLifecycle& link) noexcept;

enum class DirectPairState {
    portable_only,
    arm_allowed,
    direct_ready,
    in_use,
    partial,
    failed,
};

struct DirectPairDecision {
    DirectPairState state =
        DirectPairState::portable_only;
    bool control_ready = false;
    bool parameters_match = false;
    bool hopids_reciprocal = false;
    bool epochs_match = false;
    bool receive_rings_primed = false;
    bool completion_rearm_qualified = false;
    bool lease_available = false;
    bool dma_buf_import_common = false;
};

[[nodiscard]] DirectPairDecision
reconcile_direct_link_pair(
    const DirectLinkLifecycle& local,
    const DirectLinkLifecycle& peer) noexcept;

enum class LinkCompletionTarget {
    receive_ready,
    remote_consumer_complete,
    receiver_reusable,
};

struct LinkProfile {
    std::string id;
    std::uint32_t peer_node = 0;
    LinkHealth health = LinkHealth::absent;
    double payload_bytes_per_second = 0.0;
    double one_way_latency_seconds = 0.0;
    double consumer_completion_seconds = 0.0;
    double receive_repost_seconds = 0.0;
    std::uint64_t in_flight_bytes = 0;
    bool independent_for_striping = false;
};

struct SchedulerPolicy {
    std::uint64_t minimum_stripe_bytes = 1ULL << 20U;
    double stripe_reassembly_seconds = 0.0;
    LinkCompletionTarget completion_target =
        LinkCompletionTarget::receive_ready;
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

[[nodiscard]] double estimate_link_completion_seconds(
    const LinkProfile& link,
    std::uint64_t bytes,
    LinkCompletionTarget target) noexcept;

[[nodiscard]] double estimate_single_link_seconds(
    const LinkProfile& link,
    std::uint64_t bytes) noexcept;

[[nodiscard]] TransferPlan plan_transfer(std::span<const LinkProfile> links,
                                         std::uint64_t bytes,
                                         TransferPriority priority,
                                         const SchedulerPolicy& policy = {});

}  // namespace strix::transport
