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

TransferPlan single_plan(
    std::span<const LinkProfile> links,
    std::uint64_t bytes,
    LinkCompletionTarget target) {
    TransferPlan plan;
    double best = std::numeric_limits<double>::infinity();
    std::size_t best_index = 0;
    bool found = false;

    for (std::size_t index = 0; index < links.size(); ++index) {
        if (!available(links[index])) {
            continue;
        }
        const auto estimate =
            estimate_link_completion_seconds(
                links[index],
                bytes,
                target);
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

TransferPlan striped_pair(
    std::span<const LinkProfile> links,
    std::uint64_t bytes,
    std::size_t left,
    std::size_t right,
    double reassembly_seconds,
    LinkCompletionTarget target) {
    const auto& a = links[left];
    const auto& b = links[right];
    const auto bandwidth_sum = a.payload_bytes_per_second + b.payload_bytes_per_second;
    if (!(bandwidth_sum > 0.0) || !std::isfinite(bandwidth_sum)) {
        return {};
    }

    // Minimize max(completion_a, completion_b), including each link's
    // measured latency and already queued bytes. Solve the equal-completion
    // point, then clamp to a real two-slice transfer.
    const long double bw_a = static_cast<long double>(a.payload_bytes_per_second);
    const long double bw_b = static_cast<long double>(b.payload_bytes_per_second);
    const long double fixed_a =
        static_cast<long double>(a.one_way_latency_seconds) +
        static_cast<long double>(a.in_flight_bytes) / bw_a;
    const long double fixed_b =
        static_cast<long double>(b.one_way_latency_seconds) +
        static_cast<long double>(b.in_flight_bytes) / bw_b;
    const long double denominator = (1.0L / bw_a) + (1.0L / bw_b);
    const long double equal_point =
        (fixed_b - fixed_a + static_cast<long double>(bytes) / bw_b) /
        denominator;

    const auto maximum_first = bytes - 1;
    auto first_bytes = equal_point <= 1.0L
        ? std::uint64_t{1}
        : equal_point >= static_cast<long double>(maximum_first)
            ? maximum_first
            : static_cast<std::uint64_t>(equal_point);
    if (first_bytes == 0 || first_bytes >= bytes) {
        return {};
    }
    const auto second_bytes = bytes - first_bytes;

    const auto first_seconds =
        estimate_link_completion_seconds(
            a,
            first_bytes,
            target);
    const auto second_seconds =
        estimate_link_completion_seconds(
            b,
            second_bytes,
            target);
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


bool validate_transport_send_buffer(
    const TransportBufferView& buffer,
    std::uint64_t required_bytes) noexcept {
    if (buffer.address == 0 ||
        buffer.bytes == 0 ||
        required_bytes == 0 ||
        buffer.bytes < required_bytes ||
        buffer.allocation_generation == 0) {
        return false;
    }

    return buffer.access ==
               TransportBufferAccess::read_only ||
           buffer.access ==
               TransportBufferAccess::read_write;
}

bool validate_transport_receive_buffer(
    const TransportBufferView& buffer,
    std::uint64_t required_bytes) noexcept {
    if (buffer.address == 0 ||
        buffer.bytes == 0 ||
        required_bytes == 0 ||
        buffer.bytes < required_bytes ||
        buffer.allocation_generation == 0) {
        return false;
    }

    return buffer.access ==
               TransportBufferAccess::write_only ||
           buffer.access ==
               TransportBufferAccess::read_write;
}

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
        return to == BufferOwnership::consumer_released;
    case BufferOwnership::consumer_released:
        return to == BufferOwnership::repost_pending;
    case BufferOwnership::repost_pending:
        return to == BufferOwnership::reusable;
    case BufferOwnership::reusable:
        return to == BufferOwnership::producer_owned;
    }
    return false;
}

bool validate_descriptor(const TransferDescriptor& descriptor,
                         const DescriptorLimits& limits) noexcept {
    if (descriptor.protocol_version != limits.protocol_version ||
        limits.connection_epoch == 0 ||
        descriptor.connection_epoch != limits.connection_epoch ||
        descriptor.model_plan_id == 0 ||
        descriptor.request_id == 0 ||
        descriptor.boundary_id == 0 ||
        descriptor.shape_fingerprint == 0 ||
        descriptor.dtype_id == 0 ||
        descriptor.byte_length == 0 ||
        limits.maximum_bytes == 0 ||
        descriptor.byte_length > limits.maximum_bytes) {
        return false;
    }
    return true;
}


bool validate_gpu_commit_record(
    const GpuCommitRecord& record,
    const TransferDescriptor& descriptor) noexcept {
    if (record.protocol_version !=
            descriptor.protocol_version ||
        record.commit_word !=
            gpu_commit_word ||
        record.connection_epoch !=
            descriptor.connection_epoch ||
        record.sequence !=
            descriptor.sequence ||
        record.payload_bytes == 0 ||
        static_cast<std::uint64_t>(
            record.payload_bytes) >
            descriptor.byte_length ||
        record.stripe_count == 0 ||
        record.stripe_index >=
            record.stripe_count) {
        return false;
    }
    return true;
}


StripedMessageTracker::StripedMessageTracker(
    TransferDescriptor descriptor,
    std::uint16_t stripe_count)
    : descriptor_(descriptor),
      stripe_count_(stripe_count),
      stripe_bytes_(
          stripe_count,
          0) {
    if (stripe_count_ == 0 ||
        descriptor_.byte_length == 0) {
        throw std::invalid_argument(
            "striped message tracker requires non-zero payload and stripe count");
    }
}

bool StripedMessageTracker::observe_commit(
    const GpuCommitRecord& record) {
    if (invalid_ || complete_) {
        return false;
    }
    if (!validate_gpu_commit_record(
            record,
            descriptor_) ||
        record.stripe_count !=
            stripe_count_) {
        invalid_ = true;
        return false;
    }

    const auto stripe =
        static_cast<std::size_t>(
            record.stripe_index);
    if (stripe >=
            stripe_bytes_.size() ||
        stripe_bytes_[stripe] != 0) {
        invalid_ = true;
        return false;
    }

    const auto bytes =
        static_cast<std::uint64_t>(
            record.payload_bytes);
    if (bytes >
        descriptor_.byte_length -
            committed_bytes_) {
        invalid_ = true;
        return false;
    }

    stripe_bytes_[stripe] =
        record.payload_bytes;
    committed_bytes_ += bytes;
    ++observed_stripes_;

    if (observed_stripes_ ==
        stripe_count_) {
        if (committed_bytes_ !=
            descriptor_.byte_length) {
            invalid_ = true;
            return false;
        }
        complete_ = true;
    }
    return true;
}

bool direct_link_armable(
    const DirectLinkLifecycle& link) noexcept {
    return link.state ==
               DirectLinkLifecycleState::arm_allowed &&
           link.connection_epoch != 0 &&
           link.control_link_ready &&
           link.peer_reconciled &&
           link.exclusive_lease_available;
}

bool direct_link_usable(
    const DirectLinkLifecycle& link) noexcept {
    return (link.state ==
                DirectLinkLifecycleState::direct_ready ||
            link.state ==
                DirectLinkLifecycleState::leased) &&
           link.connection_epoch != 0 &&
           link.control_link_ready &&
           link.peer_reconciled &&
           link.receive_ring_primed &&
           link.completion_rearm_qualified &&
           (!link.dma_buf_import_supported ||
            link.exclusive_lease_available ||
            link.state ==
                DirectLinkLifecycleState::leased);
}


DirectPairDecision
reconcile_direct_link_pair(
    const DirectLinkLifecycle& local,
    const DirectLinkLifecycle& peer) noexcept {
    DirectPairDecision decision;

    decision.control_ready =
        local.control_link_ready &&
        peer.control_link_ready;

    decision.parameters_match =
        local.ring_frames != 0 &&
        local.ring_frames ==
            peer.ring_frames &&
        local.interrupt_throttling_ns ==
            peer.interrupt_throttling_ns;

    decision.hopids_reciprocal =
        local.in_hopid >= 0 &&
        local.out_hopid >= 0 &&
        peer.in_hopid >= 0 &&
        peer.out_hopid >= 0 &&
        local.in_hopid ==
            peer.out_hopid &&
        local.out_hopid ==
            peer.in_hopid;

    decision.epochs_match =
        local.connection_epoch != 0 &&
        local.connection_epoch ==
            peer.connection_epoch;

    decision.receive_rings_primed =
        local.receive_ring_primed &&
        peer.receive_ring_primed;

    decision.completion_rearm_qualified =
        local.completion_rearm_qualified &&
        peer.completion_rearm_qualified;

    decision.lease_available =
        local.exclusive_lease_available &&
        peer.exclusive_lease_available;

    decision.dma_buf_import_common =
        local.dma_buf_import_supported &&
        peer.dma_buf_import_supported;

    if (local.state ==
            DirectLinkLifecycleState::failed ||
        peer.state ==
            DirectLinkLifecycleState::failed) {
        decision.state =
            DirectPairState::failed;
        return decision;
    }

    if (!decision.control_ready) {
        decision.state =
            DirectPairState::portable_only;
        return decision;
    }

    if (!decision.parameters_match ||
        !decision.hopids_reciprocal) {
        decision.state =
            DirectPairState::partial;
        return decision;
    }

    const bool both_direct_state =
        (local.state ==
             DirectLinkLifecycleState::direct_ready ||
         local.state ==
             DirectLinkLifecycleState::leased) &&
        (peer.state ==
             DirectLinkLifecycleState::direct_ready ||
         peer.state ==
             DirectLinkLifecycleState::leased);

    if (both_direct_state) {
        if (!decision.epochs_match ||
            !decision.receive_rings_primed ||
            !decision.completion_rearm_qualified) {
            decision.state =
                DirectPairState::partial;
            return decision;
        }

        if (!decision.lease_available ||
            local.state ==
                DirectLinkLifecycleState::leased ||
            peer.state ==
                DirectLinkLifecycleState::leased) {
            decision.state =
                DirectPairState::in_use;
            return decision;
        }

        decision.state =
            DirectPairState::direct_ready;
        return decision;
    }

    if (decision.lease_available) {
        decision.state =
            DirectPairState::arm_allowed;
    } else {
        decision.state =
            DirectPairState::in_use;
    }
    return decision;
}

double estimate_link_completion_seconds(
    const LinkProfile& link,
    std::uint64_t bytes,
    LinkCompletionTarget target) noexcept {
    if (!available(link) ||
        !std::isfinite(
            link.consumer_completion_seconds) ||
        link.consumer_completion_seconds < 0.0 ||
        !std::isfinite(
            link.receive_repost_seconds) ||
        link.receive_repost_seconds < 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const auto queued =
        static_cast<long double>(
            link.in_flight_bytes) /
        static_cast<long double>(
            link.payload_bytes_per_second);
    const auto payload =
        static_cast<long double>(bytes) /
        static_cast<long double>(
            link.payload_bytes_per_second);

    long double estimate =
        static_cast<long double>(
            link.one_way_latency_seconds) +
        queued +
        payload;

    if (target ==
            LinkCompletionTarget::remote_consumer_complete ||
        target ==
            LinkCompletionTarget::receiver_reusable) {
        estimate +=
            static_cast<long double>(
                link.consumer_completion_seconds);
    }
    if (target ==
        LinkCompletionTarget::receiver_reusable) {
        estimate +=
            static_cast<long double>(
                link.receive_repost_seconds);
    }

    if (!std::isfinite(
            static_cast<double>(
                estimate))) {
        return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(estimate);
}

double estimate_single_link_seconds(
    const LinkProfile& link,
    std::uint64_t bytes) noexcept {
    return estimate_link_completion_seconds(
        link,
        bytes,
        LinkCompletionTarget::receive_ready);
}

TransferPlan plan_transfer(std::span<const LinkProfile> links,
                           std::uint64_t bytes,
                           TransferPriority priority,
                           const SchedulerPolicy& policy) {
    if (bytes == 0) {
        throw std::invalid_argument("transport plan requires a non-zero payload");
    }

    const auto single =
        single_plan(
            links,
            bytes,
            policy.completion_target);
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
            const auto candidate =
                striped_pair(
                    links,
                    bytes,
                    left,
                    right,
                    policy.stripe_reassembly_seconds,
                    policy.completion_target);
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
