#include "strix/distributed_native_backend.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

const transport::LinkSlice&
single_slice(
    const DistributedBoundaryTransfer& transfer) {
    if (transfer.link_plan.kind !=
            transport::PlanKind::single_link ||
        transfer.link_plan.slices.size() != 1) {
        throw std::logic_error(
            "native distributed backend currently requires one-link transfer plan");
    }
    const auto& slice =
        transfer.link_plan.slices.front();
    if (slice.byte_offset != 0 ||
        slice.byte_length == 0 ||
        slice.byte_length !=
            transfer.descriptor.byte_length) {
        throw std::invalid_argument(
            "native distributed single-link slice does not cover full descriptor payload");
    }
    return slice;
}

transport::GpuCommitRecord
make_single_commit(
    const transport::TransferDescriptor& descriptor) {
    if (descriptor.byte_length >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "native direct commit payload exceeds 32-bit record");
    }

    transport::GpuCommitRecord record;
    record.protocol_version =
        descriptor.protocol_version;
    record.commit_word =
        transport::gpu_commit_word;
    record.connection_epoch =
        descriptor.connection_epoch;
    record.sequence =
        descriptor.sequence;
    record.payload_bytes =
        static_cast<std::uint32_t>(
            descriptor.byte_length);
    record.stripe_index = 0;
    record.stripe_count = 1;
    return record;
}

}  // namespace

SingleLinkNativeDistributedBackend::
SingleLinkNativeDistributedBackend(
    std::span<transport::DirectTransportNativeLinkSession*> sessions,
    DistributedNativeControlPlane& control,
    DistributedNativeLocalExecutor& local,
    DistributedBoundaryStagingProvider& staging,
    DistributedNativeBackendPolicy policy)
    : sessions_(sessions),
      control_(&control),
      local_(&local),
      staging_(&staging),
      policy_(policy) {
    if (sessions_.empty() ||
        policy_.maximum_event_polls == 0) {
        throw std::invalid_argument(
            "native distributed backend requires sessions and non-zero poll budget");
    }
}

transport::DirectTransportNativeLinkSession&
SingleLinkNativeDistributedBackend::session_for(
    const DistributedBoundaryTransfer& transfer) const {
    const auto& slice =
        single_slice(transfer);
    if (slice.link_index >=
            sessions_.size() ||
        sessions_[slice.link_index] == nullptr ||
        !sessions_[slice.link_index]->active()) {
        throw std::invalid_argument(
            "distributed transfer references unavailable native link session");
    }
    return *sessions_[slice.link_index];
}

void SingleLinkNativeDistributedBackend::execute_local_step(
    const DistributedStepPlacement& step) {
    local_->execute_local_step(step);
}

void SingleLinkNativeDistributedBackend::execute_remote_step(
    const DistributedStepPlacement& step) {
    control_->execute_remote_step(step);
}

void SingleLinkNativeDistributedBackend::wait_send_reusable(
    transport::DirectTransportNativeLinkSession& session,
    transport::TransferHandle handle) const {
    for (std::uint64_t poll = 0;
         poll < policy_.maximum_event_polls;
         ++poll) {
        const auto& state =
            session.send_state(handle);
        if (state.failed) {
            throw std::runtime_error(
                "native distributed send failed");
        }
        if (state.local_buffer_reusable) {
            return;
        }
        static_cast<void>(
            session.poll_event());
    }
    session.cancel(handle);
    throw std::runtime_error(
        "native distributed send completion poll budget exhausted");
}

void SingleLinkNativeDistributedBackend::wait_receive_complete(
    transport::DirectTransportNativeLinkSession& session,
    transport::TransferHandle handle) const {
    for (std::uint64_t poll = 0;
         poll < policy_.maximum_event_polls;
         ++poll) {
        const auto& state =
            session.receive_state(handle);
        if (state.failed) {
            throw std::runtime_error(
                "native distributed receive failed");
        }
        if (state.ownership ==
            transport::BufferOwnership::receive_complete) {
            return;
        }
        static_cast<void>(
            session.poll_event());
    }
    session.cancel(handle);
    throw std::runtime_error(
        "native distributed receive completion poll budget exhausted");
}

void SingleLinkNativeDistributedBackend::wait_receive_reusable(
    transport::DirectTransportNativeLinkSession& session,
    transport::TransferHandle handle) const {
    for (std::uint64_t poll = 0;
         poll < policy_.maximum_event_polls;
         ++poll) {
        const auto& state =
            session.receive_state(handle);
        if (state.failed) {
            throw std::runtime_error(
                "native distributed receive failed during repost");
        }
        if (state.ownership ==
            transport::BufferOwnership::reusable) {
            return;
        }
        static_cast<void>(
            session.poll_event());
    }
    throw std::runtime_error(
        "native distributed receive repost poll budget exhausted");
}

void SingleLinkNativeDistributedBackend::send_boundary(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer) {
    auto& session =
        session_for(transfer);
    const auto staged =
        staging_->stage_outbound(
            boundary,
            transfer);

    const auto handle =
        session.prepare_send(
            transfer.descriptor,
            staged);

    control_->publish_boundary_transfer(
        boundary,
        transfer);

    const auto commit =
        make_single_commit(
            transfer.descriptor);
    session.publish_commit(
        handle,
        transfer.descriptor,
        commit);
    session.submit(handle);
    wait_send_reusable(
        session,
        handle);
}

DistributedBoundaryTransfer
SingleLinkNativeDistributedBackend::receive_boundary(
    const DistributedExecutionBoundary& boundary) {
    auto transfer =
        control_->await_boundary_transfer(
            boundary);
    auto& session =
        session_for(transfer);
    const auto staged =
        staging_->stage_inbound(
            boundary,
            transfer);

    const auto handle =
        session.prepare_receive(
            transfer.descriptor,
            staged);
    session.submit(handle);
    wait_receive_complete(
        session,
        handle);

    session.acquire_receive(handle);
    staging_->consume_inbound(
        boundary,
        staged);
    session.release_receive(handle);
    session.publish_receive_credit(handle);
    session.repost_receive(handle);
    wait_receive_reusable(
        session,
        handle);
    return transfer;
}

}  // namespace strix
