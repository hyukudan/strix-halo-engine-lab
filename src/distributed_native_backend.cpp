#include "strix/distributed_native_backend.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
namespace {

struct NativePreparedTransfer {
    transport::DirectTransportNativeLinkSession* session = nullptr;
    transport::TransferHandle handle = 0;
    transport::TransferDescriptor descriptor;
    transport::TransportBufferView staged;
    transport::LinkSlice slice;
    std::uint16_t stripe_index = 0;
};

void require_default_single_slice(
    const DistributedBoundaryTransfer& transfer,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count) {
    if (stripe_count != 1 ||
        stripe_index != 0 ||
        transfer.link_plan.kind !=
            transport::PlanKind::single_link ||
        slice.byte_offset != 0 ||
        slice.byte_length !=
            transfer.descriptor.byte_length) {
        throw std::logic_error(
            "staging provider does not implement striped boundary access");
    }
}

std::uint16_t validate_slices(
    const DistributedBoundaryTransfer& transfer,
    std::size_t session_count) {
    const auto count =
        transfer.link_plan.slices.size();
    if (count == 0 ||
        count >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint16_t>::max()) ||
        transfer.descriptor.byte_length == 0) {
        throw std::invalid_argument(
            "native distributed transfer has invalid stripe count/payload");
    }

    if (transfer.link_plan.kind ==
            transport::PlanKind::single_link &&
        count != 1) {
        throw std::invalid_argument(
            "single-link transfer plan must contain exactly one slice");
    }
    if (transfer.link_plan.kind ==
            transport::PlanKind::striped &&
        count < 2) {
        throw std::invalid_argument(
            "striped transfer plan requires at least two slices");
    }
    if (transfer.link_plan.kind != transport::PlanKind::single_link &&
        transfer.link_plan.kind != transport::PlanKind::striped) {
        throw std::invalid_argument(
            "unavailable transfer plan cannot reach native backend");
    }

    std::vector<bool> used_links(
        session_count,
        false);
    std::uint64_t expected_offset = 0;
    for (const auto& slice :
         transfer.link_plan.slices) {
        if (slice.link_index >=
                session_count ||
            slice.byte_length == 0 ||
            slice.byte_length > std::numeric_limits<std::uint32_t>::max() ||
            slice.byte_offset !=
                expected_offset ||
            slice.byte_length >
                transfer.descriptor.byte_length -
                    expected_offset) {
            throw std::invalid_argument(
                "native distributed transfer slices are not an exact contiguous cover");
        }
        if (transfer.link_plan.kind ==
                transport::PlanKind::striped &&
            used_links[slice.link_index]) {
            throw std::invalid_argument(
                "striped native transfer requires distinct physical links");
        }
        used_links[slice.link_index] = true;
        expected_offset +=
            slice.byte_length;
    }

    if (expected_offset !=
        transfer.descriptor.byte_length) {
        throw std::invalid_argument(
            "native distributed transfer slices do not cover logical payload");
    }

    return static_cast<std::uint16_t>(
        count);
}

transport::TransferDescriptor slice_descriptor(
    const transport::TransferDescriptor& logical,
    const transport::LinkSlice& slice,
    const DistributedNativeSliceIdentity& identity) {
    auto descriptor =
        logical;
    descriptor.byte_length =
        slice.byte_length;
    descriptor.connection_epoch = identity.connection_epoch;
    descriptor.sequence = identity.sequence;
    return descriptor;
}

transport::GpuCommitRecord make_commit(
    const transport::TransferDescriptor& physical,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count) {
    if (slice.byte_length >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "native direct stripe payload exceeds 32-bit commit record");
    }

    transport::GpuCommitRecord record;
    record.protocol_version =
        physical.protocol_version;
    record.commit_word =
        transport::gpu_commit_word;
    record.connection_epoch =
        physical.connection_epoch;
    record.sequence =
        physical.sequence;
    record.payload_bytes =
        static_cast<std::uint32_t>(
            slice.byte_length);
    record.stripe_index =
        stripe_index;
    record.stripe_count =
        stripe_count;
    return record;
}

}  // namespace

transport::TransportBufferView
DistributedBoundaryStagingProvider::stage_outbound_slice(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count) {
    require_default_single_slice(
        transfer,
        slice,
        stripe_index,
        stripe_count);
    return stage_outbound(
        boundary,
        transfer);
}

transport::TransportBufferView
DistributedBoundaryStagingProvider::stage_inbound_slice(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count) {
    require_default_single_slice(
        transfer,
        slice,
        stripe_index,
        stripe_count);
    return stage_inbound(
        boundary,
        transfer);
}

void DistributedBoundaryStagingProvider::consume_inbound_slice(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count,
    const transport::TransportBufferView& staged) {
    require_default_single_slice(
        transfer,
        slice,
        stripe_index,
        stripe_count);
    consume_inbound(
        boundary,
        staged);
}

NativeDistributedBackend::NativeDistributedBackend(
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
        policy_.maximum_event_polls == 0 ||
        policy_.maximum_consumer_polls == 0) {
        throw std::invalid_argument(
            "native distributed backend requires sessions and non-zero poll budget");
    }
}

void NativeDistributedBackend::require_healthy() const {
    if (failed_) {
        throw std::logic_error("native distributed backend is poisoned");
    }
}

transport::DirectTransportNativeLinkSession&
NativeDistributedBackend::session_for_link(
    std::size_t link_index) const {
    if (link_index >=
            sessions_.size() ||
        sessions_[link_index] == nullptr ||
        !sessions_[link_index]->active() ||
        sessions_[link_index]->lease().link_index != link_index) {
        throw std::invalid_argument(
            "distributed transfer references unavailable native link session");
    }
    return *sessions_[link_index];
}

void NativeDistributedBackend::execute_local_step(
    const DistributedStepPlacement& step) {
    require_healthy();
    local_->execute_local_step(step);
}

void NativeDistributedBackend::execute_remote_step(
    const DistributedStepPlacement& step) {
    require_healthy();
    control_->execute_remote_step(step);
}

void NativeDistributedBackend::wait_send_reusable(
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
    const auto& final = session.send_state(handle);
    if (!final.failed && final.local_buffer_reusable) {
        return;
    }
    throw std::runtime_error(
        "native distributed send completion poll budget exhausted");
}

void NativeDistributedBackend::wait_receive_complete(
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
    const auto& final = session.receive_state(handle);
    if (!final.failed && final.ownership == transport::BufferOwnership::receive_complete) {
        return;
    }
    throw std::runtime_error(
        "native distributed receive completion poll budget exhausted");
}

void NativeDistributedBackend::wait_receive_reusable(
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
    const auto& final = session.receive_state(handle);
    if (!final.failed && final.ownership == transport::BufferOwnership::reusable) {
        return;
    }
    throw std::runtime_error(
        "native distributed receive repost poll budget exhausted");
}

void NativeDistributedBackend::send_boundary(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer) {
    require_healthy();
    const auto stripe_count =
        validate_slices(
            transfer,
            sessions_.size());
    if (!transfer.native_slices.empty()) {
        throw std::invalid_argument("native send requires an unbound logical transfer");
    }
    if (transfer.descriptor.boundary_id != boundary.boundary_id ||
        transfer.descriptor.byte_length != boundary.payload_bytes ||
        transfer.flow.direction != transport::LinkDirection::outbound ||
        transfer.descriptor.sequence == 0 || transfer.descriptor.generation == 0 ||
        !transport::validate_descriptor(transfer.descriptor,
            {1, transfer.descriptor.connection_epoch, boundary.payload_bytes})) {
        throw std::invalid_argument("native send metadata does not match boundary");
    }

    auto physical_transfer = transfer;
    physical_transfer.native_slices.reserve(stripe_count);

    std::vector<NativePreparedTransfer>
        prepared;
    prepared.reserve(
        transfer.link_plan.slices.size());

    try {
        for (const auto& slice : transfer.link_plan.slices) {
            auto& session = session_for_link(slice.link_index);
            physical_transfer.native_slices.push_back({
                session.lease().connection_epoch, session.reserve_send_sequence(),
            });
        }
        for (std::size_t index = 0;
             index <
                transfer.link_plan.slices.size();
             ++index) {
            const auto& slice =
                transfer.link_plan.slices[index];
            auto& session =
                session_for_link(
                    slice.link_index);
            const auto stripe_index =
                static_cast<std::uint16_t>(
                    index);
            const auto& identity = physical_transfer.native_slices[index];
            const auto staged =
                staging_->stage_outbound_slice(
                    boundary,
                    physical_transfer,
                    slice,
                    stripe_index,
                    stripe_count);
            const auto descriptor =
                slice_descriptor(
                    transfer.descriptor,
                    slice,
                    identity);
            const auto handle =
                session.prepare_send(
                    descriptor,
                    staged);
            prepared.push_back({
                &session,
                handle,
                descriptor,
                staged,
                slice,
                stripe_index,
            });
        }

        control_->publish_boundary_transfer(
            boundary,
            physical_transfer);

        for (auto& item :
             prepared) {
            const auto commit =
                make_commit(
                    item.descriptor,
                    item.slice,
                    item.stripe_index,
                    stripe_count);
            item.session->publish_commit(
                item.handle,
                item.descriptor,
                commit);
            item.session->submit(
                item.handle);
        }

        for (auto& item :
             prepared) {
            wait_send_reusable(
                *item.session,
                item.handle);
        }
    } catch (...) {
        failed_ = true;
        for (auto& item :
             prepared) {
            try {
                if (!item.session->
                        send_state(
                            item.handle).
                        local_buffer_reusable) {
                    item.session->cancel(
                        item.handle);
                }
            } catch (...) {
            }
        }
        throw;
    }
}

DistributedBoundaryTransfer
NativeDistributedBackend::receive_boundary(
    const DistributedExecutionBoundary& boundary) {
    require_healthy();
    auto transfer =
        control_->await_boundary_transfer(
            boundary);
    const auto stripe_count =
        validate_slices(
            transfer,
            sessions_.size());
    if (transfer.native_slices.size() != stripe_count ||
        transfer.descriptor.boundary_id != boundary.boundary_id ||
        transfer.descriptor.byte_length != boundary.payload_bytes ||
        transfer.flow.direction != transport::LinkDirection::inbound ||
        transfer.descriptor.sequence == 0 || transfer.descriptor.generation == 0 ||
        !transport::validate_descriptor(transfer.descriptor,
            {1, transfer.descriptor.connection_epoch, boundary.payload_bytes})) {
        throw std::invalid_argument("native receive lacks exact physical/logical boundary identities");
    }

    // Validate every current epoch before reserving sequences or staging any
    // payload. Logical request/plan reconciliation belongs to the control
    // plane; physical descriptors retain those identities unchanged.
    for (std::size_t index = 0; index < stripe_count; ++index) {
        auto& session = session_for_link(transfer.link_plan.slices[index].link_index);
        const auto& identity = transfer.native_slices[index];
        if (identity.connection_epoch != session.lease().connection_epoch ||
            identity.sequence == 0) {
            throw std::invalid_argument("native receive slice epoch does not match leased link");
        }
    }

    std::vector<NativePreparedTransfer>
        prepared;
    prepared.reserve(
        transfer.link_plan.slices.size());

    try {
        for (std::size_t index = 0;
             index < transfer.link_plan.slices.size();
             ++index) {
            const auto& slice = transfer.link_plan.slices[index];
            auto& session = session_for_link(slice.link_index);
            const auto stripe_index = static_cast<std::uint16_t>(index);
            const auto& identity = transfer.native_slices[index];
            session.reserve_receive_sequence(identity.sequence);
            const auto staged = staging_->stage_inbound_slice(
                boundary, transfer, slice, stripe_index, stripe_count);
            const auto descriptor = slice_descriptor(
                transfer.descriptor, slice, identity);
            const auto handle = session.prepare_receive(descriptor, staged);
            prepared.push_back({
                &session, handle, descriptor, staged, slice, stripe_index,
            });
        }

        for (auto& item :
             prepared) {
            item.session->submit(
                item.handle);
        }

        std::vector<transport::TransferDescriptor> physical_descriptors;
        physical_descriptors.reserve(prepared.size());
        for (const auto& item : prepared) {
            physical_descriptors.push_back(item.descriptor);
        }
        transport::StripedMessageTracker
            tracker(
                transfer.descriptor,
                physical_descriptors);

        for (auto& item :
             prepared) {
            wait_receive_complete(
                *item.session,
                item.handle);
            const auto commit =
                item.session->read_commit(
                    item.handle,
                    item.descriptor);
            if (commit.stripe_index != item.stripe_index) {
                throw std::runtime_error("native receive commit belongs to a different physical stripe");
            }
            if (!tracker.observe_commit(
                    commit)) {
                throw std::runtime_error(
                    "native striped receive commit validation failed");
            }
        }

        if (!tracker.complete() ||
            tracker.invalid() ||
            tracker.committed_bytes() !=
                transfer.descriptor.byte_length) {
            throw std::runtime_error(
                "native striped receive did not publish complete logical payload");
        }

        for (auto& item :
             prepared) {
            item.session->acquire_receive(
                item.handle);
            staging_->consume_inbound_slice(
                boundary,
                transfer,
                item.slice,
                item.stripe_index,
                stripe_count,
                item.staged);
        }
        inbound_completion_ = staging_->finish_inbound(
            boundary,
            transfer);
        if (!inbound_completion_) {
            throw std::runtime_error("native receive consumer supplied no completion proof");
        }
        bool consumer_complete = false;
        for (std::uint64_t poll = 0; poll < policy_.maximum_consumer_polls; ++poll) {
            const auto status = inbound_completion_->poll();
            if (status == DistributedInboundCompletionStatus::complete) {
                consumer_complete = true;
                break;
            }
            if (status == DistributedInboundCompletionStatus::failed) {
                throw std::runtime_error("native receive consumer completion failed");
            }
        }
        if (!consumer_complete) {
            throw std::runtime_error("native receive consumer completion poll budget exhausted");
        }

        for (auto& item :
             prepared) {
            item.session->release_receive(
                item.handle);
            item.session->
                publish_receive_credit(
                    item.handle);
            item.session->repost_receive(
                item.handle);
        }
        for (auto& item :
             prepared) {
            wait_receive_reusable(
                *item.session,
                item.handle);
        }
        inbound_completion_.reset();
    } catch (...) {
        failed_ = true;
        for (auto& item :
             prepared) {
            try {
                const auto& state =
                    item.session->
                        receive_state(
                            item.handle);
                if (state.ownership ==
                        transport::BufferOwnership::transport_owned) {
                    try { item.session->cancel(item.handle); } catch (...) {}
                }
                if (state.ownership != transport::BufferOwnership::reusable) {
                    item.session->quarantine_receive(item.handle);
                }
            } catch (...) {
            }
        }
        throw;
    }

    return transfer;
}

}  // namespace strix
