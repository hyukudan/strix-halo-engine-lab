#include "strix/direct_transport_completion.hpp"

#include <stdexcept>

namespace strix::transport {
namespace {

void require_completion_bytes(
    const Completion& completion,
    std::uint64_t expected) {
    if (completion.bytes != expected) {
        throw std::invalid_argument(
            "direct transport completion byte count mismatch");
    }
}

void transition(
    BufferOwnership& ownership,
    BufferOwnership next) {
    if (!can_transition(
            ownership,
            next)) {
        throw std::logic_error(
            "direct receive completion violates buffer ownership lifecycle");
    }
    ownership = next;
}

}  // namespace

void DirectReceiveCompletionBridge::register_receive(
    TransferHandle handle,
    const TransferDescriptor& descriptor) {
    if (handle == 0 ||
        descriptor.byte_length == 0 ||
        descriptor.connection_epoch == 0) {
        throw std::invalid_argument(
            "direct receive registration requires valid handle/descriptor");
    }
    DirectReceiveCompletionState state;
    state.handle = handle;
    state.descriptor = descriptor;
    const auto inserted =
        states_.emplace(
            handle,
            state);
    if (!inserted.second) {
        throw std::logic_error(
            "direct receive handle is already registered");
    }
}

DirectReceiveCompletionState&
DirectReceiveCompletionBridge::mutable_state(
    TransferHandle handle) {
    const auto found =
        states_.find(handle);
    if (found == states_.end()) {
        throw std::invalid_argument(
            "direct receive completion references unknown handle");
    }
    return found->second;
}

const DirectReceiveCompletionState&
DirectReceiveCompletionBridge::state(
    TransferHandle handle) const {
    const auto found =
        states_.find(handle);
    if (found == states_.end()) {
        throw std::invalid_argument(
            "direct receive state references unknown handle");
    }
    return found->second;
}

void DirectReceiveCompletionBridge::observe(
    const Completion& completion) {
    auto& current =
        mutable_state(
            completion.handle);
    if (current.failed) {
        throw std::logic_error(
            "failed direct receive cannot accept more completions");
    }

    switch (completion.kind) {
    case CompletionKind::receive_complete:
        require_completion_bytes(
            completion,
            current.descriptor.byte_length);
        transition(
            current.ownership,
            BufferOwnership::receive_complete);
        current.completed_bytes =
            completion.bytes;
        return;

    case CompletionKind::receive_reposted:
        require_completion_bytes(
            completion,
            current.descriptor.byte_length);
        transition(
            current.ownership,
            BufferOwnership::reusable);
        return;

    case CompletionKind::failed:
        current.failed = true;
        return;

    case CompletionKind::local_buffer_reusable:
    case CompletionKind::remote_transport_ack:
    case CompletionKind::remote_consumer_complete:
        throw std::invalid_argument(
            "send-side completion delivered to receive bridge");
    }
}

void DirectReceiveCompletionBridge::acquire_consumer(
    TransferHandle handle) {
    auto& current =
        mutable_state(handle);
    if (current.failed) {
        throw std::logic_error(
            "failed direct receive cannot be acquired");
    }
    transition(
        current.ownership,
        BufferOwnership::consumer_acquired);
    transition(
        current.ownership,
        BufferOwnership::consumer_owned);
}

void DirectReceiveCompletionBridge::release_consumer(
    TransferHandle handle) {
    auto& current =
        mutable_state(handle);
    if (current.failed) {
        throw std::logic_error(
            "failed direct receive cannot be released");
    }
    transition(
        current.ownership,
        BufferOwnership::consumer_released);
}

void DirectReceiveCompletionBridge::submit_repost(
    TransferHandle handle) {
    auto& current =
        mutable_state(handle);
    if (current.failed) {
        throw std::logic_error(
            "failed direct receive cannot be reposted");
    }
    transition(
        current.ownership,
        BufferOwnership::repost_pending);
}

void DirectSendCompletionBridge::register_send(
    TransferHandle handle,
    std::uint64_t expected_bytes) {
    if (handle == 0 ||
        expected_bytes == 0) {
        throw std::invalid_argument(
            "direct send registration requires valid handle/bytes");
    }
    DirectSendCompletionState state;
    state.handle = handle;
    state.expected_bytes =
        expected_bytes;
    const auto inserted =
        states_.emplace(
            handle,
            state);
    if (!inserted.second) {
        throw std::logic_error(
            "direct send handle is already registered");
    }
}

DirectSendCompletionState&
DirectSendCompletionBridge::mutable_state(
    TransferHandle handle) {
    const auto found =
        states_.find(handle);
    if (found == states_.end()) {
        throw std::invalid_argument(
            "direct send completion references unknown handle");
    }
    return found->second;
}

const DirectSendCompletionState&
DirectSendCompletionBridge::state(
    TransferHandle handle) const {
    const auto found =
        states_.find(handle);
    if (found == states_.end()) {
        throw std::invalid_argument(
            "direct send state references unknown handle");
    }
    return found->second;
}

void DirectSendCompletionBridge::observe(
    const Completion& completion) {
    auto& current =
        mutable_state(
            completion.handle);
    if (current.failed) {
        throw std::logic_error(
            "failed direct send cannot accept more completions");
    }

    if (completion.kind !=
            CompletionKind::failed) {
        require_completion_bytes(
            completion,
            current.expected_bytes);
    }

    switch (completion.kind) {
    case CompletionKind::local_buffer_reusable:
        if (current.local_buffer_reusable) {
            throw std::logic_error(
                "duplicate local-buffer-reusable completion");
        }
        current.local_buffer_reusable = true;
        return;

    case CompletionKind::remote_transport_ack:
        if (current.remote_transport_acked) {
            throw std::logic_error(
                "duplicate remote transport ACK");
        }
        current.remote_transport_acked = true;
        return;

    case CompletionKind::remote_consumer_complete:
        if (!current.remote_transport_acked) {
            throw std::logic_error(
                "remote consumer-complete arrived before transport ACK");
        }
        if (current.remote_consumer_complete) {
            throw std::logic_error(
                "duplicate remote consumer-complete credit");
        }
        current.remote_consumer_complete = true;
        return;

    case CompletionKind::failed:
        current.failed = true;
        return;

    case CompletionKind::receive_complete:
    case CompletionKind::receive_reposted:
        throw std::invalid_argument(
            "receive-side completion delivered to send bridge");
    }
}

}  // namespace strix::transport
