#pragma once

#include "strix/transport.hpp"

#include <cstdint>
#include <unordered_map>

namespace strix::transport {

struct DirectReceiveCompletionState {
    TransferHandle handle = 0;
    TransferDescriptor descriptor;
    BufferOwnership ownership =
        BufferOwnership::transport_owned;
    std::uint64_t completed_bytes = 0;
    bool consumer_credit_published = false;
    bool failed = false;
};

class DirectReceiveCompletionBridge {
public:
    void register_receive(
        TransferHandle handle,
        const TransferDescriptor& descriptor);

    void observe(
        const Completion& completion);

    void acquire_consumer(
        TransferHandle handle);

    void release_consumer(
        TransferHandle handle);

    void publish_consumer_credit(
        TransferHandle handle);

    void submit_repost(
        TransferHandle handle);

    [[nodiscard]] const DirectReceiveCompletionState&
    state(TransferHandle handle) const;

private:
    DirectReceiveCompletionState&
    mutable_state(TransferHandle handle);

    std::unordered_map<
        TransferHandle,
        DirectReceiveCompletionState> states_;
};

struct DirectSendCompletionState {
    TransferHandle handle = 0;
    std::uint64_t expected_bytes = 0;
    bool local_buffer_reusable = false;
    bool remote_transport_acked = false;
    bool remote_consumer_complete = false;
    bool failed = false;
};

class DirectSendCompletionBridge {
public:
    void register_send(
        TransferHandle handle,
        std::uint64_t expected_bytes);

    void observe(
        const Completion& completion);

    [[nodiscard]] const DirectSendCompletionState&
    state(TransferHandle handle) const;

private:
    DirectSendCompletionState&
    mutable_state(TransferHandle handle);

    std::unordered_map<
        TransferHandle,
        DirectSendCompletionState> states_;
};

}  // namespace strix::transport
