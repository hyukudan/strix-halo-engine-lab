#include "strix/direct_transport_completion.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::transport::TransferDescriptor descriptor() {
    strix::transport::TransferDescriptor value;
    value.connection_epoch = 7;
    value.model_plan_id = 1;
    value.request_id = 2;
    value.boundary_id = 3;
    value.sequence = 4;
    value.shape_fingerprint = 5;
    value.dtype_id = 1;
    value.byte_length = 4096;
    return value;
}

}  // namespace

int main() {
    using namespace strix::transport;

    DirectReceiveCompletionBridge rx;
    const auto desc = descriptor();
    rx.register_receive(
        11,
        desc);
    rx.observe(
        Completion{
            11,
            CompletionKind::receive_complete,
            desc.byte_length,
        });
    expect(
        rx.state(11).ownership ==
            BufferOwnership::receive_complete,
        "DMA receive completion does not publish reusable buffer");

    rx.acquire_consumer(11);
    expect(
        rx.state(11).ownership ==
            BufferOwnership::consumer_owned,
        "consumer acquisition follows explicit receive-complete state");

    rx.release_consumer(11);
    rx.submit_repost(11);
    expect(
        rx.state(11).ownership ==
            BufferOwnership::repost_pending,
        "released receive buffer waits for repost completion");

    rx.observe(
        Completion{
            11,
            CompletionKind::receive_reposted,
            desc.byte_length,
        });
    expect(
        rx.state(11).ownership ==
            BufferOwnership::reusable,
        "receive buffer becomes reusable only after repost completion");

    DirectReceiveCompletionBridge skipped;
    skipped.register_receive(
        12,
        desc);
    bool skipped_rejected = false;
    try {
        skipped.submit_repost(12);
    } catch (const std::logic_error&) {
        skipped_rejected = true;
    }
    expect(
        skipped_rejected,
        "RX bridge rejects lifecycle skips before consumer release");

    DirectSendCompletionBridge tx;
    tx.register_send(
        21,
        desc.byte_length);
    tx.observe(
        Completion{
            21,
            CompletionKind::local_buffer_reusable,
            desc.byte_length,
        });
    expect(
        tx.state(21).local_buffer_reusable &&
            !tx.state(21).remote_transport_acked &&
            !tx.state(21).remote_consumer_complete,
        "local send-buffer reuse is independent of remote completion");

    bool early_consumer_rejected = false;
    try {
        tx.observe(
            Completion{
                21,
                CompletionKind::remote_consumer_complete,
                desc.byte_length,
            });
    } catch (const std::logic_error&) {
        early_consumer_rejected = true;
    }
    expect(
        early_consumer_rejected,
        "remote consumer-complete credit cannot precede transport ACK");

    tx.observe(
        Completion{
            21,
            CompletionKind::remote_transport_ack,
            desc.byte_length,
        });
    tx.observe(
        Completion{
            21,
            CompletionKind::remote_consumer_complete,
            desc.byte_length,
        });
    expect(
        tx.state(21).remote_transport_acked &&
            tx.state(21).remote_consumer_complete,
        "TX bridge distinguishes transport ACK from remote GPU consumer completion");

    DirectSendCompletionBridge bad_bytes;
    bad_bytes.register_send(
        31,
        desc.byte_length);
    bool byte_mismatch_rejected = false;
    try {
        bad_bytes.observe(
            Completion{
                31,
                CompletionKind::remote_transport_ack,
                desc.byte_length - 1,
            });
    } catch (const std::invalid_argument&) {
        byte_mismatch_rejected = true;
    }
    expect(
        byte_mismatch_rejected,
        "completion bridge rejects mismatched byte accounting");

    if (failures == 0) {
        std::cout
            << "direct transport completion tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
