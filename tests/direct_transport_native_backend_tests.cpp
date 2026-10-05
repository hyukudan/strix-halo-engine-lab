#include "strix/direct_transport_native_backend.hpp"

#include <deque>
#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeBackend final
    : public strix::transport::DirectTransportNativeBackend {
public:
    strix::transport::BackendCapabilities capabilities() const override {
        strix::transport::BackendCapabilities value;
        value.name = "fake-direct";
        value.maximum_payload_bytes = 1U << 20U;
        value.external_dma = true;
        value.supports_gpu_transport_pool = true;
        value.requires_exclusive_lease = true;
        value.explicit_receive_repost = true;
        value.gpu_commit_record = true;
        value.buffer_mode =
            strix::transport::TransportBufferMode::imported_gpu_dma_buf;
        return value;
    }

    strix::transport::DirectNativeCapabilities
    direct_capabilities() const override {
        return {
            true, true, true, true, true, true, true,
        };
    }

    void import_link_pool(
        const strix::transport::DirectTransportLease&,
        const strix::transport::DirectTransportLinkPoolBinding&) override {
        ++imports;
    }

    void release_link_pool(
        const strix::transport::DirectTransportLease&) noexcept override {
        ++releases;
    }

    void publish_gpu_commit(
        strix::transport::TransferHandle,
        const strix::transport::GpuCommitRecord&) override {
        ++commits;
    }

    void publish_consumer_credit(
        strix::transport::TransferHandle,
        std::uint64_t) override {
        ++credits;
    }

    void repost_receive(
        strix::transport::TransferHandle) override {
        ++reposts;
    }

    std::optional<strix::transport::DirectNativeEvent>
    poll_direct_event() override {
        if (events.empty()) {
            return std::nullopt;
        }
        auto value = events.front();
        events.pop_front();
        return value;
    }

    int imports = 0;
    int releases = 0;
    int commits = 0;
    int credits = 0;
    int reposts = 0;
    std::deque<strix::transport::DirectNativeEvent> events;
};

}  // namespace

int main() {
    using namespace strix::transport;

    DirectLinkLifecycle link;
    link.physical_link_id = "usb4-a";
    link.state = DirectLinkLifecycleState::direct_ready;
    link.connection_epoch = 17;
    link.control_link_ready = true;
    link.peer_reconciled = true;
    link.receive_ring_primed = true;
    link.exclusive_lease_available = true;
    link.completion_rearm_qualified = true;
    link.dma_buf_import_supported = true;

    DirectTransportPoolMemoryBinding pools;
    pools.binding_identity = 0x1234ULL;
    pools.total_gpu_bytes = 8192;
    pools.links.resize(1);

    DirectTransportLeaseRegistry leases;
    const auto lease =
        leases.acquire(link, pools, 0);

    FakeBackend backend;
    const auto readiness =
        assess_direct_native_readiness(
            backend.capabilities(),
            backend.direct_capabilities());
    expect(readiness.ready && readiness.blockers.empty(),
           "complete native capabilities satisfy direct backend contract");

    prepare_direct_native_link(
        backend, leases, link, pools, lease);
    expect(backend.imports == 1,
           "native link preparation imports the leased link pool");

    TransferDescriptor descriptor;
    descriptor.connection_epoch = 17;
    descriptor.model_plan_id = 1;
    descriptor.request_id = 2;
    descriptor.boundary_id = 3;
    descriptor.sequence = 4;
    descriptor.shape_fingerprint = 5;
    descriptor.dtype_id = 6;
    descriptor.byte_length = 128;

    DirectReceiveCompletionBridge receives;
    DirectSendCompletionBridge sends;
    receives.register_receive(10, descriptor);
    sends.register_send(20, descriptor.byte_length);

    backend.events.push_back({
        Completion{10, CompletionKind::receive_complete, 128},
        true,
    });
    expect(observe_direct_native_event(
               backend, receives, sends) &&
               receives.state(10).ownership ==
                   BufferOwnership::receive_complete,
           "native receive event enters portable ownership bridge");

    receives.acquire_consumer(10);
    receives.release_consumer(10);
    publish_direct_native_consumer_credit(
        backend, receives, 10);
    expect(
        backend.credits == 1 &&
            receives.state(10).consumer_credit_published,
        "native consumer credit is emitted exactly after consumer release");

    bool duplicate_credit_rejected = false;
    try {
        publish_direct_native_consumer_credit(
            backend, receives, 10);
    } catch (const std::logic_error&) {
        duplicate_credit_rejected = true;
    }
    expect(
        duplicate_credit_rejected &&
            backend.credits == 1,
        "duplicate remote consumer credit is rejected before backend call");

    repost_direct_native_receive(
        backend, receives, 10);
    expect(backend.reposts == 1 &&
               receives.state(10).ownership ==
                   BufferOwnership::repost_pending,
           "native repost is submitted only after consumer release");

    backend.events.push_back({
        Completion{10, CompletionKind::receive_reposted, 128},
        true,
    });
    static_cast<void>(
        observe_direct_native_event(
            backend, receives, sends));
    expect(receives.state(10).ownership ==
               BufferOwnership::reusable,
           "native repost completion returns receive buffer to reusable");

    backend.events.push_back({
        Completion{20, CompletionKind::remote_transport_ack, 128},
        false,
    });
    backend.events.push_back({
        Completion{20, CompletionKind::remote_consumer_complete, 128},
        false,
    });
    static_cast<void>(
        observe_direct_native_event(
            backend, receives, sends));
    static_cast<void>(
        observe_direct_native_event(
            backend, receives, sends));
    expect(sends.state(20).remote_transport_acked &&
               sends.state(20).remote_consumer_complete,
           "native send events preserve transport ACK and consumer-credit ordering");

    GpuCommitRecord record;
    record.protocol_version = descriptor.protocol_version;
    record.commit_word = gpu_commit_word;
    record.connection_epoch = descriptor.connection_epoch;
    record.sequence = descriptor.sequence;
    record.payload_bytes = 128;
    record.stripe_index = 0;
    record.stripe_count = 1;
    publish_direct_native_commit(
        backend, 20, descriptor, record);
    expect(backend.commits == 1,
           "validated GPU commit reaches native backend");

    auto incomplete = backend.direct_capabilities();
    incomplete.system_scope_gpu_commit = false;
    const auto blocked =
        assess_direct_native_readiness(
            backend.capabilities(),
            incomplete);
    expect(!blocked.ready && !blocked.blockers.empty(),
           "missing native commit primitive blocks direct readiness");

    if (failures == 0) {
        std::cout << "direct native backend tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
