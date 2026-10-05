#include "strix/direct_transport_native_session.hpp"

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
        value.name = "fake-session-direct";
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
        const strix::transport::DirectTransportLease& lease,
        const strix::transport::DirectTransportLinkPoolBinding&) override {
        imported_generation = lease.lease_generation;
        ++imports;
    }

    void release_link_pool(
        const strix::transport::DirectTransportLease& lease) noexcept override {
        released_generation = lease.lease_generation;
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
        auto event = events.front();
        events.pop_front();
        return event;
    }

    int imports = 0;
    int releases = 0;
    int commits = 0;
    int credits = 0;
    int reposts = 0;
    std::uint64_t imported_generation = 0;
    std::uint64_t released_generation = 0;
    std::deque<strix::transport::DirectNativeEvent> events;
};

strix::transport::DirectLinkLifecycle link() {
    strix::transport::DirectLinkLifecycle value;
    value.physical_link_id = "usb4-a";
    value.state =
        strix::transport::DirectLinkLifecycleState::direct_ready;
    value.connection_epoch = 17;
    value.control_link_ready = true;
    value.peer_reconciled = true;
    value.receive_ring_primed = true;
    value.exclusive_lease_available = true;
    value.completion_rearm_qualified = true;
    value.dma_buf_import_supported = true;
    return value;
}

strix::transport::DirectTransportPoolMemoryBinding pools() {
    strix::transport::DirectTransportPoolMemoryBinding value;
    value.binding_identity = 0x1234ULL;
    value.total_gpu_bytes = 8192;
    value.links.resize(1);
    value.links[0].tx.address =
        static_cast<std::uintptr_t>(0x10000000ULL);
    value.links[0].tx.bytes = 4096;
    value.links[0].tx.allocation_generation = 1;
    value.links[0].tx.domain =
        strix::transport::TransportMemoryDomain::gpu_transport_pool;
    value.links[0].tx.access =
        strix::transport::TransportBufferAccess::read_write;
    value.links[0].rx =
        value.links[0].tx;
    value.links[0].rx.address =
        static_cast<std::uintptr_t>(0x10001000ULL);
    value.links[0].rx.allocation_generation = 2;
    return value;
}

strix::transport::TransferDescriptor descriptor() {
    strix::transport::TransferDescriptor value;
    value.connection_epoch = 17;
    value.model_plan_id = 1;
    value.request_id = 2;
    value.boundary_id = 3;
    value.sequence = 4;
    value.shape_fingerprint = 5;
    value.dtype_id = 6;
    value.byte_length = 128;
    return value;
}

}  // namespace

int main() {
    using namespace strix::transport;

    auto lifecycle = link();
    auto memory = pools();
    DirectTransportLeaseRegistry leases;
    FakeBackend backend;

    {
        auto session =
            DirectTransportNativeLinkSession::open(
                backend, leases, lifecycle, memory, 0);
        expect(
            session.active() &&
                backend.imports == 1 &&
                leases.leased("usb4-a") &&
                session.lease().lease_generation ==
                    backend.imported_generation,
            "native link session acquires lease and imports exact pool generation");

        const auto desc = descriptor();
        session.register_receive(10, desc);
        session.register_send(20, desc);

        backend.events.push_back({
            Completion{10, CompletionKind::receive_complete, 128},
            true,
        });
        expect(
            session.poll_event() &&
                session.receive_state(10).ownership ==
                    BufferOwnership::receive_complete,
            "session routes receive event through owned completion bridge");

        session.acquire_receive(10);
        session.release_receive(10);
        session.publish_receive_credit(10);
        expect(
            backend.credits == 1 &&
                session.receive_state(10).
                    consumer_credit_published,
            "session publishes remote consumer credit on current lease");
        session.repost_receive(10);
        expect(
            backend.reposts == 1 &&
                session.receive_state(10).ownership ==
                    BufferOwnership::repost_pending,
            "session repost remains bound to current lease");

        backend.events.push_back({
            Completion{10, CompletionKind::receive_reposted, 128},
            true,
        });
        static_cast<void>(session.poll_event());
        expect(
            session.receive_state(10).ownership ==
                BufferOwnership::reusable,
            "session observes repost completion");

        backend.events.push_back({
            Completion{20, CompletionKind::remote_transport_ack, 128},
            false,
        });
        backend.events.push_back({
            Completion{20, CompletionKind::remote_consumer_complete, 128},
            false,
        });
        static_cast<void>(session.poll_event());
        static_cast<void>(session.poll_event());
        expect(
            session.send_state(20).remote_consumer_complete,
            "session routes send credits through owned completion bridge");

        GpuCommitRecord record;
        record.protocol_version = desc.protocol_version;
        record.commit_word = gpu_commit_word;
        record.connection_epoch = desc.connection_epoch;
        record.sequence = desc.sequence;
        record.payload_bytes = 128;
        record.stripe_index = 0;
        record.stripe_count = 1;
        session.publish_commit(20, desc, record);
        expect(
            backend.commits == 1,
            "session publishes validated commit on leased backend");

        session.close();
        expect(
            !session.active() &&
                !leases.leased("usb4-a") &&
                backend.releases == 1 &&
                backend.released_generation ==
                    backend.imported_generation,
            "explicit close releases pool before lease lifetime ends");
    }

    {
        auto session =
            DirectTransportNativeLinkSession::open(
                backend, leases, lifecycle, memory, 0);
        expect(
            leases.leased("usb4-a"),
            "second session reacquires released physical link");
    }
    expect(
        !leases.leased("usb4-a") &&
            backend.releases == 2,
        "destructor releases imported pool and lease");

    {
        auto session =
            DirectTransportNativeLinkSession::open(
                backend, leases, lifecycle, memory, 0);
        ++lifecycle.connection_epoch;
        bool stale_rejected = false;
        try {
            static_cast<void>(session.poll_event());
        } catch (const std::logic_error&) {
            stale_rejected = true;
        }
        expect(
            stale_rejected,
            "session rejects operations after leased link epoch changes");
    }

    if (failures == 0) {
        std::cout << "direct native link session tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
