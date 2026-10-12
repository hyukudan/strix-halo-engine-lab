#include "strix/direct_transport_driver.hpp"
#include "strix/direct_transport_native_session.hpp"
#include "direct_native_fixture.hpp"

#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace {
using namespace strix::transport;
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template<class Function> bool rejected(Function&& function) {
    try { function(); } catch (const std::exception&) { return true; }
    return false;
}
DirectLinkLifecycle link() {
    DirectLinkLifecycle value;
    value.physical_link_id = "synthetic-physical-link";
    value.connection_epoch = 17;
    value.state = DirectLinkLifecycleState::direct_ready;
    value.control_link_ready = value.peer_reconciled = value.receive_ring_primed = true;
    value.exclusive_lease_available = value.dma_buf_import_supported = value.completion_rearm_qualified = true;
    value.expected_native_interface = synthetic_native_identity(value);
    return value;
}
DirectDriverBinding device() { return {1, "synthetic-character-device-instance", 7, *link().expected_native_interface}; }
DirectTransportPoolMemoryBinding pools() {
    DirectTransportPoolMemoryBinding value;
    value.binding_identity = 5;
    value.total_gpu_bytes = 8192;
    value.links.push_back({{0x10000, 4096, 1, TransportMemoryDomain::gpu_transport_pool, TransportBufferAccess::read_write},
                           {0x20000, 4096, 2, TransportMemoryDomain::gpu_transport_pool, TransportBufferAccess::read_write}});
    return value;
}
DirectTransportLease lease() { return {link().physical_link_id, 17, 5, 0, 1}; }
TransferDescriptor descriptor(std::uint64_t sequence = 1, std::uint64_t request = 8) {
    return {1, 17, 5, request, 2, 7, sequence, 9, 1, 128};
}
GpuCommitRecord commit(const TransferDescriptor& value) {
    return {1, gpu_commit_word, value.connection_epoch, value.sequence,
            static_cast<std::uint32_t>(value.byte_length), 0, 1};
}
struct Counts { int freed = 0, queries = 0, activated = 0, prepared = 0, submitted = 0,
    canceled = 0, credited = 0, reposted = 0, retired = 0, release_requests = 0, released = 0; };
class Import final : public DirectDriverImport {
public:
    Import(DirectDriverImportBinding initial_value, std::shared_ptr<Counts> initial_counts)
        : value(std::move(initial_value)), counts(std::move(initial_counts)) {}
    ~Import() override { ++counts->freed; }
    const DirectDriverImportBinding& binding() const noexcept override { return value; }
    DirectDriverImportBinding value;
    std::shared_ptr<Counts> counts;
};
// Entirely synthetic host records; no ioctl numbers, DMA-BUF file descriptors,
// external DMA, GPU events or hardware qualification are present in this fixture.
class Io final : public DirectTransportDriverIo {
public:
    Io() {
        observed.binding = device();
        observed.backend.maximum_payload_bytes = 4096;
        observed.backend.external_dma = observed.backend.supports_gpu_transport_pool = true;
        observed.backend.requires_exclusive_lease = observed.backend.explicit_receive_repost = true;
        observed.backend.gpu_commit_record = true;
        observed.backend.buffer_mode = TransportBufferMode::imported_gpu_dma_buf;
        observed.native = synthetic_native_capabilities(observed.binding.interface);
        current = observed.binding;
    }
    DirectDriverQuery query() override { ++counts->queries; if (throw_query) throw std::runtime_error("query inaccessible"); return observed; }
    std::optional<DirectDriverBinding> current_binding() const override { return current; }
    std::shared_ptr<DirectDriverImport> prepare_import(const DirectDriverBinding& binding,
        const DirectTransportLease& token, const DirectTransportLinkPoolBinding& pool) override {
        auto result = std::make_shared<Import>(DirectDriverImportBinding{binding, token, pool, next_import++}, counts);
        if (foreign_import) ++result->value.pool.rx.allocation_generation;
        last_import = result;
        return result;
    }
    void activate_import(const std::shared_ptr<DirectDriverImport>&) override {
        ++counts->activated;
        if (throw_activate) throw std::runtime_error("partial native import activation");
    }
    void prepare(const DirectDriverOperation& operation) override {
        ++counts->prepared;
        operations[operation.handle] = operation;
        if (throw_prepare) throw std::runtime_error("prepare error");
    }
    void submit(const DirectDriverOperation&) override { ++counts->submitted; if (throw_submit) throw std::runtime_error("queued then errored"); }
    void cancel(const DirectDriverOperation&) override { ++counts->canceled; }
    DirectDriverVisibilityProof proof(const DirectDriverOperation& operation, DirectDriverVisibility visibility) {
        DirectDriverVisibilityProof result{operation, visibility};
        if (mutate_proof) mutate_proof(result);
        return result;
    }
    DirectDriverVisibilityProof publish_gpu_commit(const DirectDriverOperation& operation, const GpuCommitRecord&) override {
        return proof(operation, DirectDriverVisibility::producer_release_to_external_dma);
    }
    DirectDriverCommitRead read_gpu_commit(const DirectDriverOperation& operation) override {
        auto record = commit(operation.descriptor);
        if (stale_commit) --record.sequence;
        if (replace_device_during_acquire) ++current->device_instance_generation;
        return {record, proof(operation, DirectDriverVisibility::consumer_acquire_from_external_dma)};
    }
    DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation& operation) override {
        return proof(operation, consumer_done ? DirectDriverVisibility::consumer_release_complete : DirectDriverVisibility::unavailable);
    }
    void publish_consumer_credit(const DirectDriverOperation&, std::uint64_t) override { ++counts->credited; }
    void repost_receive(const DirectDriverOperation&) override { ++counts->reposted; }
    std::optional<DirectDriverEvent> poll_event() override {
        if (events.empty()) return std::nullopt;
        auto result = events.front(); events.pop_front();
        if (replace_device_during_event) ++current->device_instance_generation;
        return result;
    }
    void retire(const DirectDriverOperation& operation) override { ++counts->retired; operations.erase(operation.handle); }
    void request_import_release(const std::shared_ptr<DirectDriverImport>&) override {
        ++counts->release_requests;
        if (throw_release) throw std::runtime_error("release request error");
    }
    DirectDriverQuiescence poll_import_quiescence(const std::shared_ptr<DirectDriverImport>& imported) override {
        DirectDriverQuiescence result{imported->binding(), pending_release_polls-- > 0
            ? DirectDriverQuiescenceStatus::pending : DirectDriverQuiescenceStatus::quiescent};
        if (mutate_quiescence) mutate_quiescence(result);
        if (mutate_token_during_quiescence) {
            ++std::static_pointer_cast<Import>(imported)->value.pool.rx.allocation_generation;
        }
        return result;
    }
    void release_import(const std::shared_ptr<DirectDriverImport>&) override { ++counts->released; operations.clear(); }
    void event(TransferHandle handle, CompletionKind kind) {
        const auto& operation = operations.at(handle);
        events.push_back({operation, {{handle, kind, operation.descriptor.byte_length}, operation.receive_path, 1.0, 128}});
    }
    DirectDriverQuery observed;
    std::optional<DirectDriverBinding> current;
    std::shared_ptr<Counts> counts = std::make_shared<Counts>();
    std::weak_ptr<DirectDriverImport> last_import;
    std::unordered_map<TransferHandle, DirectDriverOperation> operations;
    std::deque<DirectDriverEvent> events;
    std::function<void(DirectDriverVisibilityProof&)> mutate_proof;
    std::function<void(DirectDriverQuiescence&)> mutate_quiescence;
    std::uint64_t next_import = 1;
    int pending_release_polls = 0;
    bool throw_query = false, foreign_import = false, throw_activate = false, throw_prepare = false,
         throw_submit = false, throw_release = false, consumer_done = false, stale_commit = false,
         replace_device_during_event = false, replace_device_during_acquire = false,
         mutate_token_during_quiescence = false;
};
DirectDriverPolicy policy(std::uint64_t polls = 4) { return {4096, 8192, 2, polls}; }
TransferHandle receive(DirectTransportDriverBackend& backend, Io& io) {
    const auto handle = backend.prepare_receive(descriptor(), pools().links[0].rx);
    backend.submit(handle); io.event(handle, CompletionKind::receive_complete);
    static_cast<void>(backend.poll_direct_event());
    return handle;
}
}  // namespace

int main() {
    {
        DirectTransportDriverIo unavailable;
        DirectTransportDriverBackend backend(unavailable, device(), policy());
        expect(backend.direct_capabilities().query_status == DirectNativeQueryStatus::unavailable &&
               !assess_direct_native_readiness(backend.capabilities(), backend.direct_capabilities(), link()).ready &&
               rejected([&] { backend.import_link_pool(lease(), pools().links[0]); }),
               "default driver IO remains unavailable without inventing a kernel ABI");
    }
    for (int mismatch = 0; mismatch < 13; ++mismatch) {
        Io io;
        if (mismatch == 0) io.observed.native.query_status = DirectNativeQueryStatus::unsupported;
        if (mismatch == 1) io.observed.native.query_status = DirectNativeQueryStatus::inaccessible;
        if (mismatch == 2) ++io.observed.binding.adapter_contract_version;
        if (mismatch == 3) ++io.observed.binding.device_instance_generation;
        if (mismatch == 4) io.observed.binding.interface.abi_identity += "changed";
        if (mismatch == 5) io.observed.native.observed_interface->peer_endpoint_identity += "changed";
        if (mismatch == 6) ++io.observed.native.observed_interface->connection_epoch;
        if (mismatch == 7) io.observed.native.observed_interface->physical_link_id += "changed";
        if (mismatch == 8) io.observed.native.external_dma_visibility_qualified = false;
        if (mismatch == 9) io.observed.native.completion_ownership_verified = false;
        if (mismatch == 10) io.throw_query = true;
        if (mismatch == 11) io.observed.native.imported_tx_rx_ownership_verified = false;
        if (mismatch == 12) io.observed.native.completion_event_source = false;
        DirectTransportDriverBackend backend(io, device(), policy());
        expect(!assess_direct_native_readiness(backend.capabilities(), backend.direct_capabilities(), link()).ready &&
               rejected([&] { backend.import_link_pool(lease(), pools().links[0]); }) && io.counts->activated == 0,
               "unsupported/unavailable/incompatible query, device, peer, ABI or synchronization cannot activate import");
    }
    {
        Io io;
        DirectTransportDriverBackend backend(io, device(), policy());
        auto physical = link(); auto memory = pools(); DirectTransportLeaseRegistry leases;
        auto session = DirectTransportNativeLinkSession::open(backend, leases, physical, memory, 0);
        const auto tx = session.prepare_send(descriptor(session.reserve_send_sequence()), memory.links[0].tx);
        expect(rejected([&] { session.submit(tx); }) && io.counts->submitted == 0,
               "TX submission requires operation-bound external-DMA producer release");
        session.publish_commit(tx, descriptor(1), commit(descriptor(1))); session.submit(tx);
        io.event(tx, CompletionKind::local_buffer_reusable); expect(session.poll_event(), "TX local completion enters existing session");
        io.event(tx, CompletionKind::remote_transport_ack); expect(session.poll_event(), "TX ACK enters existing session");
        io.event(tx, CompletionKind::remote_consumer_complete); expect(session.poll_event() && backend.active_operations() == 0,
               "TX operation retires only after all distinct ownership completions");
        const auto rx = session.prepare_receive(descriptor(2, 9), memory.links[0].rx); session.submit(rx);
        io.event(rx, CompletionKind::receive_complete); expect(session.poll_event(), "RX DMA completion enters existing session");
        static_cast<void>(session.read_commit(rx, descriptor(2, 9))); session.acquire_receive(rx); session.release_receive(rx);
        io.consumer_done = true;
        session.publish_receive_credit(rx); session.repost_receive(rx);
        io.event(rx, CompletionKind::receive_reposted); expect(session.poll_event() && backend.active_operations() == 0 &&
            session.receive_state(rx).ownership == BufferOwnership::reusable && io.counts->credited == 1,
            "RX acquire, completed-consumer proof, credit and repost complete existing native lifecycle");
        session.close();
        expect(!backend.retains_import() && io.counts->freed == 1 && io.counts->queries == 1,
               "exact quiescence releases owned import without requerying the device on each operation");
    }
    for (int drift = 0; drift < 5; ++drift) {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        if (drift == 0) ++io.current->device_instance_generation; // Same path, replacement character device.
        if (drift == 1) io.current->interface.driver_identity += "changed";
        if (drift == 2) io.current->interface.peer_endpoint_identity += "changed";
        if (drift == 3) ++io.current->interface.connection_epoch;
        if (drift == 4) io.current.reset(); // Provider invalidates lost qualification.
        expect(rejected([&] { static_cast<void>(backend.prepare_receive(descriptor(), pools().links[0].rx)); }) &&
            backend.failed() && io.counts->prepared == 0 && backend.retains_import(),
            "cached same-path device replacement/driver/peer/epoch drift is fenced before native IO");
        backend.release_link_pool(lease());
    }
    {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        const auto first = backend.prepare_send(descriptor(1), pools().links[0].tx);
        backend.publish_gpu_commit(first, commit(descriptor(1))); backend.submit(first);
        for (const auto kind : {CompletionKind::local_buffer_reusable, CompletionKind::remote_transport_ack,
                               CompletionKind::remote_consumer_complete}) {
            io.event(first, kind); static_cast<void>(backend.poll_direct_event());
        }
        backend.release_link_pool(lease());
        auto reopened = lease(); ++reopened.lease_generation;
        backend.import_link_pool(reopened, pools().links[0]);
        expect(rejected([&] { static_cast<void>(backend.prepare_send(descriptor(1), pools().links[0].tx)); }) &&
               io.counts->prepared == 1, "same-epoch reopen retains TX sequence history after terminal retirement");
        const auto next = backend.prepare_send(descriptor(2, 11), pools().links[0].tx);
        expect(next != first, "new request receives a distinct process-owned handle after pool reopen");
        backend.release_link_pool(reopened);
    }
    for (int partial = 0; partial < 3; ++partial) {
        Io io; io.throw_activate = true; io.pending_release_polls = 20;
        if (partial == 1) io.throw_release = true;
        if (partial == 2) io.mutate_quiescence = [](auto& value) { ++value.imported.import_generation; };
        DirectTransportDriverBackend backend(io, device(), policy(1));
        expect(rejected([&] { backend.import_link_pool(lease(), pools().links[0]); }) && backend.failed() &&
            backend.retains_import() && io.counts->freed == 0 && io.counts->released == 0,
            "partial activation/cancel callback/foreign quiescence retains import ownership");
        io.throw_release = false; io.mutate_quiescence = {}; io.pending_release_polls = 0;
        backend.release_link_pool(lease());
        expect(!backend.retains_import() && io.counts->freed == 1 && backend.failed(),
               "later matching quiescence cleans partial setup without recovering failed adapter");
    }
    {
        Io io; io.foreign_import = true;
        DirectTransportDriverBackend backend(io, device(), policy());
        expect(rejected([&] { backend.import_link_pool(lease(), pools().links[0]); }) && io.counts->activated == 0 &&
            io.counts->freed == 1, "foreign pool generation is rejected before activation, so unarmed token can release");
    }
    {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        io.mutate_token_during_quiescence = true;
        backend.release_link_pool(lease());
        expect(backend.failed() && backend.retains_import() && io.counts->released == 0 && io.counts->freed == 0,
               "mutated import token cannot be released using a cached matching quiescence proof");
        io.mutate_token_during_quiescence = false;
        auto original = std::static_pointer_cast<Import>(io.last_import.lock());
        original->value.pool.rx.allocation_generation = pools().links[0].rx.allocation_generation;
        original.reset();
        backend.release_link_pool(lease());
        expect(!backend.retains_import() && io.counts->freed == 1,
               "only intact token with later matching quiescence retires cleanup ownership");
    }
    for (int bad = 0; bad < 9; ++bad) {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        const auto rx = receive(backend, io);
        if (bad == 8) io.replace_device_during_acquire = true;
        io.mutate_proof = [bad](auto& proof) {
            if (bad == 0) proof.visibility = DirectDriverVisibility::unavailable;
            if (bad == 1) ++proof.operation.imported.device.device_instance_generation;
            if (bad == 2) ++proof.operation.imported.lease.lease_generation;
            if (bad == 3) ++proof.operation.imported.pool.rx.allocation_generation;
            if (bad == 4) ++proof.operation.handle;
            if (bad == 5) ++proof.operation.descriptor.request_id;
            if (bad == 6) ++proof.operation.descriptor.connection_epoch;
            if (bad == 7) ++proof.operation.descriptor.sequence;
        };
        expect(rejected([&] { static_cast<void>(backend.read_gpu_commit(rx)); }) && backend.failed() &&
            io.counts->credited == 0 && io.counts->reposted == 0 && backend.retains_import(),
            "foreign/stale/unsupported acquire proof cannot authorize consumer credit or pool reuse");
        backend.release_link_pool(lease());
    }
    {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        const auto rx = receive(backend, io); static_cast<void>(backend.read_gpu_commit(rx));
        expect(rejected([&] { backend.publish_consumer_credit(rx, 128); }) && backend.failed() && io.counts->credited == 0 &&
            io.counts->reposted == 0, "CPU callback/RX completion without completed-consumer release proof cannot publish credit");
        backend.release_link_pool(lease());
    }
    for (int bad = 0; bad < 4; ++bad) {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        const auto rx = receive(backend, io); static_cast<void>(backend.read_gpu_commit(rx)); io.consumer_done = true;
        io.mutate_proof = [bad](auto& proof) {
            if (bad == 0) proof.visibility = DirectDriverVisibility::consumer_acquire_from_external_dma;
            if (bad == 1) ++proof.operation.descriptor.request_id;
            if (bad == 2) ++proof.operation.imported.device.device_instance_generation;
            if (bad == 3) ++proof.operation.imported.import_generation;
        };
        expect(rejected([&] { backend.publish_consumer_credit(rx, 128); }) && backend.failed() &&
               io.counts->credited == 0 && io.counts->reposted == 0,
               "acquire/foreign request/device/import completion proof cannot forge consumer credit");
        backend.release_link_pool(lease());
    }
    for (int bad = 0; bad < 3; ++bad) {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        const auto tx = backend.prepare_send(descriptor(), pools().links[0].tx);
        io.mutate_proof = [bad](auto& proof) {
            if (bad == 0) proof.visibility = DirectDriverVisibility::unavailable;
            if (bad == 1) ++proof.operation.imported.device.device_instance_generation;
            if (bad == 2) ++proof.operation.descriptor.sequence;
        };
        expect(rejected([&] { backend.publish_gpu_commit(tx, commit(descriptor())); }) && backend.failed() &&
               io.counts->submitted == 0 && backend.retains_import(),
               "generic capability bits and foreign producer release proof cannot authorize TX DMA");
        backend.release_link_pool(lease());
    }
    for (int stage = 0; stage < 2; ++stage) {
        Io io; DirectTransportDriverBackend backend(io, device(), policy(1)); backend.import_link_pool(lease(), pools().links[0]);
        io.pending_release_polls = 10;
        if (stage == 0) {
            io.throw_prepare = true;
            expect(rejected([&] { static_cast<void>(backend.prepare_receive(descriptor(), pools().links[0].rx)); }),
                   "preparation callback failure is exposed");
        } else {
            const auto rx = backend.prepare_receive(descriptor(), pools().links[0].rx); io.throw_submit = true;
            expect(rejected([&] { backend.submit(rx); }), "submit error after possible enqueue is exposed");
        }
        backend.release_link_pool(lease());
        expect(backend.failed() && backend.retains_import() && backend.active_operations() == 1 && io.counts->freed == 0,
               "prepare/submit callback exception keeps imported allocation and operation ownership until drain");
        io.pending_release_polls = 0; backend.release_link_pool(lease());
        expect(!backend.retains_import() && io.counts->freed == 1,
               "typed later quiescence cleans callback failure without an early free");
    }
    for (int bad = 0; bad < 8; ++bad) {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        const auto rx = backend.prepare_receive(descriptor(), pools().links[0].rx); backend.submit(rx);
        io.event(rx, CompletionKind::receive_complete);
        auto& value = io.events.back();
        if (bad == 0) ++value.operation.descriptor.request_id;
        if (bad == 1) ++value.operation.imported.device.device_instance_generation;
        if (bad == 2) ++value.operation.imported.import_generation;
        if (bad == 3) ++value.event.completion.bytes;
        if (bad == 4) value.event.receive_path = false;
        if (bad == 5) value.event.completion.kind = CompletionKind::receive_reposted;
        if (bad == 6) value.event.monotonic_seconds = std::numeric_limits<double>::infinity();
        if (bad == 7) io.replace_device_during_event = true;
        expect(rejected([&] { static_cast<void>(backend.poll_direct_event()); }) && backend.failed() && backend.retains_import(),
            "event ownership/full descriptor/device/import/bytes/direction/ordering/time are validated before exposure");
        backend.release_link_pool(lease());
    }
    {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        const auto rx = receive(backend, io); io.event(rx, CompletionKind::receive_complete);
        expect(rejected([&] { static_cast<void>(backend.poll_direct_event()); }) && backend.failed(),
               "duplicate native completion invalidates the adapter before credit");
        backend.release_link_pool(lease());
    }
    {
        Io io; DirectTransportDriverBackend backend(io, device(), policy()); backend.import_link_pool(lease(), pools().links[0]);
        auto first_buffer = pools().links[0].rx; first_buffer.bytes = 128;
        const auto first = backend.prepare_receive(descriptor(1), first_buffer);
        expect(rejected([&] { static_cast<void>(backend.prepare_receive(descriptor(2), first_buffer)); }) &&
            io.counts->prepared == 1, "pending destination cannot overlap a second operation despite spare capacity");
        auto second_buffer = first_buffer; second_buffer.address += 128;
        static_cast<void>(backend.prepare_receive(descriptor(2, 10), second_buffer));
        auto third_buffer = second_buffer; third_buffer.address += 128;
        expect(rejected([&] { static_cast<void>(backend.prepare_receive(descriptor(3), third_buffer)); }) &&
            backend.active_operations() == 2, "operation metadata has an explicit hard bound");
        backend.cancel(first);
        expect(backend.active_operations() == 2 && backend.retains_import(), "cancel callback return does not retire buffer ownership");
        backend.release_link_pool(lease());
    }
    {
        Io io; auto small_policy = policy(); small_policy.maximum_payload_bytes = 64;
        DirectTransportDriverBackend backend(io, device(), small_policy); backend.import_link_pool(lease(), pools().links[0]);
        expect(rejected([&] { static_cast<void>(backend.prepare_receive(descriptor(), pools().links[0].rx)); }) && io.counts->prepared == 0,
               "negotiated/admission payload bound is checked before native preparation");
        backend.release_link_pool(lease());
    }
    {
        Io io; DirectTransportDriverBackend backend(io, device(), policy(1));
        auto physical = link(); auto memory = pools(); DirectTransportLeaseRegistry leases;
        auto session = DirectTransportNativeLinkSession::open(backend, leases, physical, memory, 0);
        io.pending_release_polls = 100;
        expect(rejected([&] { session.close(); }) && session.active() && backend.retains_import() &&
               leases.leased(physical.physical_link_id) && io.counts->freed == 0,
               "unproven native release retains both session and exclusive host lease");
        expect(rejected([&] { static_cast<void>(session.reserve_send_sequence()); }) &&
               rejected([&] { static_cast<void>(leases.acquire(physical, memory, 0)); }),
               "closing session cannot admit new traffic or a replacement lease");
        io.pending_release_polls = 0; session.close();
        expect(!session.active() && !leases.leased(physical.physical_link_id) && io.counts->freed == 1,
               "typed matching quiescence releases the host lease only after native pins retire");
    }
    {
        Io io; DirectTransportDriverBackend backend(io, device(), policy(1));
        auto physical = link(); auto memory = pools(); DirectTransportLeaseRegistry leases;
        DirectTransportLease retained;
        {
            auto session = DirectTransportNativeLinkSession::open(backend, leases, physical, memory, 0);
            retained = session.lease(); io.pending_release_polls = 100;
        }
        expect(leases.leased(physical.physical_link_id) && backend.retains_import() && io.counts->freed == 0,
               "session destruction cannot drop its exclusive lease while native import remains pending");
        io.pending_release_polls = 0;
        expect(backend.release_link_pool(retained) == DirectPoolReleaseStatus::released,
               "later owner cleanup obtains typed release proof for retained destructor lease");
        leases.release(retained);
        expect(!leases.leased(physical.physical_link_id) && io.counts->freed == 1,
               "explicit proven cleanup reclaims retained destructor ownership without leaking fixture resources");
    }
    {
        Io io; io.throw_activate = true; io.pending_release_polls = 100;
        DirectTransportDriverBackend backend(io, device(), policy(1));
        auto physical = link(); auto memory = pools(); DirectTransportLeaseRegistry leases;
        expect(rejected([&] { static_cast<void>(DirectTransportNativeLinkSession::open(backend, leases, physical, memory, 0)); }) &&
               leases.leased(physical.physical_link_id) && backend.retains_import() && io.counts->freed == 0,
               "partial native session open cannot release host lease on a void/error cleanup path");
        DirectTransportLease retained;
        { const auto imported = io.last_import.lock(); retained = imported->binding().lease; }
        io.pending_release_polls = 0;
        expect(backend.release_link_pool(retained) == DirectPoolReleaseStatus::released,
               "failed open cleanup requires exact import/lease quiescence acknowledgement");
        leases.release(retained);
        expect(io.counts->freed == 1 && !leases.leased(physical.physical_link_id),
               "failed open fixture is reclaimed only after proven release");
    }
    {
        Io io; io.pending_release_polls = 100;
        std::weak_ptr<DirectDriverImport> retained;
        {
            DirectTransportDriverBackend backend(io, device(), policy(1)); backend.import_link_pool(lease(), pools().links[0]);
            static_cast<void>(backend.prepare_receive(descriptor(), pools().links[0].rx)); retained = io.last_import;
            backend.release_link_pool(lease());
            expect(!retained.expired() && backend.retains_import() && backend.failed() &&
                   io.counts->freed == 0 && io.counts->released == 0,
                   "bounded release cannot destroy import/allocation pins while quiescence is unproven");
            io.pending_release_polls = 0;
            backend.release_link_pool(lease());
        }
        expect(retained.expired() && io.counts->freed == 1 && io.counts->released == 1,
               "matching later quiescence retires retained import before normal destruction");
    }
    if (failures == 0) std::cout << "direct transport driver tests passed\n";
    return failures == 0 ? 0 : 1;
}
