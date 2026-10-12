#include "strix/direct_transport_driver.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace strix::transport {
namespace {
[[noreturn]] void unavailable() { throw std::logic_error("supported native driver ABI is unavailable"); }

bool same_buffer(const TransportBufferView& a, const TransportBufferView& b) noexcept {
    return a.address == b.address && a.bytes == b.bytes &&
           a.allocation_generation == b.allocation_generation && a.domain == b.domain && a.access == b.access;
}
bool same_lease(const DirectTransportLease& a, const DirectTransportLease& b) noexcept {
    return a.physical_link_id == b.physical_link_id && a.connection_epoch == b.connection_epoch &&
           a.pool_binding_identity == b.pool_binding_identity && a.link_index == b.link_index &&
           a.lease_generation == b.lease_generation;
}
bool same_import(const DirectDriverImportBinding& a, const DirectDriverImportBinding& b) noexcept {
    return a.device == b.device && same_lease(a.lease, b.lease) && a.import_generation == b.import_generation &&
           same_buffer(a.pool.tx, b.pool.tx) && same_buffer(a.pool.rx, b.pool.rx);
}
bool same_descriptor(const TransferDescriptor& a, const TransferDescriptor& b) noexcept {
    return a.protocol_version == b.protocol_version && a.connection_epoch == b.connection_epoch &&
           a.model_plan_id == b.model_plan_id && a.request_id == b.request_id && a.generation == b.generation &&
           a.boundary_id == b.boundary_id && a.sequence == b.sequence &&
           a.shape_fingerprint == b.shape_fingerprint && a.dtype_id == b.dtype_id && a.byte_length == b.byte_length;
}
bool same_operation(const DirectDriverOperation& a, const DirectDriverOperation& b) noexcept {
    return same_import(a.imported, b.imported) && a.handle == b.handle &&
           same_descriptor(a.descriptor, b.descriptor) && same_buffer(a.buffer, b.buffer) &&
           a.receive_path == b.receive_path;
}
bool bounded_pool(const TransportBufferView& pool, bool receive) noexcept {
    return pool.domain == TransportMemoryDomain::gpu_transport_pool &&
           pool.bytes <= std::numeric_limits<std::uintptr_t>::max() - pool.address &&
           (receive ? validate_transport_receive_buffer(pool, 1) : validate_transport_send_buffer(pool, 1));
}
bool within(const TransportBufferView& buffer, const TransportBufferView& pool, std::uint64_t bytes,
            bool receive) noexcept {
    if (!bounded_pool(pool, receive) || buffer.address < pool.address ||
        buffer.domain != pool.domain || buffer.allocation_generation != pool.allocation_generation) return false;
    const auto offset = static_cast<std::uint64_t>(buffer.address - pool.address);
    return offset <= pool.bytes && buffer.bytes <= pool.bytes - offset &&
           (receive ? validate_transport_receive_buffer(buffer, bytes) : validate_transport_send_buffer(buffer, bytes));
}
bool overlap(const TransportBufferView& a, const TransportBufferView& b) noexcept {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
TransferHandle next_handle() {
    static std::atomic<TransferHandle> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;) {
        if (value == std::numeric_limits<TransferHandle>::max()) throw std::overflow_error("driver operation ids exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
    }
}
}  // namespace

DirectDriverQuery DirectTransportDriverIo::query() { return {}; }
std::optional<DirectDriverBinding> DirectTransportDriverIo::current_binding() const { return std::nullopt; }
std::shared_ptr<DirectDriverImport> DirectTransportDriverIo::prepare_import(
    const DirectDriverBinding&, const DirectTransportLease&, const DirectTransportLinkPoolBinding&) { unavailable(); }
void DirectTransportDriverIo::activate_import(const std::shared_ptr<DirectDriverImport>&) { unavailable(); }
void DirectTransportDriverIo::prepare(const DirectDriverOperation&) { unavailable(); }
void DirectTransportDriverIo::submit(const DirectDriverOperation&) { unavailable(); }
void DirectTransportDriverIo::cancel(const DirectDriverOperation&) { unavailable(); }
DirectDriverVisibilityProof DirectTransportDriverIo::publish_gpu_commit(const DirectDriverOperation&, const GpuCommitRecord&) { unavailable(); }
DirectDriverCommitRead DirectTransportDriverIo::read_gpu_commit(const DirectDriverOperation&) { unavailable(); }
DirectDriverVisibilityProof DirectTransportDriverIo::consumer_completion(const DirectDriverOperation&) { unavailable(); }
void DirectTransportDriverIo::publish_consumer_credit(const DirectDriverOperation&, std::uint64_t) { unavailable(); }
void DirectTransportDriverIo::repost_receive(const DirectDriverOperation&) { unavailable(); }
std::optional<DirectDriverEvent> DirectTransportDriverIo::poll_event() { unavailable(); }
void DirectTransportDriverIo::retire(const DirectDriverOperation&) { unavailable(); }
void DirectTransportDriverIo::request_import_release(const std::shared_ptr<DirectDriverImport>&) { unavailable(); }
DirectDriverQuiescence DirectTransportDriverIo::poll_import_quiescence(const std::shared_ptr<DirectDriverImport>&) { unavailable(); }
void DirectTransportDriverIo::release_import(const std::shared_ptr<DirectDriverImport>&) { unavailable(); }

struct DirectTransportDriverBackend::Impl {
    struct Operation {
        DirectDriverOperation binding;
        bool submitted = false;
        bool canceled = false;
        bool producer_released = false;
        bool received = false;
        bool acquired = false;
        bool credited = false;
        bool reposted = false;
        unsigned send_completions = 0;
    };
    DirectTransportDriverIo* io;
    DirectDriverBinding expected;
    DirectDriverPolicy policy;
    DirectDriverQuery query;
    std::shared_ptr<DirectDriverImport> imported;
    DirectDriverImportBinding import_binding;
    std::unordered_map<TransferHandle, Operation> operations;
    bool ready = false;
    bool failed = false;
    bool closing = false;
    bool release_requested = false;
    std::uint64_t last_send_sequence = 0;
    std::uint64_t last_receive_sequence = 0;

    bool current() const noexcept {
        try { return io->current_binding() == std::optional<DirectDriverBinding>(expected); }
        catch (...) { return false; }
    }
    [[noreturn]] void reject(const char* reason) { failed = true; throw std::runtime_error(reason); }
    void require_current() {
        if (failed || !ready || closing) throw std::logic_error("native driver adapter is unavailable, failed or closing");
        if (!current()) reject("native driver device/ABI/peer/epoch binding changed");
        if (imported && !same_import(imported->binding(), import_binding)) reject("native driver import binding changed");
    }
    Operation& require(TransferHandle handle) {
        require_current();
        if (!imported) throw std::logic_error("native driver pool is not imported");
        const auto found = operations.find(handle);
        if (found == operations.end()) throw std::invalid_argument("native driver operation is stale or unknown");
        return found->second;
    }
    void proof(const DirectDriverVisibilityProof& proof, const Operation& operation, DirectDriverVisibility visibility) {
        require_current();
        if (proof.visibility != visibility || !same_operation(proof.operation, operation.binding)) {
            reject("native driver visibility/completion proof does not match operation");
        }
    }
};

DirectTransportDriverBackend::DirectTransportDriverBackend(DirectTransportDriverIo& io,
    DirectDriverBinding expected, DirectDriverPolicy policy) : impl_(std::make_unique<Impl>()) {
    if (policy.maximum_payload_bytes == 0 || policy.maximum_payload_bytes > std::numeric_limits<std::uint32_t>::max() ||
        policy.maximum_imported_bytes == 0 || policy.maximum_operations == 0 || policy.release_polls == 0) {
        throw std::invalid_argument("native driver adapter requires bounded payloads, imports, operations and progress");
    }
    impl_->io = &io;
    impl_->expected = std::move(expected);
    impl_->policy = policy;
    try { impl_->query = io.query(); }
    catch (...) { impl_->query.native.query_status = DirectNativeQueryStatus::inaccessible; }
    auto& query = impl_->query;
    if (query.native.query_status != DirectNativeQueryStatus::verified) return;
    DirectLinkLifecycle link;
    link.physical_link_id = impl_->expected.interface.physical_link_id;
    link.connection_epoch = impl_->expected.interface.connection_epoch;
    link.expected_native_interface = impl_->expected.interface;
    impl_->ready = impl_->expected.adapter_contract_version == 1 && !impl_->expected.device_identity.empty() &&
        impl_->expected.device_instance_generation != 0 && query.binding == impl_->expected &&
        query.backend.maximum_payload_bytes != 0 && impl_->current() &&
        assess_direct_native_readiness(query.backend, query.native, link).ready;
    if (!impl_->ready) query.native.query_status = DirectNativeQueryStatus::incompatible;
    query.backend.maximum_payload_bytes = std::min(query.backend.maximum_payload_bytes, policy.maximum_payload_bytes);
}

DirectTransportDriverBackend::~DirectTransportDriverBackend() {
    if (impl_->imported) release_link_pool(impl_->import_binding.lease);
    if (impl_->imported) {
        // Keep the owning native token and bounded operation metadata alive.
        // Neither cancellation nor adapter destruction proves DMA quiescence.
        // Failed runtimes must stop; creating replacement adapters could grow
        // process-lifetime quarantine and is not a supported recovery path.
        static_cast<void>(impl_.release());
    }
}
BackendCapabilities DirectTransportDriverBackend::capabilities() const { return impl_->query.backend; }
DirectNativeCapabilities DirectTransportDriverBackend::direct_capabilities() const {
    auto result = impl_->query.native;
    if (impl_->failed || impl_->closing || (impl_->ready && !impl_->current())) result.query_status = DirectNativeQueryStatus::incompatible;
    return result;
}
bool DirectTransportDriverBackend::failed() const noexcept { return impl_->failed; }
bool DirectTransportDriverBackend::retains_import() const noexcept { return static_cast<bool>(impl_->imported); }
std::uint32_t DirectTransportDriverBackend::active_operations() const noexcept {
    return static_cast<std::uint32_t>(impl_->operations.size());
}

void DirectTransportDriverBackend::import_link_pool(const DirectTransportLease& lease,
    const DirectTransportLinkPoolBinding& pool) {
    impl_->require_current();
    if (impl_->imported) throw std::logic_error("native driver import already owns this physical connection");
    if (lease.physical_link_id != impl_->expected.interface.physical_link_id ||
        lease.connection_epoch != impl_->expected.interface.connection_epoch || lease.lease_generation == 0 ||
        lease.pool_binding_identity == 0 || !bounded_pool(pool.tx, false) || !bounded_pool(pool.rx, true) ||
        pool.tx.allocation_generation == pool.rx.allocation_generation || overlap(pool.tx, pool.rx) ||
        pool.tx.bytes > impl_->policy.maximum_imported_bytes ||
        pool.rx.bytes > impl_->policy.maximum_imported_bytes - pool.tx.bytes) {
        throw std::invalid_argument("native driver import does not match exact independent bounded TX/RX lease");
    }
    try {
        auto imported = impl_->io->prepare_import(impl_->expected, lease, pool);
        if (!imported) impl_->reject("native driver prepare returned no owning import token");
        const auto binding = imported->binding();
        if (binding.device != impl_->expected || !same_lease(binding.lease, lease) || binding.import_generation == 0 ||
            !same_buffer(binding.pool.tx, pool.tx) || !same_buffer(binding.pool.rx, pool.rx)) {
            impl_->reject("native driver prepared a foreign import token");
        }
        impl_->imported = std::move(imported);
        impl_->import_binding = binding;
        impl_->require_current();
        impl_->io->activate_import(impl_->imported);
        impl_->require_current();
    } catch (...) {
        impl_->failed = true;
        if (impl_->imported) release_link_pool(lease);
        throw;
    }
}

DirectPoolReleaseStatus DirectTransportDriverBackend::release_link_pool(const DirectTransportLease& lease) noexcept {
    if (!impl_->imported) {
        if (impl_->import_binding.import_generation == 0 || same_lease(lease, impl_->import_binding.lease)) {
            return DirectPoolReleaseStatus::released;
        }
        impl_->failed = true;
        return DirectPoolReleaseStatus::failed_retained;
    }
    if (!same_lease(lease, impl_->import_binding.lease)) {
        impl_->failed = true;
        return DirectPoolReleaseStatus::failed_retained;
    }
    impl_->closing = true;
    try {
        const auto intact = [&] {
            if (!same_import(impl_->imported->binding(), impl_->import_binding)) {
                throw std::runtime_error("native driver cleanup import token changed");
            }
        };
        intact();
        if (!impl_->release_requested) {
            impl_->release_requested = true;
            impl_->io->request_import_release(impl_->imported);
            intact();
        }
        for (std::uint64_t count = 0; count < impl_->policy.release_polls; ++count) {
            const auto proof = impl_->io->poll_import_quiescence(impl_->imported);
            intact();
            if (!same_import(proof.imported, impl_->import_binding)) {
                impl_->failed = true;
                return DirectPoolReleaseStatus::failed_retained;
            }
            if (proof.status == DirectDriverQuiescenceStatus::quiescent) {
                impl_->io->release_import(impl_->imported);
                intact();
                impl_->imported.reset();
                impl_->operations.clear();
                impl_->closing = false;
                impl_->release_requested = false;
                return DirectPoolReleaseStatus::released;
            }
            if (proof.status != DirectDriverQuiescenceStatus::pending) {
                impl_->failed = true;
                return DirectPoolReleaseStatus::failed_retained;
            }
        }
    } catch (...) { impl_->failed = true; }
    impl_->failed = true;
    return DirectPoolReleaseStatus::failed_retained;
}

TransferHandle DirectTransportDriverBackend::prepare(const TransferDescriptor& descriptor,
    const TransportBufferView& buffer, bool receive) {
    impl_->require_current();
    if (!impl_->imported) throw std::logic_error("native driver preparation requires an owning import");
    if (impl_->operations.size() >= impl_->policy.maximum_operations) throw std::runtime_error("native driver operation budget exhausted");
    if (!validate_descriptor(descriptor, {1, impl_->expected.interface.connection_epoch,
                                         impl_->query.backend.maximum_payload_bytes}) || descriptor.sequence == 0 ||
        !within(buffer, receive ? impl_->import_binding.pool.rx : impl_->import_binding.pool.tx,
                descriptor.byte_length, receive)) throw std::invalid_argument("native driver descriptor/buffer is foreign or oversized");
    for (const auto& [handle, pending] : impl_->operations) {
        static_cast<void>(handle);
        if (overlap(buffer, pending.binding.buffer)) throw std::logic_error("native driver buffer is still owned by another operation");
    }
    auto& last_sequence = receive ? impl_->last_receive_sequence : impl_->last_send_sequence;
    if (descriptor.sequence <= last_sequence) throw std::invalid_argument("native driver sequence is stale or replayed");
    // Burn before preparation: a callback error cannot make an exposed identity reusable.
    last_sequence = descriptor.sequence;
    Impl::Operation operation;
    operation.binding = {impl_->import_binding, next_handle(), descriptor, buffer, receive};
    const auto handle = operation.binding.handle;
    auto [entry, inserted] = impl_->operations.emplace(handle, std::move(operation));
    if (!inserted) impl_->reject("native driver operation id collision");
    try { impl_->io->prepare(entry->second.binding); impl_->require_current(); }
    catch (...) { impl_->failed = true; throw; }
    return handle;
}
TransferHandle DirectTransportDriverBackend::prepare_send(const TransferDescriptor& descriptor, const TransportBufferView& buffer) {
    return prepare(descriptor, buffer, false);
}
TransferHandle DirectTransportDriverBackend::prepare_receive(const TransferDescriptor& descriptor, const TransportBufferView& buffer) {
    return prepare(descriptor, buffer, true);
}
void DirectTransportDriverBackend::submit(TransferHandle handle) {
    auto& operation = impl_->require(handle);
    if (operation.submitted || operation.canceled || (!operation.binding.receive_path && !operation.producer_released)) {
        throw std::logic_error("native driver submit requires one prepared, released operation");
    }
    operation.submitted = true;
    try { impl_->io->submit(operation.binding); impl_->require_current(); }
    catch (...) { impl_->failed = true; throw; }
}
void DirectTransportDriverBackend::cancel(TransferHandle handle) {
    auto& operation = impl_->require(handle);
    if (operation.canceled) return;
    operation.canceled = true;
    try { impl_->io->cancel(operation.binding); impl_->require_current(); }
    catch (...) { impl_->failed = true; throw; }
}
void DirectTransportDriverBackend::publish_gpu_commit(TransferHandle handle, const GpuCommitRecord& record) {
    auto& operation = impl_->require(handle);
    if (operation.binding.receive_path || operation.submitted || operation.canceled || operation.producer_released ||
        !validate_gpu_commit_record(record, operation.binding.descriptor) || record.payload_bytes != operation.binding.descriptor.byte_length) {
        throw std::invalid_argument("native driver commit requires exact unreleased prepared TX");
    }
    try {
        impl_->proof(impl_->io->publish_gpu_commit(operation.binding, record), operation,
                     DirectDriverVisibility::producer_release_to_external_dma);
        operation.producer_released = true;
    } catch (...) { impl_->failed = true; throw; }
}
GpuCommitRecord DirectTransportDriverBackend::read_gpu_commit(TransferHandle handle) {
    auto& operation = impl_->require(handle);
    if (!operation.binding.receive_path || !operation.received || operation.canceled || operation.acquired) {
        throw std::logic_error("native driver acquire requires one completed RX operation");
    }
    try {
        const auto result = impl_->io->read_gpu_commit(operation.binding);
        impl_->proof(result.acquire, operation, DirectDriverVisibility::consumer_acquire_from_external_dma);
        if (!validate_gpu_commit_record(result.record, operation.binding.descriptor) ||
            result.record.payload_bytes != operation.binding.descriptor.byte_length) impl_->reject("native driver RX commit is invalid");
        operation.acquired = true;
        impl_->require_current();
        return result.record;
    } catch (...) { impl_->failed = true; throw; }
}
void DirectTransportDriverBackend::publish_consumer_credit(TransferHandle handle, std::uint64_t bytes) {
    auto& operation = impl_->require(handle);
    if (!operation.binding.receive_path || !operation.acquired || operation.canceled || operation.credited ||
        bytes != operation.binding.descriptor.byte_length) throw std::logic_error("native driver credit requires exact acquired, uncredited RX");
    try {
        impl_->proof(impl_->io->consumer_completion(operation.binding), operation, DirectDriverVisibility::consumer_release_complete);
        // Mark before the side effect: an exception cannot authorize retry.
        operation.credited = true;
        impl_->io->publish_consumer_credit(operation.binding, bytes);
        impl_->require_current();
    } catch (...) { impl_->failed = true; throw; }
}
void DirectTransportDriverBackend::repost_receive(TransferHandle handle) {
    auto& operation = impl_->require(handle);
    if (!operation.binding.receive_path || !operation.credited || operation.canceled || operation.reposted) {
        throw std::logic_error("native driver repost requires one completed consumer credit");
    }
    operation.reposted = true;
    try { impl_->io->repost_receive(operation.binding); impl_->require_current(); }
    catch (...) { impl_->failed = true; throw; }
}

std::optional<DirectNativeEvent> DirectTransportDriverBackend::poll_direct_event() {
    impl_->require_current();
    if (!impl_->imported) throw std::logic_error("native driver event polling requires an owning import");
    try {
        const auto result = impl_->io->poll_event();
        impl_->require_current();
        if (!result) return std::nullopt;
        const auto found = impl_->operations.find(result->event.completion.handle);
        if (found == impl_->operations.end()) impl_->reject("native driver event references stale/unknown operation");
        auto& operation = found->second;
        const auto& event = result->event;
        if (!same_operation(result->operation, operation.binding) || event.receive_path != operation.binding.receive_path ||
            !operation.submitted || operation.canceled || !std::isfinite(event.monotonic_seconds) || event.monotonic_seconds < 0.0 ||
            event.in_flight_bytes > impl_->policy.maximum_imported_bytes ||
            (event.completion.kind != CompletionKind::failed && event.completion.bytes != operation.binding.descriptor.byte_length)) {
            impl_->reject("native driver event binding, direction or byte count is invalid");
        }
        bool terminal = false;
        if (event.completion.kind == CompletionKind::failed) {
            impl_->failed = true;
            return event;
        }
        if (operation.binding.receive_path) {
            if (event.completion.kind == CompletionKind::receive_complete && !operation.received) operation.received = true;
            else if (event.completion.kind == CompletionKind::receive_reposted && operation.reposted) terminal = true;
            else impl_->reject("native driver RX completion is duplicate or violates ownership");
        } else {
            unsigned bit = 0;
            if (event.completion.kind == CompletionKind::local_buffer_reusable) bit = 1;
            if (event.completion.kind == CompletionKind::remote_transport_ack) bit = 2;
            if (event.completion.kind == CompletionKind::remote_consumer_complete) {
                if ((operation.send_completions & 2U) == 0) impl_->reject("native driver consumer completion precedes transport ACK");
                bit = 4;
            }
            if (bit == 0 || (operation.send_completions & bit) != 0) impl_->reject("native driver TX completion is duplicate or has wrong direction");
            operation.send_completions |= bit;
            terminal = operation.send_completions == 7;
        }
        if (terminal) {
            impl_->io->retire(operation.binding);
            impl_->require_current();
            impl_->operations.erase(found);
        }
        return event;
    } catch (...) { impl_->failed = true; throw; }
}

}  // namespace strix::transport
