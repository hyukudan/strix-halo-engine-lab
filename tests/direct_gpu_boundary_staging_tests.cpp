#include "strix/direct_gpu_boundary_staging.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {
using namespace strix;
using namespace strix::transport;
void expect(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F&& f, const char* message) {
    bool caught = false; try { f(); } catch (const std::exception&) { caught = true; } expect(caught, message);
}
class Memory final : public DirectGpuBoundaryMemoryOwner {
public:
    std::array<std::byte, 20> first{};
    std::array<std::byte, 28> second{};
    std::vector<ExecutionBufferAddress> values;
    Memory() {
        values = {{{{ExecutionBufferClass::hidden, 0}, 1}, reinterpret_cast<std::uintptr_t>(first.data()), first.size(), 8},
                  {{{ExecutionBufferClass::expanded_hidden, 1}, 2}, reinterpret_cast<std::uintptr_t>(second.data()), second.size(), 8}};
        for (std::size_t n = 0; n < first.size(); ++n) first[n] = static_cast<std::byte>(n + 1);
        for (std::size_t n = 0; n < second.size(); ++n) second[n] = static_cast<std::byte>(n + 70);
    }
    std::vector<ExecutionBufferAddress> current_buffers() const override { return values; }
};
class Gpu final : public DirectGpuStreamGpu {
public:
    std::vector<std::byte> tx = std::vector<std::byte>(8192), rx = std::vector<std::byte>(8192);
    DirectGpuExportBinding exported;
    TransferDescriptor packed;
    std::vector<TransportBufferView> parts;
    unsigned stops = 0, copies = 0;
    bool bad = false;
    Gpu() {
        exported.pool.tx = {reinterpret_cast<std::uintptr_t>(tx.data()), tx.size(), 10, TransportMemoryDomain::gpu_transport_pool, TransportBufferAccess::read_write};
        exported.pool.rx = {reinterpret_cast<std::uintptr_t>(rx.data()), rx.size(), 11, TransportMemoryDomain::gpu_transport_pool, TransportBufferAccess::read_write};
        exported.pool_binding_identity = 22;
    }
    const DirectGpuExportBinding& exported_binding() const noexcept override { return exported; }
    void require_current() override { if (bad) throw std::runtime_error("fixture lost GPU binding"); }
    std::shared_ptr<DirectGpuExportImportLease> import_lease() const noexcept override { return {}; }
    void bind_import(const DirectDriverImportBinding&) override {}
    void stage_outbound_segments(const TransferDescriptor& descriptor, std::span<const TransportBufferView> source, const TransportBufferView& dest) override {
        packed = descriptor; parts.assign(source.begin(), source.end()); std::size_t offset = 0;
        for (const auto& part : source) { std::memcpy(reinterpret_cast<void*>(dest.address + offset), reinterpret_cast<const void*>(part.address), static_cast<std::size_t>(part.bytes)); offset += static_cast<std::size_t>(part.bytes); }
        ++copies;
    }
    void prepare(const DirectDriverOperation&) override {}
    DirectDriverVisibilityProof publish_terminal(const DirectDriverOperation&, const TransportBufferView&, const GpuCommitRecord&) override { throw std::logic_error("unused"); }
    DirectDriverCommitRead read_terminal(const DirectDriverOperation&, const TransportBufferView&) override { throw std::logic_error("unused"); }
    TransportBufferView acquired_receive_view(const DirectDriverOperation&) override { throw std::logic_error("unused"); }
    void consume_received_segments(const DirectDriverOperation&, std::span<const TransportBufferView>) override { throw std::logic_error("unused"); }
    DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation&) override { throw std::logic_error("unused"); }
    void retire(const DirectDriverOperation&) override {}
    void request_stop() noexcept override { ++stops; }
    DirectDriverQuiescenceStatus poll_quiescence() noexcept override { return DirectDriverQuiescenceStatus::quiescent; }
    DirectPoolReleaseStatus close() noexcept override { return DirectPoolReleaseStatus::released; }
};
class Receive final : public DirectGpuBoundaryReceiveIo {
public:
    std::array<std::byte, 48> acquired{};
    DirectDriverOperation operation;
    unsigned unpacks = 0, completions = 0;
    bool corrupt_proof = false, fail_copy = false;
    std::function<void()> during_completion;
    void require_current() override {}
    TransportBufferView acquired_receive_view(const TransferDescriptor&) override {
        return {reinterpret_cast<std::uintptr_t>(acquired.data()), acquired.size(), 4, TransportMemoryDomain::gpu_device, TransportBufferAccess::read_write};
    }
    void consume_received_segments(const TransferDescriptor& descriptor, std::span<const TransportBufferView> destinations) override {
        expect(descriptor.sequence == operation.descriptor.sequence, "unpack uses exact physical sequence");
        std::size_t offset = 0;
        for (const auto& part : destinations) { std::memcpy(reinterpret_cast<void*>(part.address), acquired.data() + offset, static_cast<std::size_t>(part.bytes)); offset += static_cast<std::size_t>(part.bytes); }
        ++unpacks; if (fail_copy) throw std::runtime_error("fixture failure after queued copy");
    }
    DirectDriverVisibilityProof complete_receive_consumer(const TransferDescriptor& descriptor) override {
        expect(descriptor.sequence == operation.descriptor.sequence, "completion uses exact physical sequence");
        ++completions; if (during_completion) during_completion();
        auto proof = DirectDriverVisibilityProof{operation, DirectDriverVisibility::consumer_release_complete};
        if (corrupt_proof) ++proof.operation.imported.lease.lease_generation;
        return proof;
    }
};
struct Fixture {
    std::shared_ptr<Memory> memory = std::make_shared<Memory>();
    std::shared_ptr<Gpu> gpu = std::make_shared<Gpu>();
    std::shared_ptr<Receive> receive = std::make_shared<Receive>();
    DistributedExecutionBoundary boundary{7, 1, 2, 0, 1, 48, {memory->values[0].use, memory->values[1].use}};
    DistributedPlacementPlan placement;
    DistributedRuntimeRequest request{33, 9, 12, 1};
    DirectTransportLease lease{"fixture-physical", 55, 22, 0, 2};
    Fixture(bool inbound = false) {
        if (inbound) { boundary.source_node = 1; boundary.destination_node = 0; }
        placement.local_node = 0; placement.executable = true; placement.plan_identity = 6; placement.boundaries = {boundary};
        placement.compiled_execution_identity = 61; placement.manifest_layout_identity = 62; placement.tensor_access_identity = 63;
    }
    std::unique_ptr<DirectGpuBoundaryStaging> make() { return std::make_unique<DirectGpuBoundaryStaging>(placement, request, lease, memory, gpu, receive); }
    DistributedBoundaryTransfer transfer(std::uint64_t sequence = 1) {
        DistributedBoundaryTransfer t; t.flow = classify_distributed_boundary_flow(placement, boundary);
        t.descriptor = make_distributed_boundary_descriptor(placement, boundary, request.connection_epoch, request.request_id, request.generation, sequence, request.dtype_id);
        t.link_plan.kind = PlanKind::single_link; t.link_plan.slices = {{0, 0, boundary.payload_bytes}}; t.native_slices = {{lease.connection_epoch, 100 + sequence}};
        return t;
    }
    void bind_receive(const DistributedBoundaryTransfer& t, const TransportBufferView& view) {
        auto& op = receive->operation; op.handle = 8; op.receive_path = true; op.buffer = view;
        op.descriptor = t.descriptor; op.descriptor.connection_epoch = lease.connection_epoch; op.descriptor.sequence = t.native_slices[0].sequence;
        op.imported.lease = lease; op.imported.pool = gpu->exported.pool; op.imported.import_generation = 3;
        op.imported.device.device_identity = "synthetic-only"; op.imported.device.device_instance_generation = 2;
        for (std::size_t n = 0; n < receive->acquired.size(); ++n) receive->acquired[n] = static_cast<std::byte>(n + 120);
    }
};
void successful_paths() {
    Fixture outbound; auto sender = outbound.make(); const auto t = outbound.transfer();
    const auto view = sender->stage_outbound(outbound.boundary, t);
    expect(sender->requires_full_send_retirement(), "fixed staging opts into full TX retirement");
    expect(view.bytes == 48 && outbound.gpu->packed.connection_epoch == 55 && outbound.gpu->packed.sequence == 101, "logical request maps to exact physical descriptor");
    expect(std::memcmp(outbound.gpu->tx.data(), outbound.memory->first.data(), 20) == 0 && std::memcmp(outbound.gpu->tx.data() + 20, outbound.memory->second.data(), 28) == 0, "ordered noncontiguous execution payloads are packed exactly");
    rejects([&] { static_cast<void>(sender->stage_outbound(outbound.boundary, t)); }, "logical TX replay rejected");
    Fixture inbound(true); auto consumer = inbound.make(); const auto rt = inbound.transfer();
    const auto staged = consumer->stage_inbound(inbound.boundary, rt); inbound.bind_receive(rt, staged);
    consumer->consume_inbound(inbound.boundary, staged); auto proof = consumer->finish_inbound(inbound.boundary, rt);
    expect(inbound.receive->completions == 0, "CPU consume return does not issue completion");
    std::weak_ptr<Memory> retained = inbound.memory; consumer.reset(); inbound.memory.reset();
    expect(!retained.expired(), "completion retains arena owner after staging object destruction");
    expect(proof->poll() == DistributedInboundCompletionStatus::complete && proof->poll() == DistributedInboundCompletionStatus::complete && inbound.receive->completions == 1, "actual typed GPU completion is cached and idempotent");
    const auto mem = retained.lock(); expect(std::memcmp(mem->first.data(), inbound.receive->acquired.data(), 20) == 0 && std::memcmp(mem->second.data(), inbound.receive->acquired.data() + 20, 28) == 0, "acquired GPU payload is unpacked to exact execution buffers");
}
void failures() {
    for (unsigned n = 0; n < 7; ++n) {
        Fixture f; auto adapter = f.make(); auto t = f.transfer();
        if (n == 0) ++t.descriptor.generation;
        if (n == 1) ++t.native_slices[0].connection_epoch;
        if (n == 2) t.link_plan.kind = PlanKind::striped;
        if (n == 3) ++t.link_plan.slices[0].byte_offset;
        if (n == 4) ++t.flow.flow_id;
        if (n == 5) ++f.memory->values[0].allocation_generation;
        if (n == 6) { t.native_slices.clear(); }
        rejects([&] { static_cast<void>(adapter->stage_outbound(f.boundary, t)); }, "foreign metadata rejected before staging copy");
        expect(adapter->failed() && f.gpu->copies == 0 && f.gpu->stops == 1, "metadata failure is sticky and queues no payload");
    }
    for (unsigned n = 0; n < 3; ++n) {
        Fixture f(true); auto adapter = f.make(); const auto t = f.transfer(); const auto staged = adapter->stage_inbound(f.boundary, t); f.bind_receive(t, staged);
        if (n == 0) f.receive->corrupt_proof = true;
        if (n == 1) f.receive->during_completion = [&] { ++f.memory->values[0].allocation_generation; };
        if (n == 2) f.receive->fail_copy = true;
        if (n == 2) { rejects([&] { adapter->consume_inbound(f.boundary, staged); }, "late copy failure propagates"); }
        else {
            adapter->consume_inbound(f.boundary, staged); const auto proof = adapter->finish_inbound(f.boundary, t);
            expect(proof->poll() == DistributedInboundCompletionStatus::failed && proof->poll() == DistributedInboundCompletionStatus::failed, "foreign proof or completion-time owner drift cannot authorize reuse");
        }
        expect(adapter->failed() && f.gpu->stops > 0, "late consumer failure poisons staging and retains completion owners");
    }
    Fixture bad;
    bad.memory->values[1].use = bad.memory->values[0].use;
    rejects([&] { static_cast<void>(bad.make()); }, "duplicate symbolic buffer identities rejected");
    Fixture short_view; --short_view.memory->values[1].bytes;
    rejects([&] { static_cast<void>(short_view.make()); }, "payload byte cover must be exact");
    Fixture aliases; aliases.memory->values[1].address = aliases.memory->values[0].address;
    rejects([&] { static_cast<void>(aliases.make()); }, "simultaneous boundary payloads cannot overlap");
    Fixture premature(true); auto a = premature.make(); const auto t = premature.transfer(); static_cast<void>(a->stage_inbound(premature.boundary, t));
    rejects([&] { static_cast<void>(a->finish_inbound(premature.boundary, t)); }, "finish cannot substitute for GPU acquire and consumption");
}
} // namespace
int main() {
    try { successful_paths(); failures(); std::cout << "GPU boundary staging contract tests passed (synthetic GPU only).\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
