#include "strix/hip_direct_gpu_stream_gpu.hpp"
#include "direct_gpu_stream_fixture.hpp"

#include <iostream>
#include <map>

namespace {
using namespace direct_gpu_stream_fixture;
int failures = 0;
void expect(bool condition, const char* message) { if (!condition) { std::cerr << message << '\n'; ++failures; } }
template<class F> bool rejected(F function) { try { function(); } catch (const std::exception&) { return true; } return false; }
struct Fixture {
    struct Copy { void* destination; const void* source; std::size_t bytes; hipMemcpyKind kind; };
    Pool pool;
    std::vector<TransportBufferView> execution;
    std::vector<Copy> pending, history;
    std::map<hipEvent_t, bool> recorded;
    std::vector<unsigned> event_flags;
    std::vector<std::unique_ptr<std::byte[]>> hosts;
    int host_freed = 0, events_freed = 0, stream_freed = 0, records = 0, queries = 0, delay = 2;
    std::uintptr_t next_event = 0x5000;
    unsigned long long stream_id = 9;
    bool fail_copy_after_queue = false, fail_record = false, mutate_generation = false;
    HipDirectGpuStreamGpuCalls calls = HipDirectGpuStreamGpuCalls::native();
    Fixture() {
        execution = {pool.external(16384), pool.external(16384), pool.external(16384)};
        auto& observation = calls.observations;
        observation.get_device = [&](int* value) { *value = pool.state->device.index; return hipSuccess; };
        observation.device_uuid = [&](hipUUID* value, hipDevice_t) { *value = {}; value->bytes[0] = 71; return hipSuccess; };
        observation.runtime_version = [](int* value) { *value = 720; return hipSuccess; };
        observation.driver_version = [](int* value) { *value = 720; return hipSuccess; };
        observation.pointer_attributes = [](hipPointerAttribute_t* value, const void*) { *value = {}; value->device = 3; value->type = hipMemoryTypeDevice; return hipSuccess; };
        observation.address_range = [&](hipDeviceptr_t* base, std::size_t* bytes, hipDeviceptr_t address) {
            const auto [native, block] = pool.state->containing(reinterpret_cast<std::uintptr_t>(address), 1); *base = reinterpret_cast<hipDeviceptr_t>(native); *bytes = block->size(); return hipSuccess;
        };
        observation.get_fd_flags = [&](int fd) { return pool.state->fds.contains(fd) ? 1 : -1; };
        observation.fd_identity = [&](int fd) { return pool.state->fds.at(fd); };
        calls.create_stream = [](hipStream_t* value, unsigned flags) { expect(flags == hipStreamNonBlocking, "native stream request is explicit nonblocking stream"); *value = reinterpret_cast<hipStream_t>(0x3330); return hipSuccess; };
        calls.stream_device = [](hipStream_t, hipDevice_t* value) { *value = 3; return hipSuccess; };
        calls.stream_id = [&](hipStream_t, unsigned long long* value) { *value = stream_id; return hipSuccess; };
        calls.stream_capture = [](hipStream_t, hipStreamCaptureStatus* value) { *value = hipStreamCaptureStatusNone; return hipSuccess; };
        calls.query_stream = [&](hipStream_t) { return pending.empty() ? hipSuccess : hipErrorNotReady; };
        calls.destroy_stream = [&](hipStream_t) { ++stream_freed; return hipSuccess; };
        calls.allocate_host = [&](void** value, std::size_t bytes, unsigned flags) {
            expect(bytes == sizeof(GpuCommitRecord) && flags == hipHostMallocDefault, "fixed pinned metadata source/destination requests are bounded");
            hosts.push_back(std::make_unique<std::byte[]>(bytes)); *value = hosts.back().get(); return hipSuccess;
        };
        calls.free_host = [&](void* value) { for (auto& host : hosts) if (host.get() == value) { host.reset(); ++host_freed; return hipSuccess; } return hipErrorInvalidValue; };
        calls.create_event = [&](hipEvent_t* value, unsigned flags) { event_flags.push_back(flags); *value = reinterpret_cast<hipEvent_t>(next_event); next_event += 0x100; recorded[*value] = false; return hipSuccess; };
        calls.record_event = [&](hipEvent_t event, hipStream_t stream) {
            expect(stream == reinterpret_cast<hipStream_t>(0x3330), "events record on the exact owned producer/consumer stream"); ++records;
            if (fail_record) return hipErrorUnknown;
            recorded.at(event) = true; return hipSuccess;
        };
        calls.query_event = [&](hipEvent_t event) {
            ++queries;
            if (!recorded.at(event)) return hipSuccess; // HIP unrecorded event is not queued-work proof.
            if (delay-- > 0) return hipErrorNotReady;
            flush(); return hipSuccess;
        };
        calls.destroy_event = [&](hipEvent_t event) { recorded.erase(event); ++events_freed; return hipSuccess; };
        calls.copy = [&](void* destination, const void* source, std::size_t bytes, hipMemcpyKind kind, hipStream_t) {
            pending.push_back({destination, source, bytes, kind}); history.push_back(pending.back());
            if (mutate_generation) { ++execution[0].allocation_generation; mutate_generation = false; }
            return fail_copy_after_queue ? hipErrorUnknown : hipSuccess;
        };
    }
    void flush() {
        for (const auto& copy : pending) {
            auto* destination = copy.kind == hipMemcpyDeviceToHost ? static_cast<std::byte*>(copy.destination) : pool.state->pointer(reinterpret_cast<std::uintptr_t>(copy.destination), copy.bytes);
            const auto* source = copy.kind == hipMemcpyHostToDevice ? static_cast<const std::byte*>(copy.source) : pool.state->pointer(reinterpret_cast<std::uintptr_t>(copy.source), copy.bytes);
            std::memcpy(destination, source, copy.bytes);
        }
        pending.clear();
    }
    DirectDriverImportBinding imported() const {
        DirectDriverImportBinding value; value.device = {1, "synthetic-device", 1, {1, "synthetic-kernel", "synthetic-driver", "synthetic-ABI", "a", "b", "synthetic-link", 77}};
        value.lease = {"synthetic-link", 77, pool.exported->binding().pool_binding_identity, 0, 1}; value.pool = pool.exported->binding().pool; value.import_generation = 1; return value;
    }
    TransferDescriptor descriptor(std::uint64_t sequence = 1) const { return {1, 77, 2, 3, 4, 5, sequence, 6, 1, 5120}; }
    DirectDriverOperation operation(bool receive, std::uint64_t sequence = 1) const {
        auto view = receive ? imported().pool.rx : imported().pool.tx; view.bytes = 5120; return {imported(), receive ? 21ULL : 11ULL, descriptor(sequence), view, receive};
    }
    TransportBufferView terminal(bool receive) const { auto view = receive ? imported().pool.rx : imported().pool.tx; view.address += view.bytes - 4096; view.bytes = 32; return view; }
    GpuCommitRecord commit(std::uint64_t sequence = 1) const { return {1, gpu_commit_word, 77, sequence, 5120, 0, 1}; }
    std::unique_ptr<HipDirectGpuStreamGpu> prepare(std::uint64_t polls = 8) {
        HipDirectGpuStreamExecutionOwner owner; owner.lifetime = pool.state; owner.buffers = execution; owner.receive_destination = execution[1]; owner.current_buffers = [&] { return execution; };
        auto gpu = std::make_unique<HipDirectGpuStreamGpu>(pool.lease, owner, HipDirectGpuStreamGpuPolicy{true, polls, 3 * 16384}, calls); gpu->bind_import(imported()); return gpu;
    }
    void stage(HipDirectGpuStreamGpu& gpu) {
        auto first = execution[0]; first.bytes = 1280; auto second = execution[0]; second.address += 1280; second.bytes = 3840;
        gpu.stage_outbound_segments(descriptor(), std::array{first, second}, operation(false).buffer);
    }
};
}
int main() {
    {
        Fixture fixture; auto gpu = fixture.prepare(); fixture.stage(*gpu); gpu->prepare(fixture.operation(false));
        const auto release = gpu->publish_terminal(fixture.operation(false), fixture.terminal(false), fixture.commit());
        expect(release.visibility == DirectDriverVisibility::producer_release_to_external_dma && fixture.queries >= 3 && fixture.pending.empty(), "producer proof waits for delayed actual copy/event completion");
        expect(fixture.history.size() == 3 && fixture.history[0].bytes == 1280 && fixture.history[1].bytes == 3840 && fixture.history[2].bytes == 32,
               "registered noncontiguous sources pack exact opaque payload bytes before terminal");
        const auto rx = fixture.operation(true); gpu->prepare(rx);
        std::memcpy(fixture.pool.state->pointer(fixture.terminal(true).address, 32), fixture.pool.state->pointer(fixture.terminal(false).address, 32), 32);
        auto* payload = fixture.pool.state->pointer(rx.buffer.address, 5120); std::fill(payload, payload + 5120, std::byte{0x5a});
        fixture.delay = 2; const auto acquired = gpu->read_terminal(rx, fixture.terminal(true));
        expect(acquired.acquire.visibility == DirectDriverVisibility::consumer_acquire_from_external_dma &&
               fixture.history[3].kind == hipMemcpyDeviceToDevice && fixture.history[3].bytes == 5120 &&
               fixture.history[3].source == reinterpret_cast<const void*>(rx.buffer.address) && fixture.history[3].destination == reinterpret_cast<void*>(fixture.execution[1].address),
               "acquire includes entire payload D2D into exact retained destination before proof");
        auto destination = fixture.execution[2]; destination.bytes = 5120; gpu->consume_received_segments(rx, std::array{destination});
        fixture.delay = 2; const auto consumer = gpu->consumer_completion(rx); const auto records = fixture.records;
        expect(consumer.visibility == DirectDriverVisibility::consumer_release_complete && fixture.pending.empty() &&
               fixture.pool.state->pointer(destination.address, 5120)[5119] == std::byte{0x5a}, "consumer proof includes delayed exact owned unpack destination copy");
        expect(gpu->consumer_completion(rx).visibility == DirectDriverVisibility::consumer_release_complete && fixture.records == records, "same completed consumer event can be read without duplicate record");
        expect(std::all_of(fixture.event_flags.begin(), fixture.event_flags.end(), [](unsigned flags) { return flags == hipEventDisableTiming; }), "system-fence-disable and device-release flags are never requested");
        gpu->retire(fixture.operation(false)); gpu->retire(rx);
        expect(gpu->close() == DirectPoolReleaseStatus::released && fixture.host_freed == 2 && fixture.events_freed == 6 && fixture.stream_freed == 1,
               "proved quiescence destroys bounded native metadata/events/owned stream");
    }
    {
        Fixture fixture; auto gpu = fixture.prepare(2); fixture.stage(*gpu); gpu->prepare(fixture.operation(false)); fixture.delay = 100;
        expect(rejected([&] { static_cast<void>(gpu->publish_terminal(fixture.operation(false), fixture.terminal(false), fixture.commit())); }), "delayed completion beyond budget fails closed");
        expect(gpu->failed() && gpu->poll_quiescence() == DirectDriverQuiescenceStatus::pending && gpu->close() == DirectPoolReleaseStatus::failed_retained && fixture.host_freed == 0,
               "timeout retains all pending terminal source/stream/execution allocations");
        expect(fixture.pool.exported->release() == DirectPoolReleaseStatus::pending, "pending GPU source retains export owner and allocator use lease");
        fixture.flush(); expect(gpu->close() == DirectPoolReleaseStatus::released && gpu->failed(), "later actual stream completion permits cleanup without healing failed proof");
    }
    {
        Fixture fixture; auto gpu = fixture.prepare(); fixture.stage(*gpu); gpu->prepare(fixture.operation(false)); fixture.fail_record = true;
        expect(rejected([&] { static_cast<void>(gpu->publish_terminal(fixture.operation(false), fixture.terminal(false), fixture.commit())); }) && fixture.queries == 0,
               "unrecorded event hipSuccess cannot authorize queued copy reuse after record error");
        expect(gpu->close() == DirectPoolReleaseStatus::failed_retained && fixture.host_freed == 0, "record failure retains pinned source until real stream drain");
        fixture.flush(); expect(gpu->close() == DirectPoolReleaseStatus::released, "explicit completed stream drains partial copy/record failure");
    }
    {
        Fixture fixture; auto gpu = fixture.prepare(); fixture.fail_copy_after_queue = true;
        expect(rejected([&] { fixture.stage(*gpu); }) && gpu->poll_quiescence() == DirectDriverQuiescenceStatus::pending,
               "copy callback error after enqueue retains source and destination ownership");
        fixture.flush(); expect(gpu->close() == DirectPoolReleaseStatus::released, "late source/destination completion cleans rejected copy without proof publication");
    }
    {
        Fixture fixture; auto gpu = fixture.prepare(); fixture.stage(*gpu); gpu->prepare(fixture.operation(false));
        const auto copies = fixture.history.size(); const auto records = fixture.records; const auto pool = fixture.pool.exported->binding().pool;
        fixture.pool.memory->free_device(pool.tx.address, pool.tx.bytes);
        expect(fixture.pool.state->freed == 0 && rejected([&] { static_cast<void>(gpu->publish_terminal(fixture.operation(false), fixture.terminal(false), fixture.commit())); }) &&
               fixture.history.size() == copies && fixture.records == records, "forbidden allocator free invalidates guard and prevents new GPU copy/event publication");
        fixture.flush(); expect(gpu->close() == DirectPoolReleaseStatus::released, "existing retained stream work still cleans through positive quiescence");
    }
    {
        Fixture fixture; auto gpu = fixture.prepare(); const auto before = fixture.execution; fixture.mutate_generation = true;
        expect(rejected([&] { fixture.stage(*gpu); }) && !fixture.pending.empty(), "source-generation drift after enqueue rejects completion authority");
        fixture.execution = before; fixture.flush(); expect(gpu->close() == DirectPoolReleaseStatus::released, "exact restored retained owner permits cleanup only");
    }
    if (!failures) std::cout << "injected HIP GPU transport copy/event ownership tests passed; no GPU executed\n";
    return failures ? 1 : 0;
}
