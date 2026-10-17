#include "strix/hip_direct_gpu_pool_allocator.hpp"

#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

// Only injected HIP/POSIX calls run here. Synthetic addresses and descriptors
// are not device memory, DMA-BUFs or native qualification evidence.
namespace {
using namespace strix;
using namespace strix::transport;
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; ++failures; }
}
template<class F> bool rejected(F function) {
    try { function(); } catch (const std::exception&) { return true; } return false;
}
struct State {
    DirectGpuExportDevice device{3, {71}, 720, 720};
    std::map<std::uintptr_t, std::size_t> allocations;
    std::map<int, int> descriptors;
    std::vector<unsigned> flags;
    std::vector<std::size_t> requests;
    std::uintptr_t next = 0x100003;
    std::size_t rounded_extra = 0;
    int alloc_calls = 0, free_calls = 0, next_fd = 90;
    bool alloc_error_with_output = false, alloc_throw_with_output = false;
    bool free_error = false, drift_after_free = false, managed = false, extent_error = false;
};
HipDirectGpuPoolCalls fixture_calls(const std::shared_ptr<State>& state) {
    HipDirectGpuPoolCalls calls;
    auto& observation = calls.observations;
    observation.get_device = [state](int* value) { *value = state->device.index; return hipSuccess; };
    observation.device_uuid = [state](hipUUID* value, hipDevice_t) {
        *value = {}; for (std::size_t i = 0; i < state->device.uuid.size(); ++i) value->bytes[i] = static_cast<char>(state->device.uuid[i]);
        return hipSuccess;
    };
    observation.runtime_version = [state](int* value) { *value = state->device.runtime_version; return hipSuccess; };
    observation.driver_version = [state](int* value) { *value = state->device.driver_version; return hipSuccess; };
    observation.pointer_attributes = [state](hipPointerAttribute_t* value, const void*) {
        *value = {}; value->type = hipMemoryTypeDevice; value->device = state->device.index; value->isManaged = state->managed ? 1 : 0;
        return hipSuccess;
    };
    observation.address_range = [state](hipDeviceptr_t* base, std::size_t* bytes, hipDeviceptr_t address) {
        if (state->extent_error) return hipErrorUnknown;
        const auto pointer = reinterpret_cast<std::uintptr_t>(address);
        for (const auto& [allocation, extent] : state->allocations) if (pointer >= allocation && pointer - allocation < extent) {
            *base = reinterpret_cast<hipDeviceptr_t>(allocation); *bytes = extent; return hipSuccess;
        }
        return hipErrorInvalidValue;
    };
    observation.dma_buf_range_export = [state](int* descriptor, hipDeviceptr_t, std::size_t) {
        *descriptor = state->next_fd++; state->descriptors.emplace(*descriptor, 0); return hipSuccess;
    };
    observation.get_fd_flags = [state](int fd) { const auto found = state->descriptors.find(fd); return found == state->descriptors.end() ? -1 : found->second; };
    observation.set_fd_flags = [state](int fd, int flags) { state->descriptors.at(fd) = flags; return 0; };
    observation.fd_identity = [state](int fd) {
        if (!state->descriptors.contains(fd)) throw std::runtime_error("fixture FD not owned");
        return DirectGpuExportFdIdentity{11, static_cast<std::uint64_t>(fd), false};
    };
    observation.close_fd = [state](int fd) { return state->descriptors.erase(fd) == 1 ? 0 : -1; };
    calls.allocate_with_flags = [state](void** raw, std::size_t bytes, unsigned flags) {
        ++state->alloc_calls; state->flags.push_back(flags); state->requests.push_back(bytes);
        const auto address = state->next; state->next += 0x100000;
        state->allocations.emplace(address, bytes + state->rounded_extra); *raw = reinterpret_cast<void*>(address);
        if (state->alloc_throw_with_output) throw std::runtime_error("fixture partial allocation exception");
        return state->alloc_error_with_output ? hipErrorOutOfMemory : hipSuccess;
    };
    calls.free_allocation = [state](void* raw) {
        ++state->free_calls; if (state->free_error) return hipErrorUnknown;
        if (state->allocations.erase(reinterpret_cast<std::uintptr_t>(raw)) != 1) return hipErrorInvalidValue;
        if (state->drift_after_free) { ++state->device.driver_version; state->drift_after_free = false; }
        return hipSuccess;
    };
    return calls;
}
HipDirectGpuPoolPolicy policy(std::uint64_t maximum = 1ULL << 20) { return {true, maximum, 4096}; }
std::shared_ptr<HipDirectGpuPoolAllocator> allocator(const std::shared_ptr<State>& state, std::uint64_t maximum = 1ULL << 20) {
    return std::make_shared<HipDirectGpuPoolAllocator>(state->device, policy(maximum), fixture_calls(state));
}
}

int main() {
    {
        auto state = std::make_shared<State>();
        HipDirectGpuPoolPolicy selected{true, 6ULL * 1024ULL * 1024ULL, 4096, 2ULL * 1024ULL * 1024ULL};
        HipDirectGpuPoolAllocator owner(state->device, selected, fixture_calls(state));
        const auto address = owner.allocate_device(4ULL * 1024ULL * 1024ULL, 4096);
        expect(state->requests == std::vector<std::size_t>{6ULL * 1024ULL * 1024ULL}, "selected backing quantum includes alignment slack before allocation");
        expect(owner.retained_bytes() == 6ULL * 1024ULL * 1024ULL, "whole rounded native allocation is charged");
        expect(rejected([&] { static_cast<void>(owner.allocate_device(1, 1)); }) && state->alloc_calls == 1, "rounded request cannot exceed remaining budget");
        owner.free_device(address, 4ULL * 1024ULL * 1024ULL);
        expect(owner.retained_bytes() == 0 && state->free_calls == 1, "rounded backing still frees by exact public view");
    }
    for (const auto quantum : {0ULL, 3ULL, 2ULL * 1024ULL * 1024ULL}) {
        auto state = std::make_shared<State>(); auto selected = policy(); selected.allocation_quantum = quantum;
        expect(rejected([&] { HipDirectGpuPoolAllocator owner(state->device, selected, fixture_calls(state)); }) && state->alloc_calls == 0, "invalid quantum is rejected without allocating");
    }

    {
        auto state = std::make_shared<State>(); auto calls = fixture_calls(state);
        expect(rejected([&] { HipDirectGpuPoolAllocator disabled(state->device, {}, calls); }), "uncached allocator requires explicit opt-in before allocation");
        expect(state->alloc_calls == 0, "default policy performs no allocation call");
    }
    {
        auto state = std::make_shared<State>(); state->rounded_extra = 32; auto owner = allocator(state);
        const auto first = owner->allocate_device(256, 64); const auto second = owner->allocate_device(128, 256);
        expect(first % 64 == 0 && second % 256 == 0, "aligned public views use existing allocator geometry");
        expect(state->requests == std::vector<std::size_t>{319, 383}, "native request includes exact alignment reservation");
        expect(state->flags == std::vector<unsigned>{hipDeviceMallocUncached, hipDeviceMallocUncached}, "every native allocation requests the SDK uncached flag exactly");
        expect(owner->retained_bytes() == 766 && owner->accounting_complete(), "full rounded native capacities count against admission");
        expect(rejected([&] { static_cast<void>(owner->allocate_device(1, 1)); }), "third live block is rejected before native allocation");
        expect(rejected([&] { static_cast<void>(owner->allocate_host(1, 1)); }), "dedicated transport allocator cannot create host allocations");
        owner->free_device(first, 256); owner->free_device(second, 128);
        expect(owner->retained_bytes() == 0 && state->free_calls == 2, "matching local frees release both full backing allocations");
    }
    {
        auto state = std::make_shared<State>(); auto owner = allocator(state, 512);
        const auto address = owner->allocate_device(256, 64);
        expect(rejected([&] { static_cast<void>(owner->allocate_device(256, 64)); }) && state->alloc_calls == 1, "byte admission is checked before native allocation");
        expect(rejected([&] { static_cast<void>(owner->allocate_device(1, 8192)); }), "alignment admission rejects excessive alignment");
        owner->free_device(address, 255);
        expect(owner->failed() && state->free_calls == 0 && owner->retained_bytes() == 319, "foreign free cannot release a matching-address different-extent owner");
        expect(owner->drain() == DirectPoolReleaseStatus::released, "explicit drain proves retained local blocks released without restoring allocation authority");
        expect(rejected([&] { static_cast<void>(owner->allocate_device(1, 1)); }), "failure remains sticky after cleanup");
    }
    for (const bool throws : {false, true}) {
        auto state = std::make_shared<State>(); state->alloc_error_with_output = !throws; state->alloc_throw_with_output = throws; state->free_error = true;
        auto owner = allocator(state);
        expect(rejected([&] { static_cast<void>(owner->allocate_device(128, 64)); }), "partial output is adopted before callback error or exception");
        expect(owner->failed() && owner->retained_bytes() == 191 && !owner->accounting_complete() && state->allocations.size() == 1,
               "unknown partial allocation cleanup retains source allocation and conservative reservation");
        state->free_error = false;
        expect(owner->drain() == DirectPoolReleaseStatus::released && state->allocations.empty(), "partial failure cleanup requires an explicit positive native free");
    }
    {
        auto state = std::make_shared<State>(); state->rounded_extra = 512; auto owner = allocator(state, 512);
        expect(rejected([&] { static_cast<void>(owner->allocate_device(128, 64)); }), "observed native rounding above budget cannot publish a buffer");
        expect(owner->failed() && owner->retained_bytes() == 0 && state->free_calls == 1, "rejected full extent is locally cleaned through an explicit free result");
    }
    {
        auto state = std::make_shared<State>(); state->managed = true; auto owner = allocator(state);
        expect(rejected([&] { static_cast<void>(owner->allocate_device(128, 64)); }) && state->free_calls == 1, "managed allocation cannot be exposed as an uncached transport pool");
    }
    {
        auto state = std::make_shared<State>(); auto owner = allocator(state); const auto address = owner->allocate_device(128, 64);
        const auto original = state->device; ++state->device.uuid[1]; owner->free_device(address, 128);
        expect(owner->failed() && state->free_calls == 0 && owner->retained_bytes() == 191, "UUID drift retains allocation and forbids native free against replaced device");
        state->device = original;
        expect(owner->drain() == DirectPoolReleaseStatus::released, "restored exact observation permits explicit local cleanup only");
    }
    {
        auto state = std::make_shared<State>(); auto owner = allocator(state); const auto address = owner->allocate_device(128, 64);
        const auto original = state->device; state->drift_after_free = true; owner->free_device(address, 128);
        expect(owner->failed() && owner->retained_bytes() == 191 && state->free_calls == 1, "post-free device drift rejects publication and retains release tombstone");
        state->device = original;
        expect(owner->drain() == DirectPoolReleaseStatus::released && state->free_calls == 1, "completed native free is never retried against a reused address");
    }
    {
        auto state = std::make_shared<State>(); auto owner = allocator(state); const auto address = owner->allocate_device(128, 64);
        state->extent_error = true; owner->free_device(address, 128);
        expect(owner->failed() && state->free_calls == 0 && owner->retained_bytes() == 191, "unavailable extent proof cannot authorize native free");
        state->extent_error = false; expect(owner->drain() == DirectPoolReleaseStatus::released, "verified native extent cleanup remains available after failure");
    }
#if defined(__linux__) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
    {
        auto state = std::make_shared<State>(); auto memory = allocator(state); auto io = std::make_shared<HipDirectGpuExportIo>(fixture_calls(state).observations);
        AllocationGenerationSource generations;
        const auto plan = plan_multi_link_transport_pools(std::array{DirectFrameGeometry{4096, 4, true}});
        auto exported = DirectGpuExportOwner::allocate_and_export(io, memory, plan, generations, state->device);
        auto imported = exported->retain_for_import();
        expect(exported->release() == DirectPoolReleaseStatus::pending && memory->drain() == DirectPoolReleaseStatus::pending && state->free_calls == 0,
               "live import lease excludes both explicit export release and allocator drain");
        expect(rejected([&] { static_cast<void>(exported->retain_for_import()); }), "one exclusive importer owns the exact pool pair");
        const auto pool = exported->binding().pool;
        memory->free_device(pool.tx.address, pool.tx.bytes);
        expect(memory->failed() && state->free_calls == 0 && state->allocations.size() == 2 && rejected([&] { imported->require_current(); }),
               "forbidden free during import retains buffers and invalidates productive use guard");
        // Only the real provider may drop this guard after controller AND GPU
        // quiescence. This injected fixture has enqueued neither kind of work.
        imported.reset();
        expect(exported->release() == DirectPoolReleaseStatus::released && state->allocations.empty(), "dropping a quiescent import guard permits exact local retirement");
    }
    {
        auto state = std::make_shared<State>(); auto memory = allocator(state); auto io = std::make_shared<HipDirectGpuExportIo>(fixture_calls(state).observations);
        AllocationGenerationSource generations;
        const auto plan = plan_multi_link_transport_pools(std::array{DirectFrameGeometry{4096, 4, true}});
        auto exported = DirectGpuExportOwner::allocate_and_export(io, memory, plan, generations, state->device);
        const auto capabilities = exported->native_capabilities();
        expect(memory->requests_uncached_memory() && capabilities.dma_buf_export && !capabilities.uncached_gpu_pool && !capabilities.external_dma_visibility_qualified,
               "requested uncached policy and local FD export cannot fabricate native allocation/visibility readiness");
        auto foreign_pool = exported->binding().pool; ++foreign_pool.tx.allocation_generation;
        expect(rejected([&] { memory->bind_owned_pool(foreign_pool); }), "bound generation cannot be replaced by a same-address foreign pool");
        state->free_error = true;
        expect(exported->release() == DirectPoolReleaseStatus::failed_retained && exported->retains_resources() && memory->retained_bytes() != 0,
               "export owner cannot report release after void allocator callback fails");
        expect(state->descriptors.empty() && state->allocations.size() == 2, "local FD close does not stand in for native backing release");
        state->free_error = false;
        expect(memory->drain() == DirectPoolReleaseStatus::released && exported->release() == DirectPoolReleaseStatus::released &&
               !exported->retains_resources(), "typed allocator release witness permits owner retirement only after explicit successful drain");
        expect(exported->failed() && memory->failed(), "successful cleanup does not heal failed allocation or export authority");
    }
#endif
    if (!failures) std::cout << "injected HIP uncached pool allocation/lifetime tests passed; no GPU executed\n";
    return failures ? 1 : 0;
}
