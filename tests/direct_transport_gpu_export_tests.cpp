#include "strix/direct_transport_gpu_export.hpp"

#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
using namespace strix;
using namespace strix::transport;
int failures = 0;
void expect(bool value, const char* message) { if (!value) { std::cerr << "FAIL: " << message << '\n'; ++failures; } }
template<class F> bool rejected(F function) { try { function(); } catch (const std::exception&) { return true; } return false; }
DirectGpuExportDevice device() { DirectGpuExportDevice value; value.index = 2; value.uuid[0] = 81; value.runtime_version = 720; value.driver_version = 720; return value; }
MultiLinkTransportPoolPlan plan() {
    const std::array geometry{DirectFrameGeometry{4096, 4, true}};
    return plan_multi_link_transport_pools(geometry);
}
struct AllocatorState { int allocated = 0, freed = 0; };
class Allocator final : public RuntimeMemoryAllocator {
public:
    explicit Allocator(std::shared_ptr<AllocatorState> state) : state_(std::move(state)) {}
    std::uintptr_t allocate_device(std::uint64_t bytes, std::uint64_t alignment) override {
        const auto address = next_; next_ += bytes + alignment; ++state_->allocated; return address;
    }
    std::uintptr_t allocate_host(std::uint64_t, std::uint64_t) override { throw std::logic_error("GPU pool only"); }
    void free_device(std::uintptr_t, std::uint64_t) noexcept override { ++state_->freed; }
    void free_host(std::uintptr_t, std::uint64_t) noexcept override {}
private:
    std::uintptr_t next_ = 0x100000;
    std::shared_ptr<AllocatorState> state_;
};
// Synthetic local GPU/FD records only. These fixtures cannot qualify DMA.
class Io final : public DirectGpuExportIo {
public:
    DirectGpuExportDevice inspect_device() override { return current_device; }
    DirectGpuExportAllocation inspect_allocation(const TransportBufferView& view) override {
        ++inspections; const auto base = view.address - offset;
        return {base, view.bytes + extent_extra, native_device, device_memory, managed};
    }
    void export_dma_buf(const DirectGpuExportAllocation&, int& descriptor) override {
        ++exports; descriptor = duplicate && exports == 2 ? 10 : next_fd++;
        descriptors[descriptor] = {1, static_cast<std::uint64_t>(descriptor) + 200, cloexec};
        if (on_export) on_export();
        if (throw_export_at == exports) throw std::runtime_error("partial export failed after FD ownership transfer");
    }
    DirectGpuExportFdIdentity inspect_descriptor(int descriptor) override {
        if (descriptor_throw_once) { descriptor_throw_once = false; throw std::runtime_error("descriptor query failure"); }
        const auto found = descriptors.find(descriptor);
        if (found == descriptors.end()) throw std::runtime_error("FD no longer owned");
        return found->second;
    }
    bool close_descriptor(int descriptor) noexcept override {
        if (close_fails) return false;
        if (descriptors.erase(descriptor) == 0) return false;
        ++closed;
        if (on_close) on_close();
        return true;
    }
    DirectGpuExportDevice current_device = device();
    int native_device = 2, exports = 0, closed = 0, inspections = 0, throw_export_at = -1, next_fd = 10;
    std::uint64_t extent_extra = 0, offset = 0;
    bool device_memory = true, managed = false, cloexec = true, close_fails = false, duplicate = false, descriptor_throw_once = false;
    std::function<void()> on_export, on_close;
    std::map<int, DirectGpuExportFdIdentity> descriptors;
};
struct Fixture {
    std::shared_ptr<AllocatorState> state = std::make_shared<AllocatorState>();
    std::shared_ptr<Allocator> allocator = std::make_shared<Allocator>(state);
    std::shared_ptr<Io> io = std::make_shared<Io>();
    AllocationGenerationSource generations;
    std::shared_ptr<DirectGpuExportOwner> prepare(DirectGpuExportPolicy policy = {}) {
        return DirectGpuExportOwner::allocate_and_export(io, allocator, plan(), generations, device(), policy);
    }
};
DirectDriverImportBinding imported(const DirectGpuExportOwner& owner) {
    DirectDriverImportBinding binding;
    binding.device.adapter_contract_version = 1; binding.device.device_identity = "synthetic-driver"; binding.device.device_instance_generation = 1;
    binding.device.interface = {1, "synthetic-kernel", "synthetic-driver", "synthetic-only-ABI", "host-a", "host-b", "synthetic-link", 77};
    binding.lease = {"synthetic-link", 77, owner.binding().pool_binding_identity, 0, 1};
    binding.pool = owner.binding().pool; binding.import_generation = 1; return binding;
}
} // namespace

int main() {
    {
        Fixture fixture; auto owner = fixture.prepare();
        const auto& binding = owner->binding();
        expect(binding.pool.tx.address != binding.pool.rx.address && binding.pool.tx.allocation_generation != binding.pool.rx.allocation_generation &&
               binding.descriptors[0] != binding.descriptors[1] && binding.actual_gpu_bytes == plan().total_gpu_bytes,
               "existing pool owner supplies independent TX/RX generations and exactly admitted native exports");
        std::weak_ptr<Allocator> lifetime = fixture.allocator; fixture.allocator.reset();
        expect(!lifetime.expired() && owner->retains_resources(), "export retains actual allocator and pool after caller references are dropped");
        auto pending_import_owner = owner;
        expect(owner->release() == DirectPoolReleaseStatus::pending && fixture.state->freed == 0 && fixture.io->closed == 0,
               "shared importer ownership prevents early descriptor/pool release");
        pending_import_owner.reset();
        expect(owner->release() == DirectPoolReleaseStatus::released && fixture.state->freed == 2 && fixture.io->closed == 2 && lifetime.expired(),
               "sole local owner can close both exact export FDs before freeing original pool allocations");
        expect(owner->release() == DirectPoolReleaseStatus::released, "completed local export cleanup is idempotent");
    }
    {
        Fixture fixture; fixture.io->offset = 64; fixture.io->extent_extra = 128;
        auto owner = fixture.prepare();
        expect(owner->binding().view_offsets[0] == 64 && owner->binding().view_offsets[1] == 64 &&
               owner->binding().actual_gpu_bytes == plan().total_gpu_bytes + 256,
               "full native allocations including unused padding, not requested views alone, count against admission");
        auto token = imported(*owner); owner->require_import_match(token.device, token);
        auto unknown = token; unknown.device.interface.abi_identity.clear();
        expect(rejected([&] { owner->require_import_match(unknown.device, unknown); }),
               "unknown driver ABI cannot pass future import consistency seam even with matching local exports");
        for (int field = 0; field < 7; ++field) {
            auto foreign = token;
            if (field == 0) ++foreign.lease.connection_epoch;
            if (field == 1) foreign.lease.physical_link_id += "-foreign";
            if (field == 2) ++foreign.lease.pool_binding_identity;
            if (field == 3) ++foreign.pool.rx.allocation_generation;
            if (field == 4) ++foreign.pool.tx.address;
            if (field == 5) ++foreign.device.device_instance_generation;
            if (field == 6) foreign.device.interface.abi_identity += "-foreign";
            expect(rejected([&] { owner->require_import_match(token.device, foreign); }),
                   "future import seam rejects foreign ABI/device/lease/physical identity/pool window");
        }
        const auto native = owner->native_capabilities();
        expect(native.dma_buf_export && !native.dma_buf_import && !native.uncached_gpu_pool && !native.external_dma_visibility_qualified &&
               native.query_status == DirectNativeQueryStatus::unavailable && !assess_direct_native_readiness({}, native).ready,
               "successful local export alone cannot self-qualify native import/events/visibility or transport readiness");
        DirectTransportDriverIo unavailable_driver; DirectTransportDriverBackend backend(unavailable_driver, token.device);
        expect(backend.direct_capabilities().query_status == DirectNativeQueryStatus::unavailable,
               "concrete GPU export does not replace missing driver ABI query with fabricated readiness");
    }
    for (int fault = 0; fault < 8; ++fault) {
        Fixture fixture;
        auto expected = device(); auto geometry = plan(); auto policy = DirectGpuExportPolicy{};
        if (fault == 0) ++expected.uuid[0];
        if (fault == 1) policy.maximum_gpu_bytes = geometry.total_gpu_bytes - 1;
        if (fault == 2) fixture.io->managed = true;
        if (fault == 3) fixture.io->native_device = 1;
        if (fault == 4) fixture.io->device_memory = false;
        if (fault == 5) fixture.io->extent_extra = 128ULL * 1024ULL * 1024ULL;
        if (fault == 6) fixture.io->cloexec = false;
        if (fault == 7) fixture.io->duplicate = true;
        expect(rejected([&] { static_cast<void>(DirectGpuExportOwner::allocate_and_export(
            fixture.io, fixture.allocator, geometry, fixture.generations, expected, policy)); }),
            "incompatible UUID/admission/managed/foreign device/native extent/CLOEXEC/duplicate export rejects cleanly");
        expect(fixture.state->freed == fixture.state->allocated && fixture.io->descriptors.empty(),
               "failed never-armed export preparation cleans each locally owned allocation/FD exactly once");
    }
    for (int at : {1, 2}) {
        Fixture fixture; fixture.io->throw_export_at = at;
        expect(rejected([&] { static_cast<void>(fixture.prepare()); }) && fixture.io->closed == at &&
               fixture.state->freed == 2 && fixture.io->descriptors.empty(),
               "partial native callback exception cannot hide an already returned FD or free allocator first");
    }
    for (int fault = 0; fault < 5; ++fault) {
        Fixture fixture; auto owner = fixture.prepare(); const auto snapshots = owner->binding().descriptor_identities;
        if (fault == 0) ++fixture.io->current_device.uuid[0];
        if (fault == 1) ++fixture.io->current_device.runtime_version;
        if (fault == 2) ++fixture.io->current_device.driver_version;
        if (fault == 3) ++fixture.io->extent_extra;
        if (fault == 4) ++fixture.io->descriptors[owner->binding().descriptors[0]].inode;
        expect(rejected([&] { owner->require_current(); }) && owner->failed() && !owner->native_capabilities().dma_buf_export,
               "post-export device/runtime/driver/native extent/FD replacement invalidates cached export proof");
        expect(owner->release() == DirectPoolReleaseStatus::failed_retained && fixture.io->closed == 0 && fixture.state->freed == 0,
               "known device/allocation/FD replacement cannot authorize release of old export resources");
        if (fault == 4) {
            fixture.io->descriptors[owner->binding().descriptors[0]] = snapshots[0];
        }
        fixture.io->current_device = device(); fixture.io->extent_extra = 0;
        expect(owner->release() == DirectPoolReleaseStatus::released && owner->failed(),
               "exact restored cleanup can release locals without making failed export usable again");
    }
    {
        Fixture fixture; auto owner = fixture.prepare(); fixture.io->close_fails = true;
        expect(owner->release() == DirectPoolReleaseStatus::failed_retained && owner->retains_resources() && fixture.state->freed == 0,
               "unproved FD close cannot authorize pool release");
        fixture.io->close_fails = false;
        expect(owner->release() == DirectPoolReleaseStatus::released && fixture.state->freed == 2,
               "later exact FD cleanup releases retained bounded pools");
    }
    {
        Fixture fixture; auto owner = fixture.prepare(); ++fixture.io->current_device.uuid[0];
        expect(!owner->native_capabilities().dma_buf_export && owner->failed(),
               "capability inspection itself records known qualification loss instead of healing on later restore");
        fixture.io->current_device = device();
        expect(!owner->native_capabilities().dma_buf_export && owner->release() == DirectPoolReleaseStatus::released,
               "restored local cleanup cannot recover failed export capability");
    }
    {
        Fixture fixture; auto owner = fixture.prepare(); fixture.io->on_close = [&] { fixture.io->extent_extra = 128; };
        expect(owner->release() == DirectPoolReleaseStatus::failed_retained && fixture.io->closed == 2 && fixture.state->freed == 0,
               "native allocation drift during FD-close callback retains pools after local descriptors are closed");
        fixture.io->on_close = {}; fixture.io->extent_extra = 0;
        expect(owner->release() == DirectPoolReleaseStatus::released && owner->failed(),
               "matching later cleanup frees retained pools without exposing stale export success");
    }
    {
        Fixture fixture; std::shared_ptr<DirectGpuExportIo> absent = std::make_shared<DirectGpuExportIo>();
        expect(rejected([&] { static_cast<void>(DirectGpuExportOwner::allocate_and_export(absent, fixture.allocator, plan(), fixture.generations, device())); }) &&
               fixture.state->allocated == 0, "unavailable export provider rejects before any allocation or guessed syscall");
    }
    if (failures == 0) std::cout << "direct GPU export ownership fixtures passed; native driver/DMA unqualified\n";
    return failures == 0 ? 0 : 1;
}
