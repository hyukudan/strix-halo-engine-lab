#include "strix/direct_transport_gpu_export.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix::transport {
namespace {
[[noreturn]] void unavailable() { throw std::logic_error("GPU DMA-BUF export API is unavailable"); }
bool same_view(const TransportBufferView& a, const TransportBufferView& b) {
    return a.address == b.address && a.bytes == b.bytes && a.allocation_generation == b.allocation_generation &&
        a.domain == b.domain && a.access == b.access;
}
bool valid_device(const DirectGpuExportDevice& d) {
    return d.index >= 0 && d.runtime_version > 0 && d.driver_version > 0 &&
        std::any_of(d.uuid.begin(), d.uuid.end(), [](auto byte) { return byte != 0; });
}
void validate_allocation(const DirectGpuExportAllocation& a, const TransportBufferView& view, int device,
                         std::uint64_t maximum) {
    if (!a.device_memory || a.managed || a.device != device || a.base == 0 || a.bytes == 0 || a.bytes > maximum ||
        a.bytes > std::numeric_limits<std::uintptr_t>::max() - a.base || view.address < a.base ||
        view.address - a.base > a.bytes || view.bytes > a.bytes - (view.address - a.base)) {
        throw std::invalid_argument("GPU export view is outside an exact bounded native device allocation");
    }
}
} // namespace
DirectGpuExportDevice DirectGpuExportIo::inspect_device() { unavailable(); }
DirectGpuExportAllocation DirectGpuExportIo::inspect_allocation(const TransportBufferView&) { unavailable(); }
void DirectGpuExportIo::export_dma_buf(const DirectGpuExportAllocation&, int&) { unavailable(); }
DirectGpuExportFdIdentity DirectGpuExportIo::inspect_descriptor(int) { unavailable(); }
bool DirectGpuExportIo::close_descriptor(int) noexcept { return false; }

struct DirectGpuExportOwner::Impl {
    std::shared_ptr<DirectGpuExportIo> io;
    // Declaration order ensures pool is destroyed before its allocator.
    std::shared_ptr<RuntimeMemoryAllocator> allocator;
    std::shared_ptr<DirectTransportPoolMemoryOwner> pool;
    DirectGpuExportBinding binding;
    std::array<bool, 2> identified{};
    DirectGpuExportPolicy policy;
    bool failed = false;
    void current() {
        if (!pool || !pool->valid() || io->inspect_device() != binding.device) {
            throw std::runtime_error("GPU export device or owning pool changed");
        }
        const auto& p = pool->binding();
        if (p.binding_identity != binding.pool_binding_identity || p.links.size() != 1 ||
            !same_view(p.links[0].tx, binding.pool.tx) || !same_view(p.links[0].rx, binding.pool.rx)) {
            throw std::runtime_error("GPU export allocation generation/view changed");
        }
        for (std::size_t i = 0; i < 2; ++i) {
            const auto& view = i == 0 ? binding.pool.tx : binding.pool.rx;
            if (binding.descriptors[i] < 0 || !identified[i] ||
                io->inspect_allocation(view) != binding.allocations[i] ||
                io->inspect_descriptor(binding.descriptors[i]) != binding.descriptor_identities[i]) {
                throw std::runtime_error("GPU export native extent or owned FD changed");
            }
        }
        if (io->inspect_device() != binding.device) throw std::runtime_error("GPU export device changed during inspection");
    }
    DirectPoolReleaseStatus cleanup() noexcept {
        const auto allocator_released = [&]() {
            if (const auto* witness = dynamic_cast<const DirectGpuPoolReleaseWitness*>(allocator.get())) {
                return witness->release_owned_pool(binding.pool);
            }
            return DirectPoolReleaseStatus::released;
        };
        if (!pool && binding.descriptors[0] < 0 && binding.descriptors[1] < 0) {
            const auto result = allocator_released();
            if (result == DirectPoolReleaseStatus::released) allocator.reset();
            else failed = true;
            return result;
        }
        try {
            // We enqueue no GPU work. Current device is nevertheless required
            // before the existing allocator can release these local addresses.
            if (io->inspect_device() != binding.device) throw std::runtime_error("GPU export cleanup device changed");
            if (pool) {
                const auto& observed = pool->binding();
                if (observed.binding_identity != binding.pool_binding_identity || observed.links.size() != 1 ||
                    !same_view(observed.links[0].tx, binding.pool.tx) || !same_view(observed.links[0].rx, binding.pool.rx)) {
                    throw std::runtime_error("GPU export cleanup owning pool changed");
                }
                for (std::size_t i = 0; i < 2; ++i) {
                    if (binding.allocations[i].base != 0 && io->inspect_allocation(i == 0 ? binding.pool.tx : binding.pool.rx) != binding.allocations[i]) {
                        throw std::runtime_error("GPU export cleanup native allocation changed");
                    }
                }
            }
            for (std::size_t i = 0; i < 2; ++i) {
                auto& fd = binding.descriptors[i];
                if (fd < 0) continue;
                const auto observed = io->inspect_descriptor(fd);
                if (!identified[i]) { binding.descriptor_identities[i] = observed; identified[i] = true; }
                if (observed != binding.descriptor_identities[i] || !io->close_descriptor(fd)) {
                    throw std::runtime_error("GPU export cleanup FD identity/close unproved");
                }
                fd = -1;
            }
            if (io->inspect_device() != binding.device) throw std::runtime_error("GPU export cleanup device changed during close");
            if (pool) {
                for (std::size_t i = 0; i < 2; ++i) {
                    if (binding.allocations[i].base != 0 && io->inspect_allocation(i == 0 ? binding.pool.tx : binding.pool.rx) != binding.allocations[i]) {
                        throw std::runtime_error("GPU export cleanup native allocation changed during close");
                    }
                }
            }
            pool.reset();
            const auto result = allocator_released();
            if (result != DirectPoolReleaseStatus::released) { failed = true; return result; }
            allocator.reset(); return DirectPoolReleaseStatus::released;
        } catch (...) { failed = true; return DirectPoolReleaseStatus::failed_retained; }
    }
};
DirectGpuExportOwner::DirectGpuExportOwner() : impl_(std::make_unique<Impl>()) {}
std::shared_ptr<DirectGpuExportOwner> DirectGpuExportOwner::allocate_and_export(
    std::shared_ptr<DirectGpuExportIo> io, std::shared_ptr<RuntimeMemoryAllocator> allocator,
    const MultiLinkTransportPoolPlan& plan, AllocationGenerationSource& generations,
    DirectGpuExportDevice expected, DirectGpuExportPolicy policy) {
    if (!io || !allocator || !valid_device(expected) || policy.maximum_gpu_bytes == 0 ||
        policy.maximum_gpu_bytes > 128ULL * 1024ULL * 1024ULL || policy.maximum_alignment == 0 ||
        policy.maximum_alignment > 4096 || plan.links.size() != 1 || plan.total_gpu_bytes == 0 ||
        plan.total_gpu_bytes > policy.maximum_gpu_bytes) throw std::invalid_argument("GPU export requires one bounded owned pool pair and exact local device");
    const auto& link = plan.links[0];
    if (link.tx_pool_bytes == 0 || link.rx_pool_bytes == 0 || link.pool_alignment == 0 ||
        (link.pool_alignment & (link.pool_alignment - 1)) != 0 ||
        link.pool_alignment > policy.maximum_alignment || link.tx_pool_bytes > policy.maximum_gpu_bytes ||
        link.rx_pool_bytes > policy.maximum_gpu_bytes - link.tx_pool_bytes ||
        link.total_gpu_bytes != link.tx_pool_bytes + link.rx_pool_bytes || plan.total_gpu_bytes != link.total_gpu_bytes ||
        2 * (link.pool_alignment - 1) > policy.maximum_gpu_bytes - plan.total_gpu_bytes) {
        throw std::invalid_argument("GPU export pool allocation and alignment exceed admission");
    }
    if (io->inspect_device() != expected) throw std::invalid_argument("GPU export expected local HIP device/version does not match");
    auto owner = std::shared_ptr<DirectGpuExportOwner>(new DirectGpuExportOwner);
    auto& impl = *owner->impl_; impl.io = std::move(io); impl.allocator = std::move(allocator);
    impl.policy = policy; impl.binding.device = expected;
    impl.pool = std::make_shared<DirectTransportPoolMemoryOwner>(DirectTransportPoolMemoryOwner::allocate(plan, *impl.allocator, generations));
    impl.binding.pool_binding_identity = impl.pool->binding().binding_identity;
    impl.binding.pool = impl.pool->binding().links[0];
    if (auto* witness = dynamic_cast<DirectGpuPoolReleaseWitness*>(impl.allocator.get())) witness->bind_owned_pool(impl.binding.pool);
    for (std::size_t i = 0; i < 2; ++i) {
        if (impl.io->inspect_device() != expected) throw std::runtime_error("GPU export device changed during allocation");
        const auto& view = i == 0 ? impl.binding.pool.tx : impl.binding.pool.rx;
        auto& allocation = impl.binding.allocations[i]; allocation = impl.io->inspect_allocation(view);
        validate_allocation(allocation, view, expected.index, policy.maximum_gpu_bytes - impl.binding.actual_gpu_bytes);
        if (i == 1) {
            const auto& previous = impl.binding.allocations[0];
            if (allocation.base < previous.base + previous.bytes && previous.base < allocation.base + allocation.bytes) {
                throw std::invalid_argument("GPU export TX/RX must occupy disjoint native allocations");
            }
        }
        impl.binding.actual_gpu_bytes += allocation.bytes;
        impl.binding.view_offsets[i] = view.address - allocation.base;
        impl.io->export_dma_buf(allocation, impl.binding.descriptors[i]);
        if (i == 1 && impl.binding.descriptors[0] == impl.binding.descriptors[1]) {
            // A malformed callback returned the first owned FD twice. It did
            // not create a second ownership reference to close on cleanup.
            impl.binding.descriptors[1] = -1;
            throw std::runtime_error("GPU export repeated an already owned descriptor");
        }
        if (impl.binding.descriptors[i] < 0) {
            throw std::runtime_error("GPU export did not return a distinct owned DMA-BUF descriptor");
        }
        impl.binding.descriptor_identities[i] = impl.io->inspect_descriptor(impl.binding.descriptors[i]);
        impl.identified[i] = true;
        if (!impl.binding.descriptor_identities[i].close_on_exec) throw std::runtime_error("GPU export FD must be close-on-exec");
        if (impl.io->inspect_device() != expected || impl.io->inspect_allocation(view) != allocation) {
            throw std::runtime_error("GPU export device/allocation changed during callback");
        }
    }
    owner->require_current(); return owner;
}
DirectGpuExportOwner::~DirectGpuExportOwner() {
    if (impl_->cleanup() != DirectPoolReleaseStatus::released) static_cast<void>(impl_.release());
}
const DirectGpuExportBinding& DirectGpuExportOwner::binding() const noexcept { return impl_->binding; }
void DirectGpuExportOwner::require_current() {
    if (impl_->failed) throw std::logic_error("GPU export owner is failed");
    try { impl_->current(); } catch (...) { impl_->failed = true; throw; }
}
void DirectGpuExportOwner::require_import_match(const DirectDriverBinding& expected, const DirectDriverImportBinding& imported) {
    require_current();
    const auto& lease = imported.lease;
    const auto& interface = expected.interface;
    if (expected.adapter_contract_version != 1 || expected.device_identity.empty() || expected.device_instance_generation == 0 ||
        interface.proof_contract_version != 1 || interface.kernel_release.empty() || interface.driver_identity.empty() ||
        interface.abi_identity.empty() || interface.local_endpoint_identity.empty() || interface.peer_endpoint_identity.empty() ||
        interface.local_endpoint_identity == interface.peer_endpoint_identity ||
        imported.device != expected || lease.link_index != 0 || lease.lease_generation == 0 || imported.import_generation == 0 ||
        lease.physical_link_id.empty() || lease.physical_link_id != expected.interface.physical_link_id ||
        lease.connection_epoch == 0 || lease.connection_epoch != expected.interface.connection_epoch ||
        lease.pool_binding_identity != impl_->binding.pool_binding_identity ||
        !same_view(imported.pool.tx, impl_->binding.pool.tx) || !same_view(imported.pool.rx, impl_->binding.pool.rx)) {
        throw std::invalid_argument("GPU export does not match future driver's exact lease/import/pool binding");
    }
}
std::shared_ptr<DirectGpuExportImportLease> DirectGpuExportOwner::retain_for_import() {
    require_current();
    auto* witness = dynamic_cast<DirectGpuPoolReleaseWitness*>(impl_->allocator.get());
    if (!witness) throw std::logic_error("native import requires a guarded local pool allocator");
    auto guard = witness->retain_owned_pool(impl_->binding.pool);
    if (!guard) throw std::runtime_error("native import allocator returned no ownership guard");
    require_current();
    return std::shared_ptr<DirectGpuExportImportLease>(new DirectGpuExportImportLease(shared_from_this(), std::move(guard)));
}
void DirectGpuExportImportLease::require_current() {
    owner_->require_current();
    if (!allocator_guard_->current()) throw std::runtime_error("native import's allocator ownership guard is invalidated");
}
DirectNativeCapabilities DirectGpuExportOwner::native_capabilities() const {
    DirectNativeCapabilities result; // Driver query/import/events/visibility remain unavailable.
    try { if (!impl_->failed && impl_->pool) { impl_->current(); result.dma_buf_export = true; } }
    catch (...) { impl_->failed = true; }
    return result;
}
DirectPoolReleaseStatus DirectGpuExportOwner::release() noexcept {
    if (weak_from_this().use_count() > 1) return DirectPoolReleaseStatus::pending;
    return impl_->cleanup();
}
bool DirectGpuExportOwner::failed() const noexcept { return impl_->failed; }
bool DirectGpuExportOwner::retains_resources() const noexcept {
    return impl_->allocator || impl_->pool || impl_->binding.descriptors[0] >= 0 || impl_->binding.descriptors[1] >= 0;
}
} // namespace strix::transport
