#pragma once

#include "strix/direct_transport_driver.hpp"

#include <array>
#include <memory>

namespace strix::transport {

// Local HIP observations only. A UUID/version match is not a transport ABI,
// external-DMA visibility or hardware-reset-generation proof.
struct DirectGpuExportDevice {
    std::int32_t index = -1;
    std::array<std::uint8_t, 16> uuid{};
    std::int32_t runtime_version = 0;
    std::int32_t driver_version = 0;
    bool operator==(const DirectGpuExportDevice&) const = default;
};
struct DirectGpuExportAllocation {
    std::uintptr_t base = 0;
    std::uint64_t bytes = 0;
    std::int32_t device = -1;
    bool device_memory = false;
    bool managed = false;
    bool operator==(const DirectGpuExportAllocation&) const = default;
};
struct DirectGpuExportFdIdentity {
    std::uint64_t device = 0, inode = 0;
    bool close_on_exec = false;
    bool operator==(const DirectGpuExportFdIdentity&) const = default;
};

// Export does not arm DMA or enqueue GPU work. The callback writes a newly
// owned descriptor into the supplied slot before any later failing work, so
// partial success cannot hide it. Never return a borrowed/reused descriptor.
class DirectGpuExportIo {
public:
    virtual ~DirectGpuExportIo() = default;
    [[nodiscard]] virtual DirectGpuExportDevice inspect_device();
    [[nodiscard]] virtual DirectGpuExportAllocation inspect_allocation(const TransportBufferView&);
    virtual void export_dma_buf(const DirectGpuExportAllocation&, int& owned_descriptor);
    [[nodiscard]] virtual DirectGpuExportFdIdentity inspect_descriptor(int);
    // True proves this exact owned descriptor was closed. No retry after an
    // ambiguous close is permitted unless the provider still proves identity.
    [[nodiscard]] virtual bool close_descriptor(int) noexcept;
};

struct DirectGpuExportPolicy {
    std::uint64_t maximum_gpu_bytes = 128ULL * 1024ULL * 1024ULL;
    std::uint64_t maximum_alignment = 4096;
};
struct DirectGpuExportBinding {
    DirectGpuExportDevice device;
    DirectTransportLinkPoolBinding pool;
    std::uint64_t pool_binding_identity = 0;
    std::uint64_t actual_gpu_bytes = 0;
    std::array<DirectGpuExportAllocation, 2> allocations; // TX then RX, full native extents.
    std::array<int, 2> descriptors{-1, -1};
    std::array<DirectGpuExportFdIdentity, 2> descriptor_identities;
    std::array<std::uint64_t, 2> view_offsets{};
};

// Optional exact release witness for allocators that retain failed native frees.
// Void RuntimeMemoryAllocator callbacks alone cannot prove such pool retirement.
class DirectGpuPoolUseLease {
public:
    virtual ~DirectGpuPoolUseLease() = default;
    [[nodiscard]] virtual bool current() const noexcept = 0;
};
class DirectGpuPoolReleaseWitness {
public:
    virtual ~DirectGpuPoolReleaseWitness() = default;
    virtual void bind_owned_pool(const DirectTransportLinkPoolBinding&) = 0;
    [[nodiscard]] virtual DirectPoolReleaseStatus release_owned_pool(const DirectTransportLinkPoolBinding&) const noexcept = 0;
    // Opaque shared guard prohibits local allocation free/drain while an
    // importer or GPU operation still owns the exact bound pair.
    [[nodiscard]] virtual std::shared_ptr<DirectGpuPoolUseLease> retain_owned_pool(const DirectTransportLinkPoolBinding&) = 0;
};

class DirectGpuExportImportLease;

// One serialized owner, one newly allocated TX/RX pair. Reuses the existing
// pool allocator/binding contract and retains its allocator lifetime. The
// returned FDs are borrowed local references; consumers must retain this
// shared owner and never close or replace them. A future DirectDriverImport
// must retain it until real driver AND GPU quiescence have been proved.
// This component never imports/arms a driver or marks a native link ready.
class DirectGpuExportOwner final : public std::enable_shared_from_this<DirectGpuExportOwner> {
public:
    [[nodiscard]] static std::shared_ptr<DirectGpuExportOwner> allocate_and_export(
        std::shared_ptr<DirectGpuExportIo>, std::shared_ptr<RuntimeMemoryAllocator>,
        const MultiLinkTransportPoolPlan&, AllocationGenerationSource&,
        DirectGpuExportDevice expected, DirectGpuExportPolicy policy = {});
    ~DirectGpuExportOwner();
    DirectGpuExportOwner(const DirectGpuExportOwner&) = delete;
    DirectGpuExportOwner& operator=(const DirectGpuExportOwner&) = delete;
    [[nodiscard]] const DirectGpuExportBinding& binding() const noexcept;
    void require_current();
    // Local consistency check for a future documented importer; not import,
    // lease acquisition, ABI negotiation or positive readiness evidence.
    void require_import_match(const DirectDriverBinding& expected, const DirectDriverImportBinding&);
    // Requires a guarded allocator. Its lease blocks explicit owner release
    // and allocator drain until the importer proves controller+GPU quiescence.
    [[nodiscard]] std::shared_ptr<DirectGpuExportImportLease> retain_for_import();
    [[nodiscard]] DirectNativeCapabilities native_capabilities() const;
    // Other shared owners may represent importer use, so explicit release is
    // refused while they remain. Destruction with unresolved identity/close
    // failure retains the bounded pool/allocator/FD ownership for process life.
    [[nodiscard]] DirectPoolReleaseStatus release() noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] bool retains_resources() const noexcept;
private:
    DirectGpuExportOwner();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DirectGpuExportImportLease final {
public:
    [[nodiscard]] const DirectGpuExportBinding& binding() const noexcept { return owner_->binding(); }
    void require_current();
private:
    friend class DirectGpuExportOwner;
    DirectGpuExportImportLease(std::shared_ptr<DirectGpuExportOwner> owner, std::shared_ptr<DirectGpuPoolUseLease> allocator_guard)
        : owner_(std::move(owner)), allocator_guard_(std::move(allocator_guard)) {}
    std::shared_ptr<DirectGpuExportOwner> owner_;
    std::shared_ptr<DirectGpuPoolUseLease> allocator_guard_;
};

} // namespace strix::transport
