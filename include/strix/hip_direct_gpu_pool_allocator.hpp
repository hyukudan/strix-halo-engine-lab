#pragma once

#include "strix/hip_direct_transport_gpu_export.hpp"

namespace strix::transport {

struct HipDirectGpuPoolPolicy {
    bool experimental_opt_in = false;
    std::uint64_t maximum_gpu_bytes = 128ULL * 1024ULL * 1024ULL;
    std::uint64_t maximum_alignment = 4096;
    // Optional whole-allocation request quantum; observed native capacity is still checked.
    std::uint64_t allocation_quantum = 1;
};
struct HipDirectGpuPoolCalls {
    HipDirectGpuExportCalls observations;
    std::function<hipError_t(void**, std::size_t, unsigned)> allocate_with_flags;
    std::function<hipError_t(void*)> free_allocation;
    [[nodiscard]] static HipDirectGpuPoolCalls native();
};

// Explicit device-only C1 allocator with exactly two owned allocation slots.
// Requests documented HIP uncached memory; successful allocation is not native
// uncached/visibility/driver qualification. Whole observed native capacities
// count against admission before publishing a view. A trusted serialized owner
// excludes resource replacement and proves driver/GPU quiescence before freeing.
// Errors stop new allocation; explicit drain can release retained local blocks,
// while unknown cleanup at destruction retains ownership for process life.
class HipDirectGpuPoolAllocator final : public RuntimeMemoryAllocator, public DirectGpuPoolReleaseWitness {
public:
    HipDirectGpuPoolAllocator(DirectGpuExportDevice, HipDirectGpuPoolPolicy,
                             HipDirectGpuPoolCalls calls = HipDirectGpuPoolCalls::native());
    ~HipDirectGpuPoolAllocator() override;
    HipDirectGpuPoolAllocator(const HipDirectGpuPoolAllocator&) = delete;
    HipDirectGpuPoolAllocator& operator=(const HipDirectGpuPoolAllocator&) = delete;
    [[nodiscard]] std::uintptr_t allocate_device(std::uint64_t bytes, std::uint64_t alignment) override;
    [[nodiscard]] std::uintptr_t allocate_host(std::uint64_t, std::uint64_t) override;
    void free_device(std::uintptr_t, std::uint64_t) noexcept override;
    void free_host(std::uintptr_t, std::uint64_t) noexcept override;
    void bind_owned_pool(const DirectTransportLinkPoolBinding&) override;
    [[nodiscard]] DirectPoolReleaseStatus release_owned_pool(const DirectTransportLinkPoolBinding&) const noexcept override;
    [[nodiscard]] std::shared_ptr<DirectGpuPoolUseLease> retain_owned_pool(const DirectTransportLinkPoolBinding&) override;
    [[nodiscard]] DirectPoolReleaseStatus drain() noexcept;
    [[nodiscard]] std::uint64_t retained_bytes() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] bool accounting_complete() const noexcept;
    [[nodiscard]] bool requests_uncached_memory() const noexcept { return true; }
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix::transport
