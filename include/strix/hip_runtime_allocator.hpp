#pragma once

#include "strix/runtime_allocator.hpp"
#include "strix/uma_owner_inventory.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace strix {

namespace transport { class HipDirectGpuPoolAllocator; }

enum class HipHostAllocationMode {
    pageable_aligned,
    pinned_mapped,
};
enum class HipRuntimeAllocationSpace { device, host };
struct HipRuntimeAllocationDescription {
    std::uint64_t lease_identity=0;
    std::uintptr_t raw_base=0;
    std::uint64_t raw_bytes=0,view_offset=0,view_bytes=0;
    bool operator==(const HipRuntimeAllocationDescription&) const = default;
};

void initialize_hip_fresh_runtime_memory(
    const ExecutionMemoryBinding& scratch,
    const SessionStateMemoryBinding& state);

class HipRuntimeAllocator final
    : public RuntimeMemoryAllocator, public std::enable_shared_from_this<HipRuntimeAllocator> {
public:
    // Process-local allocation issuer; actual lease IDs are minted on allocate,
    // independently of returned/raw addresses and caller generation numbers.
    static constexpr std::uint64_t inventory_issuer=0x484950414c4c3031ULL;
    explicit HipRuntimeAllocator(
        HipHostAllocationMode host_mode =
            HipHostAllocationMode::pageable_aligned);

    HipRuntimeAllocator(
        const HipRuntimeAllocator&) = delete;
    HipRuntimeAllocator& operator=(
        const HipRuntimeAllocator&) = delete;

    ~HipRuntimeAllocator() override;

    [[nodiscard]] std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) override;

    [[nodiscard]] std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t alignment) override;

    void free_device(
        std::uintptr_t address,
        std::uint64_t bytes) noexcept override;

    void free_host(
        std::uintptr_t address,
        std::uint64_t bytes) noexcept override;

    [[nodiscard]] HipHostAllocationMode
    host_mode() const noexcept {
        return host_mode_;
    }
    // Pure locked lookup of this allocator's LIVE exact requested allocation.
    // Returns alignment reservation, not allocator/driver physical-page size.
    // The actual owner must pin this allocator and allocation through use.
    [[nodiscard]] HipRuntimeAllocationDescription describe_allocation(
        std::uintptr_t,std::uint64_t,HipRuntimeAllocationSpace) const;
    [[nodiscard]] std::uint64_t allocation_lease_identity(
        std::uintptr_t,std::uint64_t,bool host) const override;

private:
    // Dedicated transport allocator reuses geometry, not general allocation
    // policy or cleanup semantics. Existing runtime defaults stay unchanged.
    friend class transport::HipDirectGpuPoolAllocator;
    struct BackingAllocation {
        std::uintptr_t raw_address = 0;
        std::uint64_t raw_bytes = 0;
        std::uint64_t requested_bytes = 0,lease_identity = 0;
    };

    [[nodiscard]] static std::uint64_t checked_total(
        std::uint64_t bytes,
        std::uint64_t alignment);

    [[nodiscard]] static std::uintptr_t align_address(
        std::uintptr_t address,
        std::uint64_t alignment);

    void release_all() noexcept;

    HipHostAllocationMode host_mode_;
    mutable std::mutex mutex_;
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> device_backing_;
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> pinned_host_backing_;
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> pageable_host_;
};

// Three actual retained pools only: scratch, device state and host state. The
// exact memory owner and its retained actual shared HIP allocator are required;
// borrowed/value allocators are rejected. D1/PLE/caches/transport stay outside
// this declared scope. No allocation/copy/completion or physical-fit inference.
[[nodiscard]] UmaOwnerInventory hip_runtime_memory_owner_inventory(
    std::shared_ptr<const RuntimeExecutionMemoryOwner>,UmaMemoryObservation = {});

}  // namespace strix
