#pragma once

#include "strix/runtime_allocator.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace strix {

enum class HipHostAllocationMode {
    pageable_aligned,
    pinned_mapped,
};

class HipRuntimeAllocator final
    : public RuntimeMemoryAllocator {
public:
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

private:
    struct BackingAllocation {
        std::uintptr_t raw_address = 0;
        std::uint64_t raw_bytes = 0;
    };

    [[nodiscard]] static std::uint64_t checked_total(
        std::uint64_t bytes,
        std::uint64_t alignment);

    [[nodiscard]] static std::uintptr_t align_address(
        std::uintptr_t address,
        std::uint64_t alignment);

    void release_all() noexcept;

    HipHostAllocationMode host_mode_;
    std::mutex mutex_;
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> device_backing_;
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> pinned_host_backing_;
    std::unordered_map<
        std::uintptr_t,
        std::uint64_t> pageable_host_;
};

}  // namespace strix
