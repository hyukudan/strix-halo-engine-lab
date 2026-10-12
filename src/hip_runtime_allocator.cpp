#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_setup_quiescence.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

bool valid_alignment(
    std::uint64_t alignment) noexcept {
    return alignment != 0 &&
           (alignment &
            (alignment - 1ULL)) == 0;
}

}  // namespace


void initialize_hip_fresh_runtime_memory(
    const ExecutionMemoryBinding& scratch,
    const SessionStateMemoryBinding& state) {
    if (scratch.arena_base == 0 ||
        scratch.arena_bytes == 0 ||
        scratch.allocation_generation == 0) {
        throw std::invalid_argument(
            "fresh HIP runtime initialization requires bound scratch memory");
    }

    // Validate every host/device binding before the first potentially queued
    // operation, so a later schema error cannot unwind pending initialization.
    if ((state.device_bytes != 0 &&
         (state.device_base == 0 || state.device_allocation_generation == 0)) ||
        (state.host_bytes != 0 &&
         (state.host_base == 0 || state.host_allocation_generation == 0))) {
        throw std::invalid_argument("fresh HIP runtime initialization requires bound state");
    }
    if (scratch.arena_bytes > std::numeric_limits<std::size_t>::max() ||
        state.device_bytes > std::numeric_limits<std::size_t>::max() ||
        state.host_bytes > std::numeric_limits<std::size_t>::max() ||
        scratch.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - scratch.arena_base ||
        state.device_bytes > std::numeric_limits<std::uintptr_t>::max() - state.device_base ||
        state.host_bytes > std::numeric_limits<std::uintptr_t>::max() - state.host_base) {
        throw std::overflow_error("fresh HIP runtime initialization size does not fit size_t");
    }
    const HipSetupQuiescenceGuard setup_guard;
    setup_guard.run([&] {
        check_hip(
            hipMemset(
                reinterpret_cast<void*>(scratch.arena_base),
                0,
                static_cast<std::size_t>(scratch.arena_bytes)),
            "hipMemset(fresh scratch)");

        if (state.device_bytes != 0) {
            check_hip(
                hipMemset(
                    reinterpret_cast<void*>(state.device_base),
                    0,
                    static_cast<std::size_t>(state.device_bytes)),
                "hipMemset(fresh device state)");
        }
        if (state.host_bytes != 0) {
            std::memset(
                reinterpret_cast<void*>(state.host_base),
                0,
                static_cast<std::size_t>(state.host_bytes));
        }
    });
}

HipRuntimeAllocator::HipRuntimeAllocator(
    HipHostAllocationMode host_mode)
    : host_mode_(host_mode) {}

HipRuntimeAllocator::~HipRuntimeAllocator() {
    release_all();
}

std::uint64_t HipRuntimeAllocator::checked_total(
    std::uint64_t bytes,
    std::uint64_t alignment) {
    if (bytes == 0 ||
        !valid_alignment(alignment)) {
        throw std::invalid_argument(
            "HIP runtime allocation requires non-zero bytes and power-of-two alignment");
    }
    const auto padding =
        alignment - 1ULL;
    if (padding >
        std::numeric_limits<std::uint64_t>::max() -
            bytes) {
        throw std::overflow_error(
            "HIP runtime allocation size overflow");
    }
    return bytes + padding;
}

std::uintptr_t HipRuntimeAllocator::align_address(
    std::uintptr_t address,
    std::uint64_t alignment) {
    const auto mask =
        static_cast<std::uintptr_t>(
            alignment - 1ULL);
    if (address >
        std::numeric_limits<std::uintptr_t>::max() -
            mask) {
        throw std::overflow_error(
            "HIP runtime aligned address overflow");
    }
    return (address + mask) & ~mask;
}

std::uintptr_t HipRuntimeAllocator::allocate_device(
    std::uint64_t bytes,
    std::uint64_t alignment) {
    const auto total =
        checked_total(bytes, alignment);

    void* raw = nullptr;
    check_hip(
        hipMalloc(
            &raw,
            static_cast<std::size_t>(total)),
        "hipMalloc");

    try {
        const auto raw_address =
            reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned =
            align_address(
                raw_address,
                alignment);

        std::lock_guard lock(mutex_);
        const auto [iterator, inserted] =
            device_backing_.emplace(
                aligned,
                BackingAllocation{
                    raw_address,
                    total,
                });
        static_cast<void>(iterator);
        if (!inserted) {
            throw std::runtime_error(
                "duplicate aligned HIP device allocation address");
        }
        return aligned;
    } catch (...) {
        static_cast<void>(
            hipFree(raw));
        throw;
    }
}

std::uintptr_t HipRuntimeAllocator::allocate_host(
    std::uint64_t bytes,
    std::uint64_t alignment) {
    const auto total =
        checked_total(bytes, alignment);

    if (host_mode_ ==
        HipHostAllocationMode::pageable_aligned) {
        void* pointer = nullptr;
        const auto host_alignment =
            std::max<std::uint64_t>(
                alignment,
                sizeof(void*));
        const auto result =
            posix_memalign(
                &pointer,
                static_cast<std::size_t>(
                    host_alignment),
                static_cast<std::size_t>(
                    bytes));
        if (result != 0 ||
            pointer == nullptr) {
            throw std::bad_alloc();
        }

        const auto address =
            reinterpret_cast<std::uintptr_t>(
                pointer);
        std::lock_guard lock(mutex_);
        const auto [iterator, inserted] =
            pageable_host_.emplace(
                address,
                bytes);
        static_cast<void>(iterator);
        if (!inserted) {
            std::free(pointer);
            throw std::runtime_error(
                "duplicate pageable host allocation address");
        }
        return address;
    }

    void* raw = nullptr;
    check_hip(
        hipHostMalloc(
            &raw,
            static_cast<std::size_t>(total),
            hipHostMallocMapped),
        "hipHostMalloc");

    try {
        const auto raw_address =
            reinterpret_cast<std::uintptr_t>(
                raw);
        const auto aligned =
            align_address(
                raw_address,
                alignment);

        std::lock_guard lock(mutex_);
        const auto [iterator, inserted] =
            pinned_host_backing_.emplace(
                aligned,
                BackingAllocation{
                    raw_address,
                    total,
                });
        static_cast<void>(iterator);
        if (!inserted) {
            throw std::runtime_error(
                "duplicate aligned HIP host allocation address");
        }
        return aligned;
    } catch (...) {
        static_cast<void>(
            hipHostFree(raw));
        throw;
    }
}

void HipRuntimeAllocator::free_device(
    std::uintptr_t address,
    std::uint64_t) noexcept {
    std::uintptr_t raw = 0;
    {
        std::lock_guard lock(mutex_);
        const auto found =
            device_backing_.find(
                address);
        if (found == device_backing_.end()) {
            return;
        }
        raw = found->second.raw_address;
        device_backing_.erase(found);
    }
    static_cast<void>(
        hipFree(
            reinterpret_cast<void*>(raw)));
}

void HipRuntimeAllocator::free_host(
    std::uintptr_t address,
    std::uint64_t) noexcept {
    if (host_mode_ ==
        HipHostAllocationMode::pageable_aligned) {
        {
            std::lock_guard lock(mutex_);
            const auto found =
                pageable_host_.find(
                    address);
            if (found == pageable_host_.end()) {
                return;
            }
            pageable_host_.erase(found);
        }
        std::free(
            reinterpret_cast<void*>(
                address));
        return;
    }

    std::uintptr_t raw = 0;
    {
        std::lock_guard lock(mutex_);
        const auto found =
            pinned_host_backing_.find(
                address);
        if (found ==
            pinned_host_backing_.end()) {
            return;
        }
        raw = found->second.raw_address;
        pinned_host_backing_.erase(found);
    }
    static_cast<void>(
        hipHostFree(
            reinterpret_cast<void*>(raw)));
}

void HipRuntimeAllocator::release_all() noexcept {
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> devices;
    std::unordered_map<
        std::uintptr_t,
        BackingAllocation> pinned;
    std::unordered_map<
        std::uintptr_t,
        std::uint64_t> pageable;

    {
        std::lock_guard lock(mutex_);
        devices.swap(device_backing_);
        pinned.swap(pinned_host_backing_);
        pageable.swap(pageable_host_);
    }

    for (const auto& [address, allocation] :
         devices) {
        static_cast<void>(address);
        static_cast<void>(
            hipFree(
                reinterpret_cast<void*>(
                    allocation.raw_address)));
    }
    for (const auto& [address, allocation] :
         pinned) {
        static_cast<void>(address);
        static_cast<void>(
            hipHostFree(
                reinterpret_cast<void*>(
                    allocation.raw_address)));
    }
    for (const auto& [address, bytes] :
         pageable) {
        static_cast<void>(bytes);
        std::free(
            reinterpret_cast<void*>(
                address));
    }
}

}  // namespace strix
