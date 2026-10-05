#include "strix/direct_transport_pool_memory.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeAllocator final
    : public strix::RuntimeMemoryAllocator {
public:
    std::uintptr_t next =
        static_cast<std::uintptr_t>(0x50000000ULL);
    int fail_after = -1;
    int allocations = 0;
    std::vector<std::pair<std::uintptr_t, std::uint64_t>> freed;

    std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        if (fail_after >= 0 && allocations >= fail_after) {
            throw std::runtime_error(
                "injected transport allocation failure");
        }
        ++allocations;
        const auto remainder =
            static_cast<std::uint64_t>(next) % alignment;
        if (remainder != 0) {
            next += static_cast<std::uintptr_t>(
                alignment - remainder);
        }
        const auto result = next;
        next += static_cast<std::uintptr_t>(
            bytes + alignment);
        return result;
    }

    std::uintptr_t allocate_host(
        std::uint64_t,
        std::uint64_t) override {
        throw std::logic_error(
            "direct transport pools must not allocate host memory");
    }

    void free_device(
        std::uintptr_t address,
        std::uint64_t bytes) noexcept override {
        freed.emplace_back(address, bytes);
    }

    void free_host(
        std::uintptr_t,
        std::uint64_t) noexcept override {}
};

}  // namespace

int main() {
    strix::transport::DirectFrameGeometry geometry;
    geometry.frame_bytes = 4096;
    geometry.ring_frames = 16;
    geometry.dedicated_terminal_frame = true;

    const std::array links{geometry, geometry};
    const auto plan =
        strix::transport::plan_multi_link_transport_pools(
            links);

    FakeAllocator allocator;
    strix::AllocationGenerationSource generations;

    {
        auto owner =
            strix::transport::DirectTransportPoolMemoryOwner::allocate(
                plan,
                allocator,
                generations);
        const auto& binding = owner.binding();

        expect(
            owner.valid() &&
                binding.links.size() == 2 &&
                binding.total_gpu_bytes == plan.total_gpu_bytes &&
                binding.binding_identity != 0,
            "direct transport owner materializes every planned link pool");
        expect(
            binding.links[0].tx.domain ==
                    strix::transport::TransportMemoryDomain::gpu_transport_pool &&
                binding.links[0].rx.domain ==
                    strix::transport::TransportMemoryDomain::gpu_transport_pool &&
                binding.links[0].tx.address != binding.links[0].rx.address,
            "TX and RX are distinct GPU transport-pool bindings");
        expect(
            binding.links[0].tx.allocation_generation !=
                    binding.links[0].rx.allocation_generation &&
                binding.links[1].tx.allocation_generation !=
                    binding.links[1].rx.allocation_generation,
            "every physical pool allocation has an independent generation");

        strix::transport::validate_direct_transport_pool_memory_binding(
            plan,
            binding);

        auto stale = binding;
        stale.links[0].tx.allocation_generation = 0;
        bool rejected = false;
        try {
            strix::transport::validate_direct_transport_pool_memory_binding(
                plan,
                stale);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected,
            "stale transport-pool generation is rejected");
    }

    expect(
        allocator.freed.size() == 4,
        "dual-link owner frees both TX/RX allocations per physical link");

    FakeAllocator failing;
    failing.fail_after = 1;
    strix::AllocationGenerationSource failing_generations;
    bool rolled_back = false;
    try {
        static_cast<void>(
            strix::transport::DirectTransportPoolMemoryOwner::allocate(
                plan,
                failing,
                failing_generations));
    } catch (const std::runtime_error&) {
        rolled_back = true;
    }
    expect(
        rolled_back &&
            failing.freed.size() == 1,
        "partial pool allocation failure rolls back already-owned GPU memory");

    if (failures == 0) {
        std::cout
            << "direct transport pool memory tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
