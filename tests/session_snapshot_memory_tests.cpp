#include "strix/session_snapshot_memory.hpp"

#include <iostream>
#include <stdexcept>

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
    int fail_after = -1;
    int calls = 0;
    int device_frees = 0;
    int host_frees = 0;

    std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        maybe_fail();
        const auto address = align(next_device, alignment);
        next_device =
            address +
            static_cast<std::uintptr_t>(bytes + alignment);
        return address;
    }

    std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        maybe_fail();
        const auto address = align(next_host, alignment);
        next_host =
            address +
            static_cast<std::uintptr_t>(bytes + alignment);
        return address;
    }

    void free_device(
        std::uintptr_t,
        std::uint64_t) noexcept override {
        ++device_frees;
    }

    void free_host(
        std::uintptr_t,
        std::uint64_t) noexcept override {
        ++host_frees;
    }

private:
    void maybe_fail() {
        if (fail_after >= 0 &&
            calls == fail_after) {
            ++calls;
            throw std::runtime_error(
                "injected snapshot allocation failure");
        }
        ++calls;
    }

    static std::uintptr_t align(
        std::uintptr_t value,
        std::uint64_t alignment) {
        const auto remainder =
            value %
            static_cast<std::uintptr_t>(alignment);
        return remainder == 0
            ? value
            : value +
                  static_cast<std::uintptr_t>(
                      alignment - remainder);
    }

    std::uintptr_t next_device =
        static_cast<std::uintptr_t>(0xb0000000ULL);
    std::uintptr_t next_host =
        static_cast<std::uintptr_t>(0xc0000000ULL);
};

}  // namespace

int main() {
    strix::SerialSessionSnapshotPlan snapshot;
    snapshot.plan_identity = 0x1234ULL;
    snapshot.alignment = 256;
    snapshot.device_bytes = 4096;
    snapshot.host_bytes = 512;

    FakeAllocator allocator;
    strix::AllocationGenerationSource generations;

    {
        auto owner =
            strix::SessionSnapshotMemoryOwner::allocate(
                snapshot,
                allocator,
                generations);

        expect(owner.valid(),
               "snapshot memory owner becomes valid after allocation");
        expect(owner.binding().snapshot_plan_identity ==
                   snapshot.plan_identity &&
               owner.binding().allocation_generation == 1,
               "snapshot memory binding preserves plan identity and allocation generation");
        expect(owner.binding().device_section_base % 256 == 0 &&
               owner.binding().host_section_base % 256 == 0,
               "snapshot sections honor requested alignment");

        auto moved = std::move(owner);
        expect(!owner.valid() && moved.valid(),
               "snapshot memory ownership moves safely");
    }

    expect(allocator.device_frees == 1 &&
               allocator.host_frees == 1,
           "snapshot memory owner frees both payload sections once");

    FakeAllocator failing;
    failing.fail_after = 1;
    strix::AllocationGenerationSource failure_generations;
    bool failure_seen = false;
    try {
        static_cast<void>(
            strix::SessionSnapshotMemoryOwner::allocate(
                snapshot,
                failing,
                failure_generations));
    } catch (const std::runtime_error&) {
        failure_seen = true;
    }
    expect(failure_seen &&
               failing.device_frees == 1 &&
               failing.host_frees == 0,
           "partial snapshot allocation failure releases device section");

    if (failures == 0) {
        std::cout << "session snapshot memory tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
