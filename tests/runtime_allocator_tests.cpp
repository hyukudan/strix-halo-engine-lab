#include "strix/runtime_allocator.hpp"

#include <iostream>
#include <stdexcept>
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
    int fail_after = -1;
    int allocation_calls = 0;
    int device_frees = 0;
    int host_frees = 0;

    std::uintptr_t next_device =
        static_cast<std::uintptr_t>(
            0x90000000ULL);
    std::uintptr_t next_host =
        static_cast<std::uintptr_t>(
            0xa0000000ULL);

    std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        maybe_fail();
        const auto address =
            align(next_device, alignment);
        next_device =
            address +
            static_cast<std::uintptr_t>(
                bytes + alignment);
        return address;
    }

    std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        maybe_fail();
        const auto address =
            align(next_host, alignment);
        next_host =
            address +
            static_cast<std::uintptr_t>(
                bytes + alignment);
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
            allocation_calls == fail_after) {
            ++allocation_calls;
            throw std::runtime_error(
                "injected allocation failure");
        }
        ++allocation_calls;
    }

    static std::uintptr_t align(
        std::uintptr_t value,
        std::uint64_t alignment) {
        const auto remainder =
            value %
            static_cast<std::uintptr_t>(
                alignment);
        return remainder == 0
            ? value
            : value +
                  static_cast<std::uintptr_t>(
                      alignment - remainder);
    }
};

strix::CompiledSerialExecution fake_compiled() {
    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x1234ULL;
    compiled.arena.alignment = 256;
    compiled.arena.total_bytes = 4096;
    return compiled;
}

strix::SerialSessionStateLayout fake_state_layout() {
    strix::SerialSessionStateLayout layout;
    layout.layout_identity =
        0x5678ULL;
    layout.device_alignment = 256;
    layout.host_alignment = 64;
    layout.device_total_bytes = 8192;
    layout.host_total_bytes = 128;
    return layout;
}

}  // namespace

int main() {
    auto compiled = fake_compiled();
    auto layout = fake_state_layout();

    FakeAllocator allocator;
    strix::AllocationGenerationSource generations;

    {
        auto owner =
            strix::RuntimeExecutionMemoryOwner::allocate(
                compiled,
                layout,
                allocator,
                generations);

        expect(owner.valid(),
               "runtime memory owner becomes valid after allocation");
        expect(owner.scratch().arena_bytes ==
                   compiled.arena.total_bytes &&
                   owner.scratch().allocation_generation == 1,
               "scratch allocation receives first generation");
        expect(owner.state().device_allocation_generation == 2 &&
                   owner.state().host_allocation_generation == 3,
               "state pools receive independent allocation generations");

        auto moved = std::move(owner);
        expect(!owner.valid() && moved.valid(),
               "runtime memory owner transfers allocation ownership by move");
    }

    expect(allocator.device_frees == 2 &&
               allocator.host_frees == 1,
           "runtime owner releases scratch and state pools exactly once");

    FakeAllocator failing;
    failing.fail_after = 1;
    strix::AllocationGenerationSource failure_generations;
    bool failed = false;
    try {
        static_cast<void>(
            strix::RuntimeExecutionMemoryOwner::allocate(
                compiled,
                layout,
                failing,
                failure_generations));
    } catch (const std::runtime_error&) {
        failed = true;
    }
    expect(failed,
           "allocation failure propagates");
    expect(failing.device_frees == 1 &&
               failing.host_frees == 0,
           "partial allocation failure releases already-owned scratch");

    bool zero_generation_rejected = false;
    try {
        strix::AllocationGenerationSource invalid(0);
        static_cast<void>(invalid);
    } catch (const std::invalid_argument&) {
        zero_generation_rejected = true;
    }
    expect(zero_generation_rejected,
           "allocation generation source rejects zero start");

    if (failures == 0) {
        std::cout << "runtime allocator tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
