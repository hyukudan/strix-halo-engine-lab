#include "strix/hip_runtime_allocator.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            hipGetErrorString(error));
    }
}
template<class F> bool rejects_description(F&& f) {
    try {f();} catch(const std::invalid_argument&) {return true;}
    return false;
}
void allocation_descriptions() {
    using namespace strix;
    HipRuntimeAllocator allocator,foreign;
    const auto device=allocator.allocate_device(65U,256U);
    const auto first=allocator.describe_allocation(device,65U,HipRuntimeAllocationSpace::device);
    expect(first.lease_identity && first.raw_bytes==320U && first.view_bytes==65U &&
        first.view_offset<256U && first.raw_base+first.view_offset==device,
        "actual device description retains the complete alignment reservation and returned view");
    expect(allocator.describe_allocation(device,65U,HipRuntimeAllocationSpace::device)==first,
        "describing a live allocation does not mint a new lease");
    expect(rejects_description([&] {(void)foreign.describe_allocation(device,65U,HipRuntimeAllocationSpace::device);}) &&
        rejects_description([&] {(void)allocator.describe_allocation(device,64U,HipRuntimeAllocationSpace::device);}) &&
        rejects_description([&] {(void)allocator.describe_allocation(device+1U,64U,HipRuntimeAllocationSpace::device);}) &&
        rejects_description([&] {(void)allocator.describe_allocation(device,65U,HipRuntimeAllocationSpace::host);}) &&
        rejects_description([&] {(void)allocator.describe_allocation(device,65U,static_cast<HipRuntimeAllocationSpace>(99));}),
        "foreign allocator, shortened/interior view, wrong space and unknown enum all reject");
    allocator.free_device(device,65U);
    expect(rejects_description([&] {(void)allocator.describe_allocation(device,65U,HipRuntimeAllocationSpace::device);}),
        "freed device allocation no longer supplies a live lease");
    const auto replacement=allocator.allocate_device(65U,256U);
    expect(allocator.describe_allocation(replacement,65U,HipRuntimeAllocationSpace::device).lease_identity!=first.lease_identity,
        "a new allocation gets a new lease even if the allocator reuses an address");
    allocator.free_device(replacement,65U);
    for(const auto mode:{HipHostAllocationMode::pageable_aligned,HipHostAllocationMode::pinned_mapped}) {
        HipRuntimeAllocator host(mode);
        const auto address=host.allocate_host(37U,64U);
        const auto view=host.describe_allocation(address,37U,HipRuntimeAllocationSpace::host);
        expect(view.lease_identity && view.view_bytes==37U && view.raw_base+view.view_offset==address &&
            view.raw_bytes==(mode==HipHostAllocationMode::pageable_aligned?37U:100U),
            "host description distinguishes exact pageable payload from pinned alignment backing");
        host.free_host(address,37U);
        expect(rejects_description([&] {(void)host.describe_allocation(address,37U,HipRuntimeAllocationSpace::host);}),
            "freed host allocation no longer supplies a live lease");
    }
}

}  // namespace

int main() try {
    allocation_descriptions();
    constexpr std::uint64_t scratch_bytes = 4096;
    constexpr std::uint64_t state_bytes = 8192;
    constexpr std::uint64_t host_bytes = 256;

    void* scratch_device = nullptr;
    void* state_device = nullptr;
    check(
        hipMalloc(
            &scratch_device,
            scratch_bytes),
        "hipMalloc scratch");
    check(
        hipMalloc(
            &state_device,
            state_bytes),
        "hipMalloc state");

    check(
        hipMemset(
            scratch_device,
            0x7f,
            scratch_bytes),
        "prime scratch");
    check(
        hipMemset(
            state_device,
            0x55,
            state_bytes),
        "prime state");

    std::vector<std::uint8_t> host(
        host_bytes,
        std::uint8_t{0xa5});

    strix::ExecutionMemoryBinding scratch;
    scratch.execution_identity = 1;
    scratch.arena_base =
        reinterpret_cast<std::uintptr_t>(
            scratch_device);
    scratch.arena_bytes =
        scratch_bytes;
    scratch.allocation_generation = 1;

    strix::SessionStateMemoryBinding state;
    state.layout_identity = 1;
    state.device_base =
        reinterpret_cast<std::uintptr_t>(
            state_device);
    state.device_bytes =
        state_bytes;
    state.device_allocation_generation = 2;
    state.host_base =
        reinterpret_cast<std::uintptr_t>(
            host.data());
    state.host_bytes =
        host_bytes;
    state.host_allocation_generation = 3;

    strix::initialize_hip_fresh_runtime_memory(
        scratch,
        state);

    std::vector<std::uint8_t> scratch_check(
        scratch_bytes);
    std::vector<std::uint8_t> state_check(
        state_bytes);

    check(
        hipMemcpy(
            scratch_check.data(),
            scratch_device,
            scratch_bytes,
            hipMemcpyDeviceToHost),
        "copy scratch");
    check(
        hipMemcpy(
            state_check.data(),
            state_device,
            state_bytes,
            hipMemcpyDeviceToHost),
        "copy state");

    expect(
        std::all_of(
            scratch_check.begin(),
            scratch_check.end(),
            [](std::uint8_t value) {
                return value == 0;
            }),
        "fresh runtime initialization clears scratch");
    expect(
        std::all_of(
            state_check.begin(),
            state_check.end(),
            [](std::uint8_t value) {
                return value == 0;
            }),
        "fresh runtime initialization clears device state");
    expect(
        std::all_of(
            host.begin(),
            host.end(),
            [](std::uint8_t value) {
                return value == 0;
            }),
        "fresh runtime initialization clears host state");

    check(
        hipFree(state_device),
        "hipFree state");
    check(
        hipFree(scratch_device),
        "hipFree scratch");

    if (failures == 0) {
        std::cout
            << "HIP fresh runtime memory tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP fresh runtime memory tests: "
        << error.what() << '\n';
    return 2;
}
