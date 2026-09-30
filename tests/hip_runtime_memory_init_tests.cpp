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

}  // namespace

int main() try {
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
