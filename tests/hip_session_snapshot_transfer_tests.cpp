#include "strix/hip_session_snapshot_transfer.hpp"

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

void check(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

}  // namespace

int main() try {
    constexpr std::size_t bytes = 4096;

    std::vector<std::uint8_t> host_source(
        bytes);
    for (std::size_t index = 0;
         index < host_source.size();
         ++index) {
        host_source[index] =
            static_cast<std::uint8_t>(
                (index * 17U + 3U) &
                0xffU);
    }

    void* device_source = nullptr;
    void* device_snapshot = nullptr;
    void* device_destination = nullptr;
    check(
        hipMalloc(
            &device_source,
            bytes),
        "hipMalloc source");
    check(
        hipMalloc(
            &device_snapshot,
            bytes),
        "hipMalloc snapshot");
    check(
        hipMalloc(
            &device_destination,
            bytes),
        "hipMalloc destination");

    check(
        hipMemcpy(
            device_source,
            host_source.data(),
            bytes,
            hipMemcpyHostToDevice),
        "prime source");

    strix::HipSessionSnapshotTransferBackend
        backend;
    backend.capture_device(
        reinterpret_cast<std::uintptr_t>(
            device_source),
        reinterpret_cast<std::uintptr_t>(
            device_snapshot),
        bytes);

    check(
        hipMemset(
            device_destination,
            0,
            bytes),
        "clear destination");

    backend.restore_device(
        reinterpret_cast<std::uintptr_t>(
            device_snapshot),
        reinterpret_cast<std::uintptr_t>(
            device_destination),
        bytes);

    std::vector<std::uint8_t> device_check(
        bytes);
    check(
        hipMemcpy(
            device_check.data(),
            device_destination,
            bytes,
            hipMemcpyDeviceToHost),
        "copy restored device");

    expect(
        device_check ==
            host_source,
        "HIP snapshot transfer round-trips device bytes");

    std::vector<std::uint8_t> host_snapshot(
        bytes);
    std::vector<std::uint8_t> host_destination(
        bytes,
        0);

    backend.capture_host(
        reinterpret_cast<std::uintptr_t>(
            host_source.data()),
        reinterpret_cast<std::uintptr_t>(
            host_snapshot.data()),
        bytes);
    backend.restore_host(
        reinterpret_cast<std::uintptr_t>(
            host_snapshot.data()),
        reinterpret_cast<std::uintptr_t>(
            host_destination.data()),
        bytes);

    expect(
        host_destination ==
            host_source,
        "HIP snapshot transfer round-trips host bytes");

    bool null_rejected = false;
    try {
        backend.capture_device(
            0,
            reinterpret_cast<std::uintptr_t>(
                device_snapshot),
            1);
    } catch (const std::invalid_argument&) {
        null_rejected = true;
    }
    expect(
        null_rejected,
        "HIP snapshot transfer rejects null non-empty range");

    check(
        hipFree(
            device_destination),
        "hipFree destination");
    check(
        hipFree(
            device_snapshot),
        "hipFree snapshot");
    check(
        hipFree(
            device_source),
        "hipFree source");

    if (failures == 0) {
        std::cout
            << "HIP session snapshot transfer tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP session snapshot transfer tests: "
        << error.what() << '\n';
    return 2;
}
