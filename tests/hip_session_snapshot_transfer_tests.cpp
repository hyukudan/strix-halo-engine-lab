#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/execution_telemetry.hpp"
#include <array>
#include <limits>
#include <memory>

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

    {
        hipStream_t stream=nullptr;
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create snapshot batch stream");
        auto guard=std::make_shared<strix::HipSetupQuiescenceGuard>();
        strix::HipSessionSnapshotTransferBackend batched(stream,guard);
        std::array<strix::SessionSnapshotTransfer,7> capture{},restore{};
        for(std::size_t i=0;i<6U;++i) {
            const auto offset=i*(bytes/6U);
            const auto extent=i==5U?bytes-offset:bytes/6U;
            capture[i]={strix::SessionStateStorage::device,reinterpret_cast<std::uintptr_t>(device_source)+offset,
                reinterpret_cast<std::uintptr_t>(device_snapshot)+offset,extent};
            restore[i]={strix::SessionStateStorage::device,reinterpret_cast<std::uintptr_t>(device_snapshot)+offset,
                reinterpret_cast<std::uintptr_t>(device_destination)+offset,extent};
        }
        capture[6]={strix::SessionStateStorage::host,reinterpret_cast<std::uintptr_t>(host_source.data()),
            reinterpret_cast<std::uintptr_t>(host_snapshot.data()),bytes};
        restore[6]={strix::SessionStateStorage::host,reinterpret_cast<std::uintptr_t>(host_snapshot.data()),
            reinterpret_cast<std::uintptr_t>(host_destination.data()),bytes};
        std::fill(host_destination.begin(),host_destination.end(),0U);
        check(hipMemset(device_destination,0,bytes),"clear batched destination");
        strix::HostDispatchCounters counters;
        {
            strix::ExecutionTelemetryBinding observed(&counters);
            batched.capture_batch(capture);
            batched.restore_batch(restore);
        }
        check(hipMemcpy(device_check.data(),device_destination,bytes,hipMemcpyDeviceToHost),"read batched restored state");
        expect(device_check==host_source && host_destination==host_source,
            "owned-stream batches complete every host/device range before return");
        const auto calls=[&](strix::ExecutionActivityKind kind) {return counters.activities()[static_cast<std::size_t>(kind)].calls;};
        expect(calls(strix::ExecutionActivityKind::copy_async)==12U &&
            calls(strix::ExecutionActivityKind::stream_sync)==2U &&
            calls(strix::ExecutionActivityKind::copy_sync)==0U && calls(strix::ExecutionActivityKind::device_sync)==0U,
            "six component copies per batch use one stream fence and no blocking global copy");
        auto bad=restore;bad.back().storage=static_cast<strix::SessionStateStorage>(99U);
        strix::HostDispatchCounters untouched;
        bool rejected=false;
        try {strix::ExecutionTelemetryBinding observed(&untouched);batched.restore_batch(bad);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected && untouched.activities()[static_cast<std::size_t>(strix::ExecutionActivityKind::copy_async)].calls==0U,
            "invalid final batch range is rejected before submitting the first device copy");
        bad=restore;bad.back().source=std::numeric_limits<std::uintptr_t>::max();
        rejected=false;try{batched.restore_batch(bad);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"batch preflight rejects overflowing host address before enqueue");
        check(hipStreamDestroy(stream),"destroy completed snapshot batch stream");
    }

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
