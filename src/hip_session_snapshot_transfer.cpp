#include "strix/hip_session_snapshot_transfer.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix {
namespace {

void validate_range(
    std::uintptr_t source,
    std::uintptr_t destination,
    std::uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (source == 0 ||
        destination == 0) {
        throw std::invalid_argument(
            "HIP snapshot transfer requires non-null addresses");
    }
    if (bytes >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "HIP snapshot transfer size does not fit size_t");
    }
}

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

void copy_device(
    std::uintptr_t source,
    std::uintptr_t destination,
    std::uint64_t bytes,
    const char* operation) {
    validate_range(
        source,
        destination,
        bytes);
    if (bytes == 0) {
        return;
    }
    check_hip(
        hipMemcpy(
            reinterpret_cast<void*>(
                destination),
            reinterpret_cast<const void*>(
                source),
            static_cast<std::size_t>(
                bytes),
            hipMemcpyDeviceToDevice),
        operation);
}

void copy_host(
    std::uintptr_t source,
    std::uintptr_t destination,
    std::uint64_t bytes) {
    validate_range(
        source,
        destination,
        bytes);
    if (bytes == 0) {
        return;
    }
    std::memcpy(
        reinterpret_cast<void*>(
            destination),
        reinterpret_cast<const void*>(
            source),
        static_cast<std::size_t>(
            bytes));
}

}  // namespace

void HipSessionSnapshotTransferBackend::capture_device(
    std::uintptr_t source_device,
    std::uintptr_t destination_snapshot,
    std::uint64_t bytes) {
    copy_device(
        source_device,
        destination_snapshot,
        bytes,
        "hipMemcpy(snapshot capture D2D)");
}

void HipSessionSnapshotTransferBackend::capture_host(
    std::uintptr_t source_host,
    std::uintptr_t destination_snapshot,
    std::uint64_t bytes) {
    copy_host(
        source_host,
        destination_snapshot,
        bytes);
}

void HipSessionSnapshotTransferBackend::restore_device(
    std::uintptr_t source_snapshot,
    std::uintptr_t destination_device,
    std::uint64_t bytes) {
    copy_device(
        source_snapshot,
        destination_device,
        bytes,
        "hipMemcpy(snapshot restore D2D)");
}

void HipSessionSnapshotTransferBackend::restore_host(
    std::uintptr_t source_snapshot,
    std::uintptr_t destination_host,
    std::uint64_t bytes) {
    copy_host(
        source_snapshot,
        destination_host,
        bytes);
}

}  // namespace strix
