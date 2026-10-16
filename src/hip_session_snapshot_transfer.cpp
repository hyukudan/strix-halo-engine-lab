#include "strix/hip_session_snapshot_transfer.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include "strix/hip_launch_context.hpp"

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

HipSessionSnapshotTransferBackend::HipSessionSnapshotTransferBackend(hipStream_t stream,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard) : stream_(stream),guard_(std::move(guard)) {
    if(!guard_ || !stream_ || stream_==hipStreamLegacy || stream_==hipStreamPerThread)
        throw std::invalid_argument("snapshot batches require an exclusive retained eager stream");
    guard_->require_original();
}
void HipSessionSnapshotTransferBackend::capture_batch(std::span<const SessionSnapshotTransfer> transfers) {
    if(!stream_) {SessionSnapshotTransferBackend::capture_batch(transfers);return;}
    copy_batch(transfers);
}
void HipSessionSnapshotTransferBackend::restore_batch(std::span<const SessionSnapshotTransfer> transfers) {
    if(!stream_) {SessionSnapshotTransferBackend::restore_batch(transfers);return;}
    copy_batch(transfers);
}
void HipSessionSnapshotTransferBackend::copy_batch(std::span<const SessionSnapshotTransfer> transfers) {
    if(current_hip_launch_context()) throw std::invalid_argument("snapshot batch cannot borrow an active launch scope");
    guard_->require_original();
    hipStreamCaptureStatus captured=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(stream_,&captured),"query snapshot batch capture");
    if(captured!=hipStreamCaptureStatusNone) throw std::invalid_argument("snapshot batch cannot enter stream capture");
    for(const auto& t:transfers) {
        validate_range(t.source,t.destination,t.bytes);
        if((t.storage!=SessionStateStorage::device && t.storage!=SessionStateStorage::host) ||
            t.bytes>std::numeric_limits<std::uintptr_t>::max()-t.source ||
            t.bytes>std::numeric_limits<std::uintptr_t>::max()-t.destination)
            throw std::invalid_argument("snapshot batch storage or address extent is invalid");
    }
    try {
        bool queued=false;
        for(const auto& t:transfers) {
            if(!t.bytes) continue;
            if(t.storage==SessionStateStorage::host) {copy_host(t.source,t.destination,t.bytes);continue;}
            check_hip(hipMemcpyAsync(reinterpret_cast<void*>(t.destination),reinterpret_cast<const void*>(t.source),
                static_cast<std::size_t>(t.bytes),hipMemcpyDeviceToDevice,stream_),"enqueue snapshot batch D2D");
            queued=true;
        }
        if(queued) check_hip(hipStreamSynchronize(stream_),"complete snapshot batch D2D");
        guard_->require_original();
    } catch(...) {guard_->recover_or_fail_stop();throw;}
}

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
