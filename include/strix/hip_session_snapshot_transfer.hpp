#pragma once

#include "strix/session_snapshot_transfer.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <memory>

namespace strix {

class HipSessionSnapshotTransferBackend final
    : public SessionSnapshotTransferBackend {
public:
    HipSessionSnapshotTransferBackend() = default;
    // Explicit borrowed exclusive stream. Its consumer retains the stream and
    // both payload owners through every synchronous batch call. The guard is
    // retained here for original-device completion and failure recovery.
    HipSessionSnapshotTransferBackend(hipStream_t owned_stream,
        std::shared_ptr<const HipSetupQuiescenceGuard> original_device);
    void capture_batch(std::span<const SessionSnapshotTransfer>) override;
    void restore_batch(std::span<const SessionSnapshotTransfer>) override;
    void capture_device(
        std::uintptr_t source_device,
        std::uintptr_t destination_snapshot,
        std::uint64_t bytes) override;

    void capture_host(
        std::uintptr_t source_host,
        std::uintptr_t destination_snapshot,
        std::uint64_t bytes) override;

    void restore_device(
        std::uintptr_t source_snapshot,
        std::uintptr_t destination_device,
        std::uint64_t bytes) override;

    void restore_host(
        std::uintptr_t source_snapshot,
        std::uintptr_t destination_host,
        std::uint64_t bytes) override;
private:
    void copy_batch(std::span<const SessionSnapshotTransfer>);
    hipStream_t stream_ = nullptr;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard_;
};

}  // namespace strix
