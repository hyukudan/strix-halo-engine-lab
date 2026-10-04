#pragma once

#include "strix/session_snapshot_transfer.hpp"

namespace strix {

class HipSessionSnapshotTransferBackend final
    : public SessionSnapshotTransferBackend {
public:
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
};

}  // namespace strix
