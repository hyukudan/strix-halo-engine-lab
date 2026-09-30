#pragma once

#include "strix/runtime_allocator.hpp"
#include "strix/session_snapshot.hpp"
#include "strix/session_snapshot_transfer.hpp"

namespace strix {

class SessionSnapshotMemoryOwner {
public:
    SessionSnapshotMemoryOwner() = default;
    SessionSnapshotMemoryOwner(
        const SessionSnapshotMemoryOwner&) = delete;
    SessionSnapshotMemoryOwner& operator=(
        const SessionSnapshotMemoryOwner&) = delete;

    SessionSnapshotMemoryOwner(
        SessionSnapshotMemoryOwner&& other) noexcept;
    SessionSnapshotMemoryOwner& operator=(
        SessionSnapshotMemoryOwner&& other) noexcept;

    ~SessionSnapshotMemoryOwner();

    [[nodiscard]] static SessionSnapshotMemoryOwner allocate(
        const SerialSessionSnapshotPlan& snapshot,
        RuntimeMemoryAllocator& allocator,
        AllocationGenerationSource& generations);

    [[nodiscard]] const SessionSnapshotMemoryBinding&
    binding() const noexcept {
        return binding_;
    }

    [[nodiscard]] bool valid() const noexcept {
        return allocator_ != nullptr;
    }

private:
    void release() noexcept;

    RuntimeMemoryAllocator* allocator_ = nullptr;
    SessionSnapshotMemoryBinding binding_;
};

}  // namespace strix
