#include "strix/session_snapshot_memory.hpp"

#include <stdexcept>
#include <utility>

namespace strix {

SessionSnapshotMemoryOwner::SessionSnapshotMemoryOwner(
    SessionSnapshotMemoryOwner&& other) noexcept
    : allocator_(other.allocator_),
      binding_(other.binding_) {
    other.allocator_ = nullptr;
    other.binding_ = {};
}

SessionSnapshotMemoryOwner&
SessionSnapshotMemoryOwner::operator=(
    SessionSnapshotMemoryOwner&& other) noexcept {
    if (this != &other) {
        release();
        allocator_ = other.allocator_;
        binding_ = other.binding_;
        other.allocator_ = nullptr;
        other.binding_ = {};
    }
    return *this;
}

SessionSnapshotMemoryOwner::~SessionSnapshotMemoryOwner() {
    release();
}

SessionSnapshotMemoryOwner
SessionSnapshotMemoryOwner::allocate(
    const SerialSessionSnapshotPlan& snapshot,
    RuntimeMemoryAllocator& allocator,
    AllocationGenerationSource& generations) {
    if (snapshot.plan_identity == 0 ||
        snapshot.alignment == 0) {
        throw std::invalid_argument(
            "snapshot memory allocation requires a valid snapshot plan");
    }

    SessionSnapshotMemoryOwner owner;
    owner.allocator_ = &allocator;
    owner.binding_.snapshot_plan_identity =
        snapshot.plan_identity;
    owner.binding_.allocation_generation =
        generations.next();

    try {
        if (snapshot.device_bytes != 0) {
            owner.binding_.device_section_bytes =
                snapshot.device_bytes;
            owner.binding_.device_section_base =
                allocator.allocate_device(
                    snapshot.device_bytes,
                    snapshot.alignment);
        }

        if (snapshot.host_bytes != 0) {
            owner.binding_.host_section_bytes =
                snapshot.host_bytes;
            owner.binding_.host_section_base =
                allocator.allocate_host(
                    snapshot.host_bytes,
                    snapshot.alignment);
        }

        validate_snapshot_memory_binding(
            snapshot,
            owner.binding_);
    } catch (...) {
        owner.release();
        throw;
    }

    return owner;
}

void SessionSnapshotMemoryOwner::release() noexcept {
    if (allocator_ == nullptr) {
        return;
    }

    if (binding_.host_section_base != 0) {
        allocator_->free_host(
            binding_.host_section_base,
            binding_.host_section_bytes);
    }
    if (binding_.device_section_base != 0) {
        allocator_->free_device(
            binding_.device_section_base,
            binding_.device_section_bytes);
    }

    allocator_ = nullptr;
    binding_ = {};
}

}  // namespace strix
