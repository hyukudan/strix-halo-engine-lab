#pragma once

#include "strix/serial_session.hpp"
#include "strix/session_restore.hpp"
#include "strix/session_snapshot.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>
#include <span>

namespace strix {

struct SessionSnapshotMemoryBinding {
    std::uint64_t snapshot_plan_identity = 0;

    std::uintptr_t device_section_base = 0;
    std::uint64_t device_section_bytes = 0;

    std::uintptr_t host_section_base = 0;
    std::uint64_t host_section_bytes = 0;

    std::uint64_t allocation_generation = 0;
};

struct SessionSnapshotTransfer {
    SessionStateStorage storage;
    std::uintptr_t source;
    std::uintptr_t destination;
    std::uint64_t bytes;
};

class SessionSnapshotTransferBackend {
public:
    virtual ~SessionSnapshotTransferBackend() = default;

    // Every range is resolved and admitted before either batch method is called.
    // A batch override may enqueue copies together, but MUST complete/drain all
    // submitted work before returning OR throwing: payload owners may unwind
    // immediately afterward. Restore metadata is published only after return.
    // Defaults preserve the individual synchronous backend contract.
    virtual void capture_batch(std::span<const SessionSnapshotTransfer> transfers);
    virtual void restore_batch(std::span<const SessionSnapshotTransfer> transfers);

    virtual void capture_device(
        std::uintptr_t source_device,
        std::uintptr_t destination_snapshot,
        std::uint64_t bytes) = 0;

    virtual void capture_host(
        std::uintptr_t source_host,
        std::uintptr_t destination_snapshot,
        std::uint64_t bytes) = 0;

    virtual void restore_device(
        std::uintptr_t source_snapshot,
        std::uintptr_t destination_device,
        std::uint64_t bytes) = 0;

    virtual void restore_host(
        std::uintptr_t source_snapshot,
        std::uintptr_t destination_host,
        std::uint64_t bytes) = 0;
};

void validate_snapshot_memory_binding(
    const SerialSessionSnapshotPlan& snapshot,
    const SessionSnapshotMemoryBinding& memory);

void capture_serial_session_snapshot(
    const SerialSessionSnapshotPlan& snapshot,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend,
    // Required only for occupied_kv_v1: verifies canonical layer geometry and
    // the complete range census before submitting any backend work.
    const ModelContract* compact_model = nullptr);

void restore_serial_session_snapshot(
    SerialSessionContext& session,
    const SerialSessionRestorePlan& restore,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend,
    const ModelContract* compact_model = nullptr);

}  // namespace strix
