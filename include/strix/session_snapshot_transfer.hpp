#pragma once

#include "strix/serial_session.hpp"
#include "strix/session_restore.hpp"
#include "strix/session_snapshot.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>

namespace strix {

struct SessionSnapshotMemoryBinding {
    std::uint64_t snapshot_plan_identity = 0;

    std::uintptr_t device_section_base = 0;
    std::uint64_t device_section_bytes = 0;

    std::uintptr_t host_section_base = 0;
    std::uint64_t host_section_bytes = 0;

    std::uint64_t allocation_generation = 0;
};

class SessionSnapshotTransferBackend {
public:
    virtual ~SessionSnapshotTransferBackend() = default;

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
    SessionSnapshotTransferBackend& backend);

void restore_serial_session_snapshot(
    SerialSessionContext& session,
    const SerialSessionRestorePlan& restore,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend);

}  // namespace strix
