#include "strix/session_snapshot_transfer.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

void validate_section(
    std::uintptr_t base,
    std::uint64_t available,
    std::uint64_t required,
    const char* name) {
    if (required == 0) {
        return;
    }
    if (base == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " snapshot section has null base");
    }
    if (available < required) {
        throw std::invalid_argument(
            std::string(name) +
            " snapshot section is undersized");
    }
    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (required >
        static_cast<std::uint64_t>(
            maximum - base)) {
        throw std::overflow_error(
            std::string(name) +
            " snapshot section address range overflows uintptr_t");
    }
}

std::uintptr_t section_address(
    const SessionSnapshotMemoryBinding& memory,
    SessionStateStorage storage,
    std::uint64_t offset,
    std::uint64_t bytes) {
    const auto base =
        storage ==
                SessionStateStorage::device
            ? memory.device_section_base
            : memory.host_section_base;
    const auto available =
        storage ==
                SessionStateStorage::device
            ? memory.device_section_bytes
            : memory.host_section_bytes;

    if (offset > available ||
        bytes > available - offset) {
        throw std::invalid_argument(
            "snapshot range exceeds bound payload section");
    }
    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (offset >
        static_cast<std::uint64_t>(
            maximum - base)) {
        throw std::overflow_error(
            "snapshot payload address overflows uintptr_t");
    }
    return base +
           static_cast<std::uintptr_t>(
               offset);
}

}  // namespace

void validate_snapshot_memory_binding(
    const SerialSessionSnapshotPlan& snapshot,
    const SessionSnapshotMemoryBinding& memory) {
    if (snapshot.plan_identity == 0 ||
        memory.snapshot_plan_identity !=
            snapshot.plan_identity) {
        throw std::invalid_argument(
            "snapshot memory identity does not match snapshot plan");
    }
    if (memory.allocation_generation == 0) {
        throw std::invalid_argument(
            "snapshot allocation generation must be non-zero");
    }

    validate_section(
        memory.device_section_base,
        memory.device_section_bytes,
        snapshot.device_bytes,
        "device");
    validate_section(
        memory.host_section_base,
        memory.host_section_bytes,
        snapshot.host_bytes,
        "host");
}

void capture_serial_session_snapshot(
    const SerialSessionSnapshotPlan& snapshot,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend) {
    validate_snapshot_memory_binding(
        snapshot,
        snapshot_memory);
    validate_session_state_memory_binding(
        state_layout,
        state_memory);

    if (snapshot.state_layout_identity !=
        state_layout.layout_identity) {
        throw std::invalid_argument(
            "snapshot plan state layout does not match source state");
    }

    for (const auto& range :
         snapshot.ranges) {
        const auto source =
            resolve_session_state(
                state_layout,
                state_memory,
                range.component);
        if (source.storage !=
                range.storage ||
            range.bytes >
                source.bytes) {
            throw std::invalid_argument(
                "snapshot capture range does not match state component");
        }

        if (range.source_offset <
            (source.address -
             (source.storage ==
                      SessionStateStorage::device
                  ? state_memory.device_base
                  : state_memory.host_base))) {
            throw std::invalid_argument(
                "snapshot source offset precedes component allocation");
        }

        const auto component_offset =
            range.source_offset -
            (source.storage ==
                     SessionStateStorage::device
                 ? find_session_state_allocation(
                       state_layout,
                       range.component)
                       ->offset
                 : find_session_state_allocation(
                       state_layout,
                       range.component)
                       ->offset);
        if (component_offset != 0) {
            throw std::invalid_argument(
                "snapshot capture currently requires component-prefix ranges");
        }

        const auto destination =
            section_address(
                snapshot_memory,
                range.storage,
                range.destination_offset,
                range.bytes);

        if (range.storage ==
            SessionStateStorage::device) {
            backend.capture_device(
                source.address,
                destination,
                range.bytes);
        } else {
            backend.capture_host(
                source.address,
                destination,
                range.bytes);
        }
    }
}

void restore_serial_session_snapshot(
    SerialSessionContext& session,
    const SerialSessionRestorePlan& restore,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend) {
    if (session.execution_identity == 0 ||
        session.execution_identity !=
            restore.compiled_execution_identity ||
        session.runtime_compatibility_identity == 0 ||
        session.state_layout_identity == 0) {
        throw std::invalid_argument(
            "restore requires initialized serial session identities");
    }
    if (restore.runtime_compatibility_identity !=
            session.runtime_compatibility_identity ||
        restore.state_layout_identity !=
            session.state_layout_identity ||
        restore.state_layout_identity !=
            state_layout.layout_identity) {
        throw std::invalid_argument(
            "restore plan is incompatible with serial session");
    }
    if (snapshot_memory.snapshot_plan_identity == 0 ||
        snapshot_memory.snapshot_plan_identity !=
            restore.snapshot_plan_identity ||
        snapshot_memory.allocation_generation == 0) {
        throw std::invalid_argument(
            "restore requires initialized snapshot memory");
    }
    validate_session_state_memory_binding(
        state_layout,
        state_memory);

    if (snapshot_memory.device_section_bytes <
            restore.snapshot_device_bytes ||
        snapshot_memory.host_section_bytes <
            restore.snapshot_host_bytes) {
        throw std::invalid_argument(
            "snapshot memory is smaller than restore payload");
    }

    try {
        for (const auto& range :
             restore.ranges) {
            const auto destination =
                resolve_session_state(
                    state_layout,
                    state_memory,
                    range.component);
            if (destination.storage !=
                    range.storage ||
                range.bytes >
                    destination.bytes) {
                throw std::invalid_argument(
                    "restore destination range does not match state component");
            }

            const auto source =
                section_address(
                    snapshot_memory,
                    range.storage,
                    range.snapshot_offset,
                    range.bytes);
            const auto expected_destination =
                (range.storage ==
                         SessionStateStorage::device
                     ? state_memory.device_base
                     : state_memory.host_base) +
                static_cast<std::uintptr_t>(
                    range.destination_offset);
            if (destination.address !=
                expected_destination) {
                throw std::invalid_argument(
                    "restore destination offset does not match bound state layout");
            }

            if (range.storage ==
                SessionStateStorage::device) {
                backend.restore_device(
                    source,
                    destination.address,
                    range.bytes);
            } else {
                backend.restore_host(
                    source,
                    destination.address,
                    range.bytes);
            }
        }
    } catch (...) {
        session.health =
            SerialSessionHealth::poisoned;
        throw;
    }

    session.next_position =
        restore.next_position;
    session.committed_tokens =
        restore.committed_tokens;
    session.state =
        restore.state;
    session.health =
        SerialSessionHealth::ready;
}

}  // namespace strix
