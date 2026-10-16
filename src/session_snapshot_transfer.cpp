#include "strix/session_snapshot_transfer.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix {
namespace {

void validate_transfer_extent(std::uintptr_t address, std::uint64_t bytes) {
    if (bytes && (!address || bytes > std::numeric_limits<std::uintptr_t>::max() - address))
        throw std::invalid_argument("snapshot transfer address extent is null or overflows");
}

void validate_storage(SessionStateStorage storage) {
    if (storage != SessionStateStorage::device && storage != SessionStateStorage::host)
        throw std::invalid_argument("snapshot transfer storage is unsupported");
}

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
    validate_storage(storage);
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
    const auto address = base +
           static_cast<std::uintptr_t>(
               offset);
    validate_transfer_extent(address, bytes);
    return address;
}

}  // namespace

void SessionSnapshotTransferBackend::capture_batch(std::span<const SessionSnapshotTransfer> transfers) {
    for (const auto& transfer : transfers) {
        validate_storage(transfer.storage);
        if (transfer.storage == SessionStateStorage::device)
            capture_device(transfer.source, transfer.destination, transfer.bytes);
        else capture_host(transfer.source, transfer.destination, transfer.bytes);
    }
}

void SessionSnapshotTransferBackend::restore_batch(std::span<const SessionSnapshotTransfer> transfers) {
    for (const auto& transfer : transfers) {
        validate_storage(transfer.storage);
        if (transfer.storage == SessionStateStorage::device)
            restore_device(transfer.source, transfer.destination, transfer.bytes);
        else restore_host(transfer.source, transfer.destination, transfer.bytes);
    }
}

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
    SessionSnapshotTransferBackend& backend,
    const ModelContract* compact_model) {
    // Authenticate the stored format before branching; a compact plan cannot
    // bypass canonical admission by changing its public policy field to full.
    validate_serial_session_snapshot_identity(snapshot);
    if (snapshot.policy == SessionSnapshotPolicy::occupied_kv_v1) {
        if (!compact_model) throw std::invalid_argument("occupied KV capture requires the canonical model contract");
        validate_serial_session_snapshot_plan(*compact_model, state_layout, snapshot);
    }
    validate_snapshot_memory_binding(
        snapshot,
        snapshot_memory);
    validate_session_state_memory_binding(
        state_layout,
        state_memory);

    if (snapshot.recurrent_storage != state_layout.recurrent_storage || snapshot.state_layout_identity !=
        state_layout.layout_identity) {
        throw std::invalid_argument(
            "snapshot plan state layout does not match source state");
    }

    // Resolve and validate every range before exposing either owner to a
    // potentially asynchronous backend. Allocation/validation exceptions here
    // cannot leave an earlier copy pending.
    std::vector<SessionSnapshotTransfer> transfers;
    transfers.reserve(snapshot.ranges.size());
    for (const auto& range :
         snapshot.ranges) {
        validate_storage(range.storage);
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

        const auto* allocation =
            find_session_state_allocation(
                state_layout,
                range.component);
        if (allocation == nullptr || range.source_offset < allocation->offset ||
            range.source_offset - allocation->offset > allocation->bytes ||
            range.bytes > allocation->bytes - (range.source_offset - allocation->offset) ||
            (snapshot.policy == SessionSnapshotPolicy::full_components &&
             range.source_offset != allocation->offset)) {
            throw std::invalid_argument(
                "snapshot capture currently requires component-prefix ranges");
        }

        const auto expected_source =
            (range.storage ==
                     SessionStateStorage::device
                 ? state_memory.device_base
                 : state_memory.host_base) +
            static_cast<std::uintptr_t>(
                range.source_offset);
        const auto relative = range.source_offset - allocation->offset;
        if (source.address > std::numeric_limits<std::uintptr_t>::max() - relative ||
            source.address + relative != expected_source) {
            throw std::invalid_argument(
                "snapshot source offset does not match bound state layout");
        }

        const auto destination =
            section_address(
                snapshot_memory,
                range.storage,
                range.destination_offset,
                range.bytes);
        validate_transfer_extent(expected_source, range.bytes);
        transfers.push_back({range.storage, expected_source, destination, range.bytes});
    }

    backend.capture_batch(transfers);
}

void restore_serial_session_snapshot(
    SerialSessionContext& session,
    const SerialSessionRestorePlan& restore,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend,
    const ModelContract* compact_model) {
    validate_serial_session_restore_identity(restore);
    if (restore.policy == SessionSnapshotPolicy::occupied_kv_v1) {
        if (!compact_model) throw std::invalid_argument("occupied KV restore requires the canonical model contract");
        SerialSessionSnapshotRequest request;
        request.compiled_execution_identity = restore.compiled_execution_identity;
        request.runtime_compatibility_identity = restore.runtime_compatibility_identity;
        request.next_position = restore.next_position;
        request.committed_tokens = restore.committed_tokens;
        request.state = restore.state;
        request.policy = restore.policy;
        // The compact payload alignment is part of both identities; carry it
        // explicitly rather than guessing it from the first range offset.
        request.alignment = restore.snapshot_alignment;
        const auto snapshot = build_serial_session_snapshot_plan(*compact_model, state_layout, request);
        const auto expected = build_serial_session_restore_plan(*compact_model, state_layout,
            restore.runtime_compatibility_identity, snapshot);
        if (restore.plan_identity != expected.plan_identity ||
            restore.snapshot_plan_identity != expected.snapshot_plan_identity ||
            restore.snapshot_device_bytes != expected.snapshot_device_bytes ||
            restore.snapshot_host_bytes != expected.snapshot_host_bytes ||
            restore.recurrent_storage != expected.recurrent_storage ||
            restore.ranges.size() != expected.ranges.size())
            throw std::invalid_argument("occupied KV restore structure or identity is invalid");
        for (std::size_t i = 0; i < expected.ranges.size(); ++i) {
            const auto& a = restore.ranges[i]; const auto& b = expected.ranges[i];
            if (a.component != b.component || a.storage != b.storage || a.snapshot_offset != b.snapshot_offset ||
                a.destination_offset != b.destination_offset || a.bytes != b.bytes)
                throw std::invalid_argument("occupied KV restore range is not canonical");
        }
    }
    if (session.execution_identity == 0 ||
        session.execution_identity !=
            restore.compiled_execution_identity ||
        session.runtime_compatibility_identity == 0 ||
        session.state_layout_identity == 0) {
        throw std::invalid_argument(
            "restore requires initialized serial session identities");
    }
    if (restore.recurrent_storage != state_layout.recurrent_storage || restore.runtime_compatibility_identity !=
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

    validate_section(snapshot_memory.device_section_base,
        snapshot_memory.device_section_bytes, restore.snapshot_device_bytes, "device");
    validate_section(snapshot_memory.host_section_base,
        snapshot_memory.host_section_bytes, restore.snapshot_host_bytes, "host");

    std::vector<SessionSnapshotTransfer> transfers;
    transfers.reserve(restore.ranges.size());
    for (const auto& range : restore.ranges) {
        validate_storage(range.storage);
        const auto destination = resolve_session_state(
            state_layout, state_memory, range.component);
        if (destination.storage != range.storage || range.bytes > destination.bytes) {
            throw std::invalid_argument(
                "restore destination range does not match state component");
        }
        const auto source = section_address(
            snapshot_memory, range.storage, range.snapshot_offset, range.bytes);
        const auto* allocation = find_session_state_allocation(state_layout, range.component);
        if (!allocation || range.destination_offset < allocation->offset ||
            range.destination_offset - allocation->offset > allocation->bytes ||
            range.bytes > allocation->bytes - (range.destination_offset - allocation->offset) ||
            (restore.policy == SessionSnapshotPolicy::full_components && range.destination_offset != allocation->offset))
            throw std::invalid_argument("restore destination offset does not match state allocation");
        const auto expected_destination =
            (range.storage == SessionStateStorage::device
                 ? state_memory.device_base : state_memory.host_base) +
            static_cast<std::uintptr_t>(range.destination_offset);
        const auto relative = range.destination_offset - allocation->offset;
        if (destination.address > std::numeric_limits<std::uintptr_t>::max() - relative ||
            destination.address + relative != expected_destination) {
            throw std::invalid_argument(
                "restore destination offset does not match bound state layout");
        }
        validate_transfer_extent(expected_destination, range.bytes);
        transfers.push_back({range.storage, source, expected_destination, range.bytes});
    }

    try {
        backend.restore_batch(transfers);
    } catch (...) {
        session.health =
            SerialSessionHealth::poisoned;
        session.arithmetic_lineage = {};
        session.arithmetic_lineage.tainted = true;
        throw;
    }

    session.next_position =
        restore.next_position;
    session.committed_tokens =
        restore.committed_tokens;
    session.state =
        restore.state;
    // A raw snapshot carries no producer history. Only the checked prefix
    // lifecycle may republish a separately validated recorded lineage.
    session.arithmetic_lineage = {};
    session.health =
        SerialSessionHealth::ready;
}

}  // namespace strix
