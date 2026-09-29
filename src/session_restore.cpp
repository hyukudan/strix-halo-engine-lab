#include "strix/session_restore.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}


bool same_snapshot_structure(
    const SerialSessionSnapshotPlan& left,
    const SerialSessionSnapshotPlan& right) {
    if (left.plan_identity !=
            right.plan_identity ||
        left.device_bytes !=
            right.device_bytes ||
        left.host_bytes !=
            right.host_bytes ||
        left.alignment !=
            right.alignment ||
        left.ranges.size() !=
            right.ranges.size()) {
        return false;
    }

    for (std::size_t index = 0;
         index < left.ranges.size();
         ++index) {
        const auto& a = left.ranges[index];
        const auto& b = right.ranges[index];
        if (a.component != b.component ||
            a.storage != b.storage ||
            a.source_offset != b.source_offset ||
            a.destination_offset !=
                b.destination_offset ||
            a.bytes != b.bytes ||
            a.provisional != b.provisional) {
            return false;
        }
    }
    return true;
}

std::uint64_t build_identity(
    const SerialSessionRestorePlan& plan) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(
        hash,
        plan.compiled_execution_identity);
    hash_u64(
        hash,
        plan.runtime_compatibility_identity);
    hash_u64(
        hash,
        plan.snapshot_plan_identity);
    hash_u64(
        hash,
        plan.state_layout_identity);
    hash_u64(hash, plan.next_position);
    hash_u64(hash, plan.committed_tokens);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            plan.state.valid));
    for (const auto generation :
         plan.state.generations) {
        hash_u64(hash, generation);
    }
    hash_u64(
        hash,
        plan.snapshot_device_bytes);
    hash_u64(
        hash,
        plan.snapshot_host_bytes);

    for (const auto& range : plan.ranges) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                range.component));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                range.storage));
        hash_u64(
            hash,
            range.snapshot_offset);
        hash_u64(
            hash,
            range.destination_offset);
        hash_u64(hash, range.bytes);
    }
    return hash;
}

}  // namespace

SerialSessionRestorePlan
build_serial_session_restore_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& destination_layout,
    std::uint64_t runtime_compatibility_identity,
    const SerialSessionSnapshotPlan& snapshot) {
    if (runtime_compatibility_identity == 0 ||
        snapshot.runtime_compatibility_identity !=
            runtime_compatibility_identity) {
        throw std::invalid_argument(
            "snapshot runtime compatibility identity mismatch");
    }
    if (snapshot.state_layout_identity !=
            destination_layout.layout_identity) {
        throw std::invalid_argument(
            "snapshot state-layout identity mismatch");
    }

    SerialSessionSnapshotRequest expected_request;
    expected_request.compiled_execution_identity =
        snapshot.compiled_execution_identity;
    expected_request.runtime_compatibility_identity =
        snapshot.runtime_compatibility_identity;
    expected_request.next_position =
        snapshot.next_position;
    expected_request.committed_tokens =
        snapshot.committed_tokens;
    expected_request.alignment =
        snapshot.alignment;
    expected_request.state =
        snapshot.state;

    const auto expected_snapshot =
        build_serial_session_snapshot_plan(
            model,
            destination_layout,
            expected_request);

    if (!same_snapshot_structure(
            snapshot,
            expected_snapshot)) {
        throw std::invalid_argument(
            "snapshot structure or identity is invalid");
    }

    SerialSessionRestorePlan restore;
    restore.compiled_execution_identity =
        snapshot.compiled_execution_identity;
    restore.runtime_compatibility_identity =
        runtime_compatibility_identity;
    restore.snapshot_plan_identity =
        snapshot.plan_identity;
    restore.state_layout_identity =
        destination_layout.layout_identity;
    restore.next_position =
        snapshot.next_position;
    restore.committed_tokens =
        snapshot.committed_tokens;
    restore.state = snapshot.state;
    restore.snapshot_device_bytes =
        snapshot.device_bytes;
    restore.snapshot_host_bytes =
        snapshot.host_bytes;
    restore.ranges.reserve(
        snapshot.ranges.size());

    for (const auto& source :
         snapshot.ranges) {
        const auto* destination =
            find_session_state_allocation(
                destination_layout,
                source.component);
        if (destination == nullptr) {
            throw std::invalid_argument(
                "restore destination component is missing");
        }
        if (destination->storage !=
            source.storage ||
            source.bytes >
                destination->bytes) {
            throw std::invalid_argument(
                "restore range is incompatible with destination component");
        }

        const auto snapshot_total =
            source.storage ==
                    SessionStateStorage::device
                ? snapshot.device_bytes
                : snapshot.host_bytes;
        if (source.destination_offset >
                snapshot_total ||
            source.bytes >
                snapshot_total -
                    source.destination_offset) {
            throw std::invalid_argument(
                "restore range exceeds compact snapshot payload");
        }

        restore.ranges.push_back(
            SessionRestoreRange{
                source.component,
                source.storage,
                source.destination_offset,
                destination->offset,
                source.bytes,
            });
    }

    restore.plan_identity =
        build_identity(restore);
    if (restore.plan_identity == 0) {
        throw std::logic_error(
            "restore plan identity must be non-zero");
    }
    return restore;
}

}  // namespace strix
