#include "strix/serial_prefix_lifecycle.hpp"

#include <stdexcept>
#include <utility>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) &
            0xffULL;
        hash *=
            fnv_prime;
    }
}

std::uint64_t build_prefix_identity(
    std::span<const std::uint32_t> tokens,
    const SerialSessionSnapshotPlan& snapshot,
    const ArithmeticLineage& lineage) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash_u64(
        hash,
        snapshot.compiled_execution_identity);
    hash_u64(
        hash,
        snapshot.runtime_compatibility_identity);
    hash_u64(
        hash,
        snapshot.state_layout_identity);
    hash_u64(
        hash,
        snapshot.plan_identity);
    hash_u64(
        hash,
        snapshot.next_position);
    hash_u64(
        hash,
        snapshot.committed_tokens);
    hash_u64(hash, lineage.history_identity);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            tokens.size()));
    for (const auto token :
         tokens) {
        hash_u64(
            hash,
            token);
    }
    return hash;
}

}  // namespace

SerialPrefixSnapshot make_serial_prefix_snapshot(
    std::span<const std::uint32_t> prefix_tokens,
    const SerialSessionSnapshotPlan& snapshot,
    const ArithmeticLineage& arithmetic_lineage) {
    if (prefix_tokens.empty()) {
        throw std::invalid_argument(
            "serial prefix snapshot requires a non-empty prefix");
    }
    if (snapshot.plan_identity == 0 ||
        snapshot.compiled_execution_identity == 0 ||
        snapshot.runtime_compatibility_identity == 0 ||
        snapshot.state_layout_identity == 0) {
        throw std::invalid_argument(
            "serial prefix snapshot requires a validated snapshot plan");
    }
    if (snapshot.next_position !=
            prefix_tokens.size() ||
        snapshot.committed_tokens !=
            prefix_tokens.size()) {
        throw std::invalid_argument(
            "serial prefix snapshot position/token metadata must match prefix length");
    }
    if (!arithmetic_lineage_valid(arithmetic_lineage, snapshot.next_position)) {
        throw std::invalid_argument("prefix snapshot requires complete independently recorded arithmetic history");
    }

    SerialPrefixSnapshot result;
    result.prefix_tokens.assign(
        prefix_tokens.begin(),
        prefix_tokens.end());
    result.snapshot =
        snapshot;
    result.arithmetic_lineage = arithmetic_lineage;
    result.prefix_identity =
        build_prefix_identity(
            result.prefix_tokens,
            result.snapshot,
            result.arithmetic_lineage);

    if (result.prefix_identity == 0) {
        throw std::logic_error(
            "serial prefix snapshot identity must be non-zero");
    }
    return result;
}

bool serial_prefix_snapshot_matches(
    const SerialPrefixSnapshot& prefix,
    std::span<const std::uint32_t> requested_prefix,
    std::uint64_t compiled_execution_identity,
    std::uint64_t runtime_compatibility_identity,
    std::uint64_t state_layout_identity,
    const ArithmeticLineage* requested_history) noexcept {
    if (prefix.prefix_identity == 0 ||
        prefix.snapshot.plan_identity == 0 ||
        requested_prefix.size() !=
            prefix.prefix_tokens.size() ||
        compiled_execution_identity == 0 ||
        runtime_compatibility_identity == 0 ||
        state_layout_identity == 0) {
        return false;
    }
    if (compiled_execution_identity !=
            prefix.snapshot.compiled_execution_identity ||
        runtime_compatibility_identity !=
            prefix.snapshot.runtime_compatibility_identity ||
        state_layout_identity !=
            prefix.snapshot.state_layout_identity) {
        return false;
    }
    if (prefix.snapshot.next_position != requested_prefix.size() ||
        prefix.snapshot.committed_tokens != requested_prefix.size()) { return false; }
    if (requested_history != nullptr) {
        if (!arithmetic_lineage_matches(prefix.arithmetic_lineage, *requested_history,
                                        prefix.snapshot.next_position)) { return false; }
    } else if (!arithmetic_lineage_serial_only(prefix.arithmetic_lineage,
                compiled_execution_identity, prefix.snapshot.next_position)) {
        // The existing caller is serial-only, not an implicit chunked-prefill plan.
        return false;
    }
    for (std::size_t index = 0;
         index < requested_prefix.size();
         ++index) {
        if (requested_prefix[index] !=
            prefix.prefix_tokens[index]) {
            return false;
        }
    }

    return prefix.prefix_identity ==
        build_prefix_identity(
            prefix.prefix_tokens,
            prefix.snapshot,
            prefix.arithmetic_lineage);
}

SerialSessionRestorePlan
build_serial_prefix_restore_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& destination_layout,
    std::uint64_t compiled_execution_identity,
    std::uint64_t runtime_compatibility_identity,
    std::span<const std::uint32_t> requested_prefix,
    const SerialPrefixSnapshot& prefix,
    const ArithmeticLineage* requested_history) {
    if (!serial_prefix_snapshot_matches(
            prefix,
            requested_prefix,
            compiled_execution_identity,
            runtime_compatibility_identity,
            destination_layout.layout_identity,
            requested_history)) {
        throw std::invalid_argument(
            "serial prefix snapshot does not match requested prefix/runtime");
    }

    return build_serial_session_restore_plan(
        model,
        destination_layout,
        runtime_compatibility_identity,
        prefix.snapshot);
}


void capture_serial_prefix_snapshot(
    const SerialPrefixSnapshot& prefix,
    const SerialSessionContext& session,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend) {
    if (!serial_prefix_snapshot_matches(
            prefix,
            prefix.prefix_tokens,
            session.execution_identity,
            session.runtime_compatibility_identity,
            session.state_layout_identity,
            &session.arithmetic_lineage)) {
        throw std::invalid_argument(
            "serial prefix capture session does not match prefix snapshot");
    }
    if (session.health !=
            SerialSessionHealth::ready ||
        session.next_position !=
            prefix.snapshot.next_position ||
        session.committed_tokens !=
            prefix.snapshot.committed_tokens ||
        session.state.valid !=
            prefix.snapshot.state.valid ||
        session.state.generations !=
            prefix.snapshot.state.generations) {
        throw std::invalid_argument(
            "serial prefix capture session metadata does not match snapshot");
    }
    if (state_layout.layout_identity !=
        prefix.snapshot.state_layout_identity) {
        throw std::invalid_argument(
            "serial prefix capture layout does not match snapshot");
    }

    capture_serial_session_snapshot(
        prefix.snapshot,
        state_layout,
        state_memory,
        snapshot_memory,
        backend);
}

void restore_serial_prefix_snapshot(
    SerialSessionContext& session,
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend,
    std::span<const std::uint32_t> requested_prefix,
    const SerialPrefixSnapshot& prefix,
    const ArithmeticLineage* requested_history) {
    const auto restore =
        build_serial_prefix_restore_plan(
            model,
            state_layout,
            session.execution_identity,
            session.runtime_compatibility_identity,
            requested_prefix,
            prefix,
            requested_history);

    // Stage the copy before transfers; publication after completion cannot allocate.
    auto restored_lineage = prefix.arithmetic_lineage;

    restore_serial_session_snapshot(
        session,
        restore,
        state_layout,
        state_memory,
        snapshot_memory,
        backend);
    session.arithmetic_lineage = std::move(restored_lineage);
}

}  // namespace strix
