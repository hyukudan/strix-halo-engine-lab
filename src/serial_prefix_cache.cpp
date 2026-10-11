#include "strix/serial_prefix_cache.hpp"

#include <stdexcept>
#include <utility>

namespace strix {

SerialPrefixCacheEntry
SerialPrefixCacheEntry::capture(
    const ModelContract& model,
    std::span<const std::uint32_t> prefix_tokens,
    const SerialSessionContext& session,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    RuntimeMemoryAllocator& allocator,
    AllocationGenerationSource& generations,
    SessionSnapshotTransferBackend& backend,
    std::uint64_t alignment) {
    if (session.health !=
            SerialSessionHealth::ready) {
        throw std::invalid_argument(
            "serial prefix cache capture requires a ready session");
    }
    if (prefix_tokens.empty()) {
        throw std::invalid_argument(
            "serial prefix cache capture requires a non-empty prefix");
    }
    if (session.execution_identity == 0 ||
        session.runtime_compatibility_identity == 0 ||
        session.state_layout_identity == 0 ||
        session.state_layout_identity !=
            state_layout.layout_identity) {
        throw std::invalid_argument(
            "serial prefix cache capture requires matching session identities");
    }
    if (session.next_position !=
            prefix_tokens.size() ||
        session.committed_tokens !=
            prefix_tokens.size()) {
        throw std::invalid_argument(
            "serial prefix cache capture requires a fully committed prefix");
    }

    SerialSessionSnapshotRequest request;
    request.compiled_execution_identity =
        session.execution_identity;
    request.runtime_compatibility_identity =
        session.runtime_compatibility_identity;
    request.next_position =
        session.next_position;
    request.committed_tokens =
        session.committed_tokens;
    request.state =
        session.state;
    request.alignment =
        alignment;

    auto snapshot =
        build_serial_session_snapshot_plan(
            model,
            state_layout,
            request);
    auto prefix =
        make_serial_prefix_snapshot(
            prefix_tokens,
            snapshot,
            session.arithmetic_lineage);
    auto memory =
        SessionSnapshotMemoryOwner::allocate(
            snapshot,
            allocator,
            generations);

    capture_serial_prefix_snapshot(
        prefix,
        session,
        state_layout,
        state_memory,
        memory.binding(),
        backend);

    SerialPrefixCacheEntry entry;
    entry.prefix_ =
        std::move(prefix);
    entry.memory_ =
        std::move(memory);
    return entry;
}

void SerialPrefixCacheEntry::restore(
    SerialSessionContext& session,
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    SessionSnapshotTransferBackend& backend,
    std::span<const std::uint32_t> requested_prefix,
    const ArithmeticLineage* requested_history) const {
    if (!valid()) {
        throw std::logic_error(
            "serial prefix cache entry is not initialized");
    }

    if (!try_restore(session, model, state_layout, state_memory, backend, requested_prefix, requested_history)) {
        throw std::invalid_argument("serial prefix cache miss for requested tokens/runtime/arithmetic history");
    }
}

bool SerialPrefixCacheEntry::matches(
    std::span<const std::uint32_t> requested_prefix,
    std::uint64_t execution_identity,
    std::uint64_t runtime_compatibility_identity,
    std::uint64_t state_layout_identity,
    const ArithmeticLineage* requested_history) const noexcept {
    return valid() && serial_prefix_snapshot_matches(prefix_, requested_prefix,
        execution_identity, runtime_compatibility_identity, state_layout_identity, requested_history);
}

bool SerialPrefixCacheEntry::try_restore(
    SerialSessionContext& session, const ModelContract& model,
    const SerialSessionStateLayout& state_layout, const SessionStateMemoryBinding& state_memory,
    SessionSnapshotTransferBackend& backend, std::span<const std::uint32_t> requested_prefix,
    const ArithmeticLineage* requested_history) const {
    if (!matches(requested_prefix, session.execution_identity, session.runtime_compatibility_identity,
                 session.state_layout_identity, requested_history)) { return false; }
    restore_serial_prefix_snapshot(session, model, state_layout, state_memory, memory_.binding(),
                                  backend, requested_prefix, prefix_, requested_history);
    return true;
}

}  // namespace strix
