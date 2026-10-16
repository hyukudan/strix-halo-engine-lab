#include "strix/serial_prefix_cache.hpp"

#include <algorithm>
#include <limits>
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

SerialPrefixCache::SerialPrefixCache(std::size_t max_entries, std::uint64_t snapshot_byte_budget)
    : max_entries_(max_entries), byte_budget_(snapshot_byte_budget) {
    if (!max_entries_ || !byte_budget_)
        throw std::invalid_argument("serial prefix cache requires nonzero entry and snapshot budgets");
    slots_.reserve(max_entries_);
}

std::shared_ptr<const SerialPrefixCacheEntry> SerialPrefixCache::find(
    std::span<const std::uint32_t> tokens, std::uint64_t execution,
    std::uint64_t compatibility, std::uint64_t layout, const ArithmeticLineage* history) {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i].entry->matches(tokens, execution, compatibility, layout, history)) continue;
        auto handle = slots_[i].entry;
        std::rotate(slots_.begin() + i, slots_.begin() + i + 1U, slots_.end());
        return handle;
    }
    return {};
}

SerialPrefixCacheAdmission SerialPrefixCache::insert(
    SerialPrefixCacheEntry entry, std::shared_ptr<const void> actual_owner) {
    struct CandidateLifetime {
        std::shared_ptr<const void> owner;
        SerialPrefixCacheEntry& snapshot;
        ~CandidateLifetime() { snapshot = SerialPrefixCacheEntry{}; }
    } candidate_lifetime{actual_owner, entry};
    // Keep the allocator alive while discarding a rejected candidate as well.
    // Function-parameter destruction order must not decide resource lifetime.
    if (!entry.valid() || !actual_owner)
        throw std::invalid_argument("serial prefix cache insertion requires a valid snapshot and actual owner");
    const auto& prefix = entry.prefix();
    if (find(prefix.prefix_tokens, prefix.snapshot.compiled_execution_identity,
        prefix.snapshot.runtime_compatibility_identity, prefix.snapshot.state_layout_identity,
        &prefix.arithmetic_lineage)) return SerialPrefixCacheAdmission::already_present;
    const auto& memory = entry.memory();
    if (memory.device_section_bytes > std::numeric_limits<std::uint64_t>::max() - memory.host_section_bytes)
        return SerialPrefixCacheAdmission::snapshot_budget_exceeded;
    const auto bytes = memory.device_section_bytes + memory.host_section_bytes;
    if (bytes > byte_budget_) return SerialPrefixCacheAdmission::snapshot_budget_exceeded;
    auto remaining_bytes = snapshot_bytes_;
    auto remaining_entries = slots_.size();
    // Prove admission before evicting anything. A failed admission preserves all
    // existing snapshots, including unpinned ones needed by another request.
    for (const auto& slot : slots_) {
        if (remaining_entries < max_entries_ && remaining_bytes <= byte_budget_ - bytes) break;
        if (slot.entry.use_count() != 1) continue;
        remaining_bytes -= slot.bytes;
        --remaining_entries;
    }
    if (remaining_entries >= max_entries_ || remaining_bytes > byte_budget_ - bytes)
        return SerialPrefixCacheAdmission::pinned_capacity;
    struct RetainedEntry {
        // Destruction releases the snapshot before its allocator/resource owner.
        std::shared_ptr<const void> owner;
        SerialPrefixCacheEntry entry;
        RetainedEntry(std::shared_ptr<const void> value, SerialPrefixCacheEntry snapshot)
            : owner(std::move(value)), entry(std::move(snapshot)) {}
    };
    auto retained = std::make_shared<RetainedEntry>(std::move(actual_owner), std::move(entry));
    auto handle = std::shared_ptr<const SerialPrefixCacheEntry>(retained, &retained->entry);
    for (std::size_t i = 0; i < slots_.size() &&
        (slots_.size() >= max_entries_ || snapshot_bytes_ > byte_budget_ - bytes);) {
        if (slots_[i].entry.use_count() != 1) { ++i; continue; }
        snapshot_bytes_ -= slots_[i].bytes;
        slots_.erase(slots_.begin() + i);
    }
    slots_.push_back({std::move(handle), bytes});
    snapshot_bytes_ += bytes;
    return SerialPrefixCacheAdmission::inserted;
}

std::size_t SerialPrefixCache::evict_unpinned() noexcept {
    std::size_t evicted = 0;
    for (std::size_t i = 0; i < slots_.size();) {
        if (slots_[i].entry.use_count() != 1) { ++i; continue; }
        snapshot_bytes_ -= slots_[i].bytes;
        slots_.erase(slots_.begin() + i);
        ++evicted;
    }
    return evicted;
}

}  // namespace strix
