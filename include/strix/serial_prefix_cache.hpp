#pragma once

#include "strix/model_contract.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/serial_prefix_lifecycle.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_snapshot_memory.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace strix {

struct SerialPrefixGreedyFrontier {
    ModelContentIdentity model_content;
    std::uint64_t vocabulary_size = 0;
    // Selected from the COMPLETE finite terminal vocabulary of this prefix.
    // This token is pending: it has not been consumed into the snapshot state.
    std::uint32_t pending_token = 0;
};

class SerialPrefixCacheEntry {
public:
    SerialPrefixCacheEntry() = default;
    SerialPrefixCacheEntry(
        const SerialPrefixCacheEntry&) = delete;
    SerialPrefixCacheEntry& operator=(
        const SerialPrefixCacheEntry&) = delete;

    SerialPrefixCacheEntry(
        SerialPrefixCacheEntry&&) noexcept = default;
    SerialPrefixCacheEntry& operator=(
        SerialPrefixCacheEntry&&) noexcept = default;

    [[nodiscard]] static SerialPrefixCacheEntry capture(
        const ModelContract& model,
        std::span<const std::uint32_t> prefix_tokens,
        const SerialSessionContext& session,
        const SerialSessionStateLayout& state_layout,
        const SessionStateMemoryBinding& state_memory,
        RuntimeMemoryAllocator& allocator,
        AllocationGenerationSource& generations,
        SessionSnapshotTransferBackend& backend,
        std::uint64_t alignment = 64,
        std::optional<SerialPrefixGreedyFrontier> greedy_frontier = std::nullopt);

    void restore(
        SerialSessionContext& session,
        const ModelContract& model,
        const SerialSessionStateLayout& state_layout,
        const SessionStateMemoryBinding& state_memory,
        SessionSnapshotTransferBackend& backend,
        std::span<const std::uint32_t> requested_prefix,
        const ArithmeticLineage* requested_history = nullptr) const;

    [[nodiscard]] bool matches(
        std::span<const std::uint32_t> requested_prefix,
        std::uint64_t execution_identity,
        std::uint64_t runtime_compatibility_identity,
        std::uint64_t state_layout_identity,
        const ArithmeticLineage* requested_history = nullptr) const noexcept;

    // Expected history/token/compatibility mismatches are ordinary cache misses;
    // transfer failures still propagate and poison the destination session.
    [[nodiscard]] bool try_restore(
        SerialSessionContext& session, const ModelContract& model,
        const SerialSessionStateLayout& state_layout,
        const SessionStateMemoryBinding& state_memory,
        SessionSnapshotTransferBackend& backend,
        std::span<const std::uint32_t> requested_prefix,
        const ArithmeticLineage* requested_history = nullptr) const;

    [[nodiscard]] bool valid() const noexcept {
        return prefix_.snapshot.policy == SessionSnapshotPolicy::full_components && prefix_.prefix_identity != 0 &&
               arithmetic_lineage_valid(prefix_.arithmetic_lineage, prefix_.snapshot.next_position) &&
               memory_.valid();
    }

    [[nodiscard]] const std::optional<SerialPrefixGreedyFrontier>& greedy_frontier() const noexcept {
        return greedy_frontier_;
    }

    [[nodiscard]] const SerialPrefixSnapshot&
    prefix() const noexcept {
        return prefix_;
    }

    [[nodiscard]] const SessionSnapshotMemoryBinding&
    memory() const noexcept {
        return memory_.binding();
    }

private:
    SerialPrefixSnapshot prefix_;
    std::optional<SerialPrefixGreedyFrontier> greedy_frontier_;
    SessionSnapshotMemoryOwner memory_;
};

enum class SerialPrefixCacheAdmission {
    inserted,
    already_present,
    snapshot_budget_exceeded,
    pinned_capacity,
};

// Serialized LRU of complete snapshots. The byte budget covers the retained
// host/device snapshot sections, not allocator overhead or externally captured
// candidates. Callers budget capture peaks separately. Entry count is bounded.
// A lookup handle pins its snapshot AND the actual allocator/resource owner;
// pinned entries cannot be evicted to admit another snapshot. Hold that handle
// through transfer completion/device quiescence, including failure recovery.
class SerialPrefixCache final {
public:
    SerialPrefixCache(std::size_t max_entries, std::uint64_t snapshot_byte_budget);
    SerialPrefixCache(const SerialPrefixCache&) = delete;
    SerialPrefixCache& operator=(const SerialPrefixCache&) = delete;

    [[nodiscard]] SerialPrefixCacheAdmission insert(
        SerialPrefixCacheEntry entry, std::shared_ptr<const void> actual_owner);
    [[nodiscard]] std::shared_ptr<const SerialPrefixCacheEntry> find(
        std::span<const std::uint32_t> prefix_tokens,
        std::uint64_t execution_identity, std::uint64_t runtime_compatibility_identity,
        std::uint64_t state_layout_identity, const ArithmeticLineage* history = nullptr);
    // Longest exact token prefix with a real scalar frontier and the same
    // model/runtime/layout/serial arithmetic history. No chunk-history promotion.
    // The returned handle pins allocation owners through restore completion.
    [[nodiscard]] std::shared_ptr<const SerialPrefixCacheEntry> find_longest_greedy(
        std::span<const std::uint32_t> request_tokens, const ModelContentIdentity& model_content,
        std::uint64_t vocabulary_size, std::uint64_t execution_identity,
        std::uint64_t runtime_compatibility_identity, std::uint64_t state_layout_identity);
    // Evicts only unpinned snapshots; outstanding handles retain their owners.
    [[nodiscard]] std::size_t evict_unpinned() noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }
    [[nodiscard]] std::uint64_t snapshot_bytes() const noexcept { return snapshot_bytes_; }
    [[nodiscard]] std::uint64_t snapshot_byte_budget() const noexcept { return byte_budget_; }
private:
    struct Slot {
        std::shared_ptr<const SerialPrefixCacheEntry> entry;
        std::uint64_t bytes = 0;
    };
    const std::size_t max_entries_;
    const std::uint64_t byte_budget_;
    std::uint64_t snapshot_bytes_ = 0;
    std::vector<Slot> slots_; // least recently used first
};

}  // namespace strix
