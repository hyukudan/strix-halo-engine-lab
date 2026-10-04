#pragma once

#include "strix/model_contract.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/serial_prefix_lifecycle.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_snapshot_memory.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>
#include <span>

namespace strix {

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
        std::uint64_t alignment = 64);

    void restore(
        SerialSessionContext& session,
        const ModelContract& model,
        const SerialSessionStateLayout& state_layout,
        const SessionStateMemoryBinding& state_memory,
        SessionSnapshotTransferBackend& backend,
        std::span<const std::uint32_t> requested_prefix) const;

    [[nodiscard]] bool valid() const noexcept {
        return prefix_.prefix_identity != 0 &&
               memory_.valid();
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
    SessionSnapshotMemoryOwner memory_;
};

}  // namespace strix
