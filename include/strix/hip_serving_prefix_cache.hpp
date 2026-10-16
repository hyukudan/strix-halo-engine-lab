#pragma once
#include "strix/model_runtime_binding.hpp"
#include "strix/serial_prefix_cache.hpp"
#include "strix/serial_request.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace strix {
struct HipServingPrefixCachePolicy {
    std::size_t max_entries=0;
    std::uint64_t snapshot_byte_budget=0,capture_peak_bytes=0,max_prefix_tokens=0;
    // Explicit recipe.maximum_history_intervals(max_prefix_tokens) for chunks.
    std::uint64_t max_history_intervals=1;
};
struct HipServingPrefixCacheStats {
    std::uint64_t lookups=0,hits=0,restored_prefix_tokens=0,capture_attempts=0;
    std::uint64_t inserted=0,skipped=0,failures=0,entries=0,snapshot_bytes=0;
    bool saturated=false;
};
// Shared serving cache of the EXISTING complete SerialPrefixCache entries.
// Serializes LRU/capture admission; a lookup pins its entry across synchronous
// transfer completion outside that lock. Each entry retains only an independent
// original-device-safe snapshot allocator, never the source request's state,
// scratch, stream or backend. There is no pool->entry->pool ownership cycle.
//
// Reserve requested_reservation_bytes in the scheduler's shared budget BEFORE
// constructing the pool/request factories. The bound covers snapshot capture
// peaks and declared token/entry bookkeeping, not GPU driver/RSS overhead.
// Default consumers use serial history/longest-prefix matching. A supplied
// compiled producer recipe enables exact-length chunk-history matching only.
class HipServingPrefixCache final {
public:
    HipServingPrefixCache(HipServingPrefixCachePolicy,ModelContentIdentity,std::uint32_t vocabulary_size);
    ~HipServingPrefixCache();
    HipServingPrefixCache(const HipServingPrefixCache&)=delete;
    HipServingPrefixCache& operator=(const HipServingPrefixCache&)=delete;
    [[nodiscard]] static std::uint64_t requested_reservation_bytes(const HipServingPrefixCachePolicy&);
    [[nodiscard]] int device_index() const noexcept;
    [[nodiscard]] const ModelContentIdentity& model_content() const noexcept;
    [[nodiscard]] std::uint32_t vocabulary_size() const noexcept;
    [[nodiscard]] HipServingPrefixCacheStats stats() const;
    // The caller owns a fresh independent ready session and quiescent producer
    // stream. backend must finish/drain EACH transfer before returning/throwing.
    // Miss/capacity are benign; transfer errors propagate and poison the target.
    [[nodiscard]] std::optional<GreedyPrefixRestore> try_restore(
        const ModelContract&,const SerialGreedyRequest&,SerialSessionContext&,
        const SerialSessionStateLayout&,const SessionStateMemoryBinding&,
        SessionSnapshotTransferBackend&,const SerialPrefixProducerRecipe* recipe=nullptr);
    // Only a completed finite scalar frontier may be captured. The production
    // serving backend supplies its own successful terminal-selection witness.
    // No captures from state-only positions or max_new_tokens==0 requests.
    void capture(const ModelContract&,std::span<const std::uint32_t>,std::uint32_t pending,
        const SerialSessionContext&,const SerialSessionStateLayout&,const SessionStateMemoryBinding&,
        SessionSnapshotTransferBackend&,const SerialPrefixProducerRecipe* recipe=nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
