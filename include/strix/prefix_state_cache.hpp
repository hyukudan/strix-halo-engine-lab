#pragma once

#include "strix/model_runtime_binding.hpp"

#include <stdexcept>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <utility>

namespace strix {

// Block-boundary prefix state cache (see docs/engine/serving/prefix-state-cache.md).
//
// A checkpoint is the complete session state after the first P = k*T prompt tokens of a tiled
// prefill (T = --known-prefix-prefill-rows), split into
//   - k row segments, one per T-row block: the per-position state rows of that block (KV and raw
//     index keys of every attention layer). Rows never change after they are written, so the
//     checkpoints of one prompt share the segments of their common blocks;
//   - one fixed blob: the state that is overwritten in place (recurrent matrices, convolution
//     histories, n-gram and position history);
//   - opaque metadata (session counters and arithmetic lineage).
// The store does not interpret any payload. It owns keys, token verification, the host byte
// budget with selectable retention and the optional disk tier.
//
// Keys. identity = SHA-256 of everything that changes the state a prefix produces (model content,
// state layout, runtime compatibility, prefill and decode executions and kernel plans, numerical
// profile pair, T, ...; built by the caller). digest_k = SHA-256(digest_{k-1} || tokens of block
// k as little-endian uint32), digest_0 = identity. A digest therefore names one identity and one
// exact token prefix; the tokens are stored and compared as well.
struct PrefixCacheBuffer {
    std::shared_ptr<std::byte> data;
    std::uint64_t bytes = 0;
    [[nodiscard]] std::span<std::byte> span() const noexcept { return {data.get(), static_cast<std::size_t>(bytes)}; }
};
// Allocates payload storage (pinned host memory on the GPU path). Must return bytes >= requested.
using PrefixCacheAllocator = std::function<PrefixCacheBuffer(std::uint64_t bytes)>;

struct PrefixCacheSegment {
    ModelContentIdentity identity, digest, parent;
    std::uint64_t block_index = 0;  // 0-based: rows [block_index*T, (block_index+1)*T)
    std::uint32_t block_rows = 0;
    std::vector<std::uint32_t> tokens;  // the T tokens of this block
    PrefixCacheBuffer payload;
};

struct PrefixCacheCheckpoint {
    ModelContentIdentity identity, digest;
    std::vector<std::uint64_t> producer_certificate; // nonempty: certified RAM-only family
    std::string conversation; // retention only; never serialized or part of numerical identity
    std::uint64_t position = 0;  // P = segments.size() * block_rows
    std::uint32_t block_rows = 0;
    std::vector<std::shared_ptr<const PrefixCacheSegment>> segments;
    PrefixCacheBuffer fixed;
    std::vector<std::byte> metadata;
    // Copy of every token of the prefix, checked on lookup.
    [[nodiscard]] std::vector<std::uint32_t> tokens() const;
};

[[nodiscard]] ModelContentIdentity prefix_cache_block_digest(const ModelContentIdentity& previous,
    std::span<const std::uint32_t> block_tokens);
// digests[k-1] names the first k blocks; at most max_blocks digests.
[[nodiscard]] std::vector<ModelContentIdentity> prefix_cache_block_digests(const ModelContentIdentity& identity,
    std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t max_blocks);
// Cheap 64-bit payload checksum used by the disk tier (not a security digest).
[[nodiscard]] std::uint64_t prefix_cache_checksum(std::span<const std::byte> bytes) noexcept;

enum class PrefixCacheRetention { fair, lru };

struct PrefixStateCachePolicy {
    std::uint64_t host_byte_budget = 0;
    std::filesystem::path disk_directory;  // empty: no disk tier
    std::uint64_t disk_byte_budget = 0;    // 0 with a directory: unbounded
    PrefixCacheRetention retention = PrefixCacheRetention::fair;
    bool legacy = false; // Baseline preserves the original explicit-cache path.
};

// Allocators may mark non-resource failures that opportunistic capture must propagate.
class PrefixCacheFatalError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PrefixCacheAllocationFailure : public std::runtime_error {
public:
    PrefixCacheAllocationFailure() : std::runtime_error("prefix state cache pinned host allocation failed") {}
};

inline constexpr std::array<std::string_view,11> prefix_certificate_decline_names{
    "serial-async-capture-unsupported",
    "serial-lookup-disabled-or-short",
    "serial-certificate-missing-or-incompatible",
    "serial-plan-unsupported",
    "serial-no-certified-checkpoint",
    "serial-producer-plan-mismatch",
    "serial-cursor-continuation-unsupported",
    "serial-checkpoint-preflight",
    "serial-capture-unit-unsupported",
    "serial-capture-boundary-unsupported",
    "serial-capture-metadata-mismatch",
};
struct PrefixStateCacheStats {
    std::uint64_t lookups = 0, plan_misses = 0, host_hits = 0, disk_hits = 0, misses = 0;
    std::uint64_t inserted = 0, evicted = 0, rejected_over_budget = 0;
    std::uint64_t host_bytes = 0, host_checkpoints = 0, host_segments = 0;
    std::uint64_t disk_writes = 0, disk_write_bytes = 0, disk_write_failures = 0, disk_evictions = 0;
    std::uint64_t disk_bytes = 0;
    std::uint64_t disk_reads = 0, disk_read_bytes = 0, disk_read_failures = 0;
    std::uint64_t lookup_tokens = 0, hit_tokens = 0, restored_tokens = 0;
    std::uint64_t captures = 0, no_store_skips = 0, budget_skips = 0, capture_failures = 0;
    std::uint64_t evictions_space = 0, evictions_conversation_cap = 0, evictions_branch_cap = 0;
    std::uint64_t evictions_explicit = 0, evictions_invalid = 0;
    std::uint64_t staging_budget_bytes=0, staging_bytes=0, staging_peak_bytes=0, capture_busy_skips=0, staging_cap_skips=0, row_prefetches=0;
    std::uint64_t capture_skips_cross_stream=0;
    std::array<std::uint64_t,prefix_certificate_decline_names.size()> certificate_declines{};
    double disk_read_ms = 0.0, disk_write_ms = 0.0;
};

class PrefixStateCache final {
    struct Ledger;
public:
    class Reservation {
        friend class PrefixStateCache;
        std::shared_ptr<Ledger> ledger_;
        std::uint64_t bytes_ = 0;
        Reservation(std::shared_ptr<Ledger> ledger, std::uint64_t bytes) : ledger_(std::move(ledger)), bytes_(bytes) {}
    public:
        Reservation() = default;
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        Reservation(Reservation&& other) noexcept : ledger_(std::move(other.ledger_)), bytes_(std::exchange(other.bytes_, 0)) {}
        Reservation& operator=(Reservation&& other) noexcept {
            if (this != &other) { Reservation old(std::move(*this)); ledger_ = std::move(other.ledger_); bytes_ = std::exchange(other.bytes_, 0); }
            return *this;
        }
        ~Reservation();
        explicit operator bool() const noexcept { return bool(ledger_); }
    };
    // Credits include outstanding reservations and live payload leases. An incoming
    // capture preserves fair directory admission; disk reads need no directory slot.
    [[nodiscard]] Reservation reserve(std::uint64_t bytes, const PrefixCacheCheckpoint* incoming = nullptr);
    [[nodiscard]] PrefixCacheBuffer allocate(std::uint64_t bytes, Reservation&, bool segment = false);

    PrefixStateCache(PrefixStateCachePolicy, PrefixCacheAllocator);
    ~PrefixStateCache();
    PrefixStateCache(const PrefixStateCache&) = delete;
    PrefixStateCache& operator=(const PrefixStateCache&) = delete;

    // Longest checkpoint for this identity whose prefix is tokens[0, P) with P = k*block_rows <=
    // limit. Host first, then disk (promote=false suppresses host publication). Null: miss.
    // The returned handle keeps its payloads alive. An eligibility predicate runs outside
    // the mutex on that lease, before hit accounting; this candidate path is RAM-only.
    [[nodiscard]] std::shared_ptr<const PrefixCacheCheckpoint> find_longest(const ModelContentIdentity& identity,
        std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t limit, bool promote = true, std::string_view conversation = {},
        const std::function<bool(const PrefixCacheCheckpoint&)>& eligible = {});
    [[nodiscard]] bool contains(const ModelContentIdentity& digest);
    // A retained segment with this digest (host tier only), or null.
    [[nodiscard]] std::shared_ptr<const PrefixCacheSegment> find_segment(const ModelContentIdentity& digest);
    // Inserts into the host tier under its retention policy and queues the disk write.
    // False when the checkpoint or bounded directory cannot be admitted.
    bool insert(std::shared_ptr<const PrefixCacheCheckpoint>, Reservation* = nullptr);
    [[nodiscard]] PrefixCacheBuffer allocate(std::uint64_t bytes) const;
    // Named complete-session blobs share this store's pinned allocator, byte accounting and LRU.
    // They never enter the opportunistic content-prefix namespace or the legacy disk writer.
    // Serialized caller reserves temporary capture/read bytes before allocating on unified memory.
    bool admit_session_working_bytes(std::uint64_t bytes);
    bool park_session(const std::string& id, PrefixCacheBuffer envelope, std::uint64_t max_sessions);
    [[nodiscard]] PrefixCacheBuffer resume_session(const std::string& id);
    [[nodiscard]] std::vector<std::string> list_sessions();
    bool delete_session(const std::string& id);
    // Waits for queued disk writes.
    void flush();
    [[nodiscard]] PrefixStateCacheStats stats();
    [[nodiscard]] PrefixStateCacheStats counters();
    void record_no_store_skip();
    void record_capture_failure();
    void record_staging(std::uint64_t cap, std::uint64_t bytes, bool active);
    void record_staging_skip(bool cap);
    void record_cross_stream_skip();
    void record_certificate_decline(const std::string& reason);
    void record_row_prefetch();
    // Advisory admission only. Use reserve() to hold credits across allocation.
    bool admit_capture(const PrefixCacheCheckpoint&);
    [[nodiscard]] const PrefixStateCachePolicy& policy() const noexcept { return policy_; }
    // Bytes a checkpoint adds to the host tier now (segments already retained are not counted).
    [[nodiscard]] std::uint64_t incremental_bytes(const PrefixCacheCheckpoint&);

private:
    struct Entry {
        std::shared_ptr<const PrefixCacheCheckpoint> checkpoint;
        std::list<ModelContentIdentity>::iterator lru;
        std::uint64_t epoch = 0, reuse = 0;
        std::set<std::string> conversations;
    };
    struct DigestLess {
        bool operator()(const ModelContentIdentity& a, const ModelContentIdentity& b) const noexcept {
            return a.bytes < b.bytes;
        }
    };
    struct Ledger { std::mutex mutex; std::uint64_t bytes = 0, segments = 0, reserved = 0; };
    struct Receipt {
        std::shared_ptr<Ledger> ledger;
        std::shared_ptr<std::byte> payload;
        std::uint64_t bytes = 0;
        bool registered = false, segment = false;
        ~Receipt() {
            if (registered) {
                std::lock_guard lock(ledger->mutex);
                payload.reset();
                ledger->bytes -= bytes;
                if (segment) --ledger->segments;
            }
        }
    };
    std::shared_ptr<Ledger> ledger_ = std::make_shared<Ledger>();
    mutable std::map<const std::byte*, std::weak_ptr<Receipt>> receipts_;
    void register_buffer_locked(const PrefixCacheBuffer&, bool segment);
    void add_locked(std::shared_ptr<const PrefixCacheCheckpoint>);
    void erase_locked(const ModelContentIdentity&);
    std::uint64_t epoch_ = 0, metadata_bytes_ = 0;
    std::uint64_t host_bytes_locked();
    std::uint64_t retained_bytes_locked();
    void touch_locked(Entry&, bool reused = false);
    void evict_to_budget_locked(std::uint64_t reserve, const PrefixCacheCheckpoint* incoming = nullptr);
    std::shared_ptr<const PrefixCacheCheckpoint> load_from_disk(const ModelContentIdentity& identity,
        const ModelContentIdentity& digest, std::span<const std::uint32_t> tokens, std::uint32_t block_rows);
    std::shared_ptr<const PrefixCacheSegment> load_segment(const ModelContentIdentity& identity,
        const ModelContentIdentity& digest, std::uint32_t block_rows, Reservation* reservation);
    void write_to_disk(const PrefixCacheCheckpoint&);
    void enforce_disk_budget();
    void update_disk_gauge();
    std::uint64_t incremental_bytes_locked(const PrefixCacheCheckpoint&);
    // Deletes temporary files and segment files no checkpoint references, older than min_age.
    void sweep_disk(std::chrono::seconds min_age);
    void writer_loop();

    std::map<std::string, ModelContentIdentity> named_sessions_;
    std::map<const std::byte*, std::pair<std::weak_ptr<std::byte>, std::uint64_t>> session_buffers_;
    PrefixStateCachePolicy policy_;
    PrefixCacheAllocator allocate_;
    mutable std::mutex mutex_;
    std::map<ModelContentIdentity, Entry, DigestLess> entries_;
    std::list<ModelContentIdentity> lru_;  // front: least recently used
    std::map<ModelContentIdentity, std::weak_ptr<const PrefixCacheSegment>, DigestLess> segments_;
    PrefixStateCacheStats stats_;
    // Disk writer.
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_, idle_cv_;
    std::deque<std::shared_ptr<const PrefixCacheCheckpoint>> queue_;
    bool writer_busy_ = false, stopping_ = false;
    std::thread writer_;
};

}  // namespace strix
