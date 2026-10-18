#pragma once

#include "strix/model_runtime_binding.hpp"

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
#include <thread>
#include <vector>

namespace strix {

// Block-boundary prefix state cache (see docs/prefix-state-cache.md).
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
// budget with LRU eviction and the optional disk tier.
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

struct PrefixStateCachePolicy {
    std::uint64_t host_byte_budget = 0;
    std::filesystem::path disk_directory;  // empty: no disk tier
    std::uint64_t disk_byte_budget = 0;    // 0 with a directory: unbounded
};

struct PrefixStateCacheStats {
    std::uint64_t lookups = 0, host_hits = 0, disk_hits = 0, misses = 0;
    std::uint64_t inserted = 0, evicted = 0, rejected_over_budget = 0;
    std::uint64_t host_bytes = 0, host_checkpoints = 0, host_segments = 0;
    std::uint64_t disk_writes = 0, disk_write_bytes = 0, disk_write_failures = 0, disk_evictions = 0;
    std::uint64_t disk_reads = 0, disk_read_bytes = 0, disk_read_failures = 0;
    double disk_read_ms = 0.0, disk_write_ms = 0.0;
};

class PrefixStateCache final {
public:
    PrefixStateCache(PrefixStateCachePolicy, PrefixCacheAllocator);
    ~PrefixStateCache();
    PrefixStateCache(const PrefixStateCache&) = delete;
    PrefixStateCache& operator=(const PrefixStateCache&) = delete;

    // Longest checkpoint for this identity whose prefix is tokens[0, P) with P = k*block_rows <=
    // limit. Host first, then disk (a disk hit is loaded into the host tier). Null: miss.
    // The returned handle keeps its payloads alive.
    [[nodiscard]] std::shared_ptr<const PrefixCacheCheckpoint> find_longest(const ModelContentIdentity& identity,
        std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t limit);
    [[nodiscard]] bool contains(const ModelContentIdentity& digest);
    // A retained segment with this digest (host tier only), or null.
    [[nodiscard]] std::shared_ptr<const PrefixCacheSegment> find_segment(const ModelContentIdentity& digest);
    // Inserts into the host tier (LRU, evicting least recently used checkpoints to stay within the
    // budget) and queues the disk write. False if the checkpoint alone exceeds the budget.
    bool insert(std::shared_ptr<const PrefixCacheCheckpoint>);
    [[nodiscard]] PrefixCacheBuffer allocate(std::uint64_t bytes) const;
    // Waits for queued disk writes.
    void flush();
    [[nodiscard]] PrefixStateCacheStats stats();
    [[nodiscard]] const PrefixStateCachePolicy& policy() const noexcept { return policy_; }
    // Bytes a checkpoint adds to the host tier now (segments already retained are not counted).
    [[nodiscard]] std::uint64_t incremental_bytes(const PrefixCacheCheckpoint&);

private:
    struct Entry {
        std::shared_ptr<const PrefixCacheCheckpoint> checkpoint;
        std::list<ModelContentIdentity>::iterator lru;
    };
    struct DigestLess {
        bool operator()(const ModelContentIdentity& a, const ModelContentIdentity& b) const noexcept {
            return a.bytes < b.bytes;
        }
    };
    std::uint64_t host_bytes_locked();
    void touch_locked(Entry&);
    void evict_to_budget_locked(std::uint64_t reserve);
    std::shared_ptr<const PrefixCacheCheckpoint> load_from_disk(const ModelContentIdentity& identity,
        const ModelContentIdentity& digest, std::span<const std::uint32_t> tokens, std::uint32_t block_rows);
    std::shared_ptr<const PrefixCacheSegment> load_segment(const ModelContentIdentity& identity,
        const ModelContentIdentity& digest, std::uint32_t block_rows);
    void write_to_disk(const PrefixCacheCheckpoint&);
    void enforce_disk_budget();
    std::uint64_t incremental_bytes_locked(const PrefixCacheCheckpoint&);
    // Deletes temporary files and segment files no checkpoint references, older than min_age.
    void sweep_disk(std::chrono::seconds min_age);
    void writer_loop();

    PrefixStateCachePolicy policy_;
    PrefixCacheAllocator allocate_;
    std::mutex mutex_;
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
