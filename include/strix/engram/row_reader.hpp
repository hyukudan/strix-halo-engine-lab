#pragma once

// Concurrent reader of Engram rows from a model file (residency file_direct / file_cached). With direct I/O every
// row is read as the aligned 4 KiB page(s) holding it (O_DIRECT), which leaves the page cache to the resident model;
// without it plain pread is used. On Linux with direct I/O the batch goes through one deep read queue
// (core/direct_row_fetch.hpp) instead of one positional read per row on the worker threads: the same bytes.
// Optional CLOCK payload cache with batch offset deduplication, sized by the caller's byte budget.
// read() blocks until every row of the batch is in `out` (request i at out[i * row_bytes]) and throws on any short
// or failed read. Row size comes from the family's RowFormat (strix/engram/spec.hpp).

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace strix::core {
class DirectRowFetcher;
}

namespace strix::engram {

// Location of one Engram layer table in the model file.
struct Table {
    std::uint64_t file_offset = 0;  // absolute offset of row 0
    std::uint64_t rows = 0;
};
struct RowRequest {
    std::uint32_t table = 0;
    std::uint64_t row = 0;
};
struct ReaderStats {
    std::uint64_t requests = 0, page_reads = 0, page_bytes = 0, batches = 0;
    // Unique offsets served by the payload cache / disk, and repeated offsets within a batch.
    std::uint64_t payload_hits = 0, payload_misses = 0, batch_duplicates = 0;
    std::uint64_t payload_hit_bytes = 0, duplicate_bytes = 0, output_bytes = 0;
    std::uint64_t cache_capacity = 0, cache_entries = 0, cache_storage_bytes = 0, cache_evictions = 0;
    double busy_seconds = 0.0;  // wall time inside read()
};

struct ReaderOptions {
    std::uint32_t row_bytes = 0;            // RowFormat::bytes(), 1 .. 4096
    unsigned threads = 16;                  // 1 .. 256 worker threads
    bool direct_io = true;
    std::uint64_t payload_cache_bytes = 0;  // 0: no payload cache
    bool ring_io = true;                    // direct I/O on Linux: one deep read queue
    unsigned ring_depth = 256;              // reads in flight on the queue (0: worker threads)
};

class RowReader {
public:
    RowReader(const std::filesystem::path& file, std::vector<Table> tables, const ReaderOptions& options);
    ~RowReader();
    RowReader(const RowReader&) = delete;
    RowReader& operator=(const RowReader&) = delete;
    void read(std::span<const RowRequest> requests, std::span<std::uint8_t> out);
    [[nodiscard]] std::uint32_t row_bytes() const noexcept { return row_bytes_; }
    [[nodiscard]] bool direct_io() const noexcept { return direct_; }
    [[nodiscard]] bool ring_io() const;  // effective backend, serialized with reads
    [[nodiscard]] bool payload_cache() const noexcept;  // a payload cache with room (rows read once are served again)
    [[nodiscard]] ReaderStats stats() const;

private:
    struct PayloadCache;
    void worker();
    void read_one(const RowRequest& request, std::uint8_t* out, std::uint8_t* page);
    std::uint64_t offset_of(const RowRequest& r) const { return tables_[r.table].file_offset + r.row * row_bytes_; }
    int fd_ = -1;
    bool direct_ = false;
    std::uint32_t row_bytes_ = 0;
    std::uint64_t file_bytes_ = 0;
    std::vector<Table> tables_;
    std::vector<std::thread> threads_;
    std::unique_ptr<core::DirectRowFetcher> fetcher_;
    std::unique_ptr<PayloadCache> cache_;
    mutable std::mutex read_mutex_;  // serializes callers and statistics, including async reader clients
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::span<const RowRequest> batch_;
    std::span<const std::size_t> destinations_;
    std::uint8_t* out_ = nullptr;
    std::uint64_t generation_ = 0;
    std::atomic<std::size_t> next_{0};
    std::size_t finished_ = 0;
    unsigned active_ = 0;
    bool stop_ = false, failed_ = false;
    std::string error_;
    ReaderStats stats_;
    std::atomic<std::uint64_t> page_reads_{0}, page_bytes_{0};
};

} // namespace strix::engram
