#pragma once
// Portable positional reads of a large model file (POSIX pread / Win32 ReadFile), optionally unbuffered (O_DIRECT /
// FILE_FLAG_NO_BUFFERING: the page cache stays with whoever else uses the machine), from a pool of worker threads.
// Two uses on the deepseek41 prefill: byte ranges of routed-expert tensors streamed into pinned staging buffers, and
// the random 264-byte Engram rows (each read as the aligned 4 KiB page(s) holding it).
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace strix::ds41 {

// One caller at a time: read_range / read_records from two threads need two FileReader objects.
class FileReader {
public:
    // direct: unbuffered reads (offsets, sizes and destinations aligned to `alignment`; read_range handles unaligned
    // requests through a bounce page).
    explicit FileReader(const std::filesystem::path& path, bool direct = true, unsigned threads = 8);
    ~FileReader();
    FileReader(const FileReader&) = delete;
    FileReader& operator=(const FileReader&) = delete;

    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] bool direct() const noexcept { return direct_; }
    static constexpr std::size_t alignment = 4096;

    // Bytes [offset, offset + bytes) into dst, split into pieces of `piece` bytes read by the pool in parallel; blocks
    // until every piece is in. Unaligned head / tail pieces go through a bounce page when direct.
    void read_range(std::uint64_t offset, std::size_t bytes, void* dst, std::size_t piece = std::size_t{8} << 20);

    // Gather of `count` records of `record` bytes at file offsets offsets[i] into out + i * record (records do not
    // cross a 4 KiB page boundary more than once). Every distinct page is read once, in file order (records sharing a
    // page, and repeated records, cost one read), or taken from the page cache.
    // per_job: records (pages with the cache) one pool job reads in turn (32 for prompt batches; 1 spreads a decode
    // step's few records over the pool so their latencies overlap).
    void read_records(std::span<const std::uint64_t> offsets, std::size_t record, std::uint8_t* out, std::size_t per_job = 32);
    // The same gather issued from the calling thread as one overlapped read per record (all in flight at once, no pool
    // wake-ups, no page cache): the latency of a decode step's few records (Windows; elsewhere read_records with one
    // record per job).
    void read_records_queued(std::span<const std::uint64_t> offsets, std::size_t record, std::uint8_t* out);
    // An LRU cache of the pages read_records reads, kept across calls (bytes rounded down to pages; 0: none).
    void set_page_cache(std::size_t bytes);

    struct Stats {
        std::uint64_t bytes = 0, reads = 0;
        double seconds = 0.0;  // wall time inside read_range / read_records
        double issue_seconds = 0.0;  // read_records_queued: time spent issuing the reads (ReadFile calls)
        std::uint64_t records = 0, pages = 0, page_hits = 0;  // read_records: distinct records, distinct pages, cached
    };
    [[nodiscard]] Stats stats() const;

private:
    void run_parallel(std::size_t jobs, const std::function<void(std::size_t job, std::uint8_t* page)>& body);
    void pread_exact(std::uint64_t offset, std::size_t bytes, void* dst);
    void pread_page(std::uint64_t offset, std::size_t bytes, std::uint8_t* dst);
    void worker();

    std::filesystem::path path_;
    bool direct_ = true;
    std::uint64_t size_ = 0;
#ifdef _WIN32
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(std::size_t, std::uint8_t*)>* job_ = nullptr;
    std::size_t job_count_ = 0, next_job_ = 0, finished_ = 0;
    std::uint64_t generation_ = 0;
    bool stop_ = false;
    std::string error_;
    Stats stats_;
    // page cache: slots of 4 KiB in one aligned block, an index by page offset, an intrusive LRU list
    std::uint8_t* cache_mem_ = nullptr;
    std::size_t cache_slots_ = 0;
    std::vector<std::uint64_t> slot_page_;
    std::vector<std::uint32_t> lru_prev_, lru_next_, slot_stamp_;
    std::uint32_t lru_head_ = 0, lru_tail_ = 0, stamp_ = 0;
    struct PageIndex;
    std::unique_ptr<PageIndex> index_;
};

}  // namespace strix::ds41
