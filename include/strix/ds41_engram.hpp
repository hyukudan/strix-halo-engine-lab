#pragma once

// deepseek41 Engram n-gram memory, host side: the hash of every position (from the admitted
// EngramContract) and a concurrent reader of 264-byte rows from the model file on disk.
//
// Hash of position t at Engram layer e: the compressed ids of tokens t, t-1, ..., t-(N-1) (N =
// max_ngram_size; a position before the start of the sequence, and every position after it once one
// is blocked, reads the compressed pad id), products p_s = id_s * multiplier[e][s] in int64, the
// running XOR r_1 = p_0 ^ p_1, r_2 = r_1 ^ p_2, ...; the (s + 1)-gram's row for head h is
// r_s % prime[e][s - 1][h] + offset[e][(s - 1) * heads + h], offsets being the running sum of the
// layer's primes in that order. Columns are ordered n-gram size major, head minor.

#include "strix/deepseek41_model.hpp"

#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace strix::ds41 {

class EngramHasher {
public:
    explicit EngramHasher(const deepseek41::EngramContract& contract);
    [[nodiscard]] std::uint32_t layers() const noexcept { return static_cast<std::uint32_t>(contract_.layer_ids.size()); }
    [[nodiscard]] std::uint32_t columns() const noexcept { return contract_.rows_per_token(); }
    // Appends `token` to the sequence and writes its rows, [layer][column] (layers() * columns()).
    void push(std::uint32_t token, std::span<std::uint64_t> rows);
    // Rows of every token of a sequence starting at position 0 (prompt): [token][layer][column].
    [[nodiscard]] std::vector<std::uint64_t> hash_sequence(std::span<const std::uint32_t> tokens);
    [[nodiscard]] std::uint64_t positions() const noexcept { return history_.size(); }
    // Verify windows: drop the trailing positions so the history ends after `positions` tokens.
    void truncate(std::uint64_t positions);
    void reset() noexcept { history_.clear(); }

private:
    deepseek41::EngramContract contract_;
    std::vector<std::uint64_t> offsets_;  // [layer][column]
    std::uint32_t pad_ = 0;
    std::vector<std::uint32_t> history_;  // compressed ids
};

// Location of one Engram table in the model file.
struct EngramTable {
    std::uint64_t file_offset = 0;  // absolute offset of row 0
    std::uint64_t rows = 0;
};
struct EngramRowRequest {
    std::uint32_t table = 0;
    std::uint64_t row = 0;
};
struct EngramReaderStats {
    std::uint64_t requests = 0, page_reads = 0, page_bytes = 0, batches = 0;
    double busy_seconds = 0.0;  // wall time inside read()
};

// Random 264-byte row reads from the model file by a pool of worker threads. With direct I/O
// (default) every row is read as the aligned 4 KiB page(s) holding it (O_DIRECT), which leaves the
// page cache to the resident model; without it plain pread is used. read() blocks until every row
// of the batch is in `out` (request i at out[i * 264]) and throws on any short or failed read.
class EngramRowReader {
public:
    EngramRowReader(const std::filesystem::path& file, std::vector<EngramTable> tables, unsigned threads = 16,
                    bool direct_io = true);
    ~EngramRowReader();
    EngramRowReader(const EngramRowReader&) = delete;
    EngramRowReader& operator=(const EngramRowReader&) = delete;
    void read(std::span<const EngramRowRequest> requests, std::span<std::uint8_t> out);
    [[nodiscard]] bool direct_io() const noexcept { return direct_; }
    [[nodiscard]] EngramReaderStats stats() const;

private:
    void worker();
    void read_one(const EngramRowRequest& request, std::uint8_t* out, std::uint8_t* page);
    int fd_ = -1;
    bool direct_ = false;
    std::uint64_t file_bytes_ = 0;
    std::vector<EngramTable> tables_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::span<const EngramRowRequest> batch_;
    std::uint8_t* out_ = nullptr;
    std::uint64_t generation_ = 0;
    std::atomic<std::size_t> next_{0};
    std::size_t finished_ = 0;
    unsigned active_ = 0;
    bool stop_ = false, failed_ = false;
    std::string error_;
    EngramReaderStats stats_;
    std::atomic<std::uint64_t> page_reads_{0}, page_bytes_{0};
};

// Worker threads of the decode runners' Engram reader: STRIX_DS41_ENGRAM_THREADS=N (1..256), else `fallback`.
// Only the number of reads in flight changes; every row is the same bytes.
[[nodiscard]] inline unsigned engram_reader_threads(unsigned fallback) {
    const char* v = std::getenv("STRIX_DS41_ENGRAM_THREADS");
    if (v == nullptr || *v == 0) return fallback;
    const unsigned long n = std::strtoul(v, nullptr, 10);
    return n >= 1UL && n <= 256UL ? static_cast<unsigned>(n) : fallback;
}

// Engram table locations of an admitted target from its manifest.
[[nodiscard]] std::vector<EngramTable> engram_tables(const deepseek41::TargetModel& model);

} // namespace strix::ds41
