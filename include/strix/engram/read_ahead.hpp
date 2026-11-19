#pragma once

// Read-ahead of Engram rows whose token ids are known before the layer that needs them (a prompt window, the next
// window of the same prompt). One worker thread reads the submitted blocks through a RowReader, in submission order,
// into buffers it owns. The consumer waits per block. A block's bytes are what RowReader::read returns for the same
// requests, so moving a read ahead changes no bit. Vendor-free: the caller copies the bytes to its device.
//
// Bounded by the caller: a block's buffer lives from submit() until release(). An I/O failure is that block's
// failure: wait() rethrows it, and the blocks after it are still read.

#include "strix/engram/row_reader.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace strix::engram {

struct ReadAheadStats {
    std::uint64_t blocks = 0, rows = 0;
    double read_seconds = 0.0;  // worker time inside RowReader::read
    double wait_seconds = 0.0;  // consumer time blocked in wait() (the read not hidden by other work)
};

class ReadAhead {
public:
    explicit ReadAhead(RowReader& reader);
    ~ReadAhead();  // finishes the queued blocks, then joins
    ReadAhead(const ReadAhead&) = delete;
    ReadAhead& operator=(const ReadAhead&) = delete;

    // Queues one block. Returns its ticket (increasing from 0).
    std::uint64_t submit(std::vector<RowRequest> requests);
    // Blocks until the ticket's rows are read. Returns requests.size() x row_bytes bytes (request i at i x row_bytes),
    // valid until release(ticket). Rethrows the block's read error. Throws std::out_of_range for an unknown or
    // released ticket.
    std::span<const std::uint8_t> wait(std::uint64_t ticket);
    void release(std::uint64_t ticket);
    // Tickets submitted and not yet released.
    [[nodiscard]] std::size_t held() const;
    [[nodiscard]] ReadAheadStats stats() const;

private:
    struct Block {
        std::vector<RowRequest> requests;
        std::vector<std::uint8_t> bytes;
        std::exception_ptr error;
        bool done = false;
    };
    void worker();
    RowReader& reader_;
    mutable std::mutex mutex_;
    std::condition_variable work_, done_;
    std::map<std::uint64_t, Block> blocks_;
    std::deque<std::uint64_t> queue_;
    std::uint64_t next_ticket_ = 0;
    bool stop_ = false;
    ReadAheadStats stats_;
    std::thread thread_;
};

} // namespace strix::engram
