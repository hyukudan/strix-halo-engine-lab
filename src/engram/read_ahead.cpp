#include "strix/engram/read_ahead.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace strix::engram {

ReadAhead::ReadAhead(RowReader& reader) : reader_(reader), thread_([this] { worker(); }) {}

ReadAhead::~ReadAhead() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    work_.notify_all();
    thread_.join();
}

std::uint64_t ReadAhead::submit(std::vector<RowRequest> requests) {
    std::uint64_t ticket = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ticket = next_ticket_++;
        Block& b = blocks_[ticket];
        b.requests = std::move(requests);
        if (b.requests.empty()) b.done = true;
        else queue_.push_back(ticket);
    }
    work_.notify_one();
    return ticket;
}

void ReadAhead::worker() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        work_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (queue_.empty()) return;  // stop_ with nothing queued
        const std::uint64_t ticket = queue_.front();
        queue_.pop_front();
        Block& b = blocks_.at(ticket);  // map nodes are stable; a queued block is never released before done
        const std::span<const RowRequest> requests(b.requests);
        lock.unlock();
        std::vector<std::uint8_t> bytes;
        std::exception_ptr error;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            bytes.resize(requests.size() * reader_.row_bytes());
            reader_.read(requests, bytes);
        } catch (...) {
            error = std::current_exception();
            bytes.clear();
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        lock.lock();
        b.bytes = std::move(bytes);
        b.error = error;
        b.done = true;
        ++stats_.blocks;
        stats_.rows += requests.size();
        stats_.read_seconds += seconds;
        done_.notify_all();
    }
}

std::span<const std::uint8_t> ReadAhead::wait(std::uint64_t ticket) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    const auto it = blocks_.find(ticket);
    if (it == blocks_.end()) throw std::out_of_range("Engram read-ahead: unknown or released ticket");
    Block& b = it->second;
    done_.wait(lock, [&b] { return b.done; });
    stats_.wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (b.error) std::rethrow_exception(b.error);
    return b.bytes;
}

void ReadAhead::release(std::uint64_t ticket) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto it = blocks_.find(ticket);
    if (it == blocks_.end()) return;
    done_.wait(lock, [&it] { return it->second.done; });  // the worker may still be writing it
    blocks_.erase(it);
}

std::size_t ReadAhead::held() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.size();
}

ReadAheadStats ReadAhead::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace strix::engram
