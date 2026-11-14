#pragma once
// A read-ahead of blocks that two ranks can balance between them (shared: any engine whose ranks each read their own
// rows from their own disk while the layers before the first consumer run). Each rank reads its blocks in order on one
// reader thread. At a point both ranks reach in the same order (a group of blocks is about to be consumed) they
// exchange a ReadProgress; from the two, with the same integer arithmetic on both sides, the rank that would finish
// later gives the tail blocks of that group to the other one, which reads them from its own copy of the file and ships
// the bytes. The split makes both finish together at the rates measured on this very run, so nothing is configured and
// a rank whose rows are cached gives or takes nothing. The bytes are the file's on either rank: the consumer sees the
// same rows whoever read them.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::core {

struct ReadProgress {
    std::uint64_t open = 0;       // units of the group this rank has not read yet (in flight, reserved or free)
    std::uint64_t free = 0;       // of those, the units of blocks [free_from, blocks): nobody has started them
    std::uint64_t free_from = 0;  // first free block
    std::uint64_t rate = 0;       // units per second read on this run so far (0: nothing measured)
};

// First block of the owner's group the helper reads (owner_units.size(): none). The helper takes x units with
// (helper.open + x) / helper.rate = (owner.open - x) / owner.rate, rounded down to whole blocks from the end; nothing
// when the owner is not the later one, when a rate is unknown, or when the owner would gain less than min_gain_us.
inline std::size_t balanced_tail(const ReadProgress& helper, const ReadProgress& owner, std::span<const std::uint32_t> owner_units,
                                 std::uint64_t min_gain_us) {
    const std::size_t none = owner_units.size();
    constexpr std::uint64_t limit = std::uint64_t{1} << 31U;  // products below stay inside 64 bits
    if (helper.rate == 0U || owner.rate == 0U || helper.rate >= limit || owner.rate >= limit || helper.open >= limit ||
        owner.open >= limit || owner.free > owner.open || owner.free_from > none)
        return none;
    const std::uint64_t later = owner.open * helper.rate, sooner = helper.open * owner.rate;
    if (later <= sooner) return none;
    const std::uint64_t x = std::min((later - sooner) / (helper.rate + owner.rate), owner.free);
    std::size_t first = none;
    std::uint64_t taken = 0;
    while (first > owner.free_from && taken + owner_units[first - 1U] <= x) taken += owner_units[--first];
    if (taken == 0U || taken * 1000000U / owner.rate < min_gain_us) return none;
    return first;
}

class BalancedReadQueue {
public:
    // units[group][block]: the size of every block this rank owns (rows, bytes: any unit the rates are counted in)
    explicit BalancedReadQueue(std::vector<std::vector<std::uint32_t>> units) : units_(std::move(units)) {
        for (const auto& g : units_) {
            std::uint64_t open = 0;
            for (const auto u : g) open += u;
            groups_.push_back({0, g.size(), g.size(), open, 0});
        }
    }
    std::span<const std::uint32_t> units(std::size_t group) const { return units_.at(group); }

    // The reader thread. read(group, block): one own block; publish(group, block): it is readable (after read, or after
    // clear when a read failed: consumers never hang behind an I/O error); help(group, first): the peer's blocks from
    // `first` on. Returns after close(), when every own block is read or given away; rethrows the first read error.
    template <class Read, class Publish, class Clear, class Help> void run(Read read, Publish publish, Clear clear, Help help) {
        std::unique_lock<std::mutex> lock(mutex_);
        start_ = std::chrono::steady_clock::now();
        for (;;) {
            if (help_) {
                const auto task = *help_;
                help_.reset();
                lock.unlock();
                std::exception_ptr error;
                try {
                    if (error_) std::rethrow_exception(error_);
                    help(task.first, task.second);
                } catch (...) {
                    error = std::current_exception();
                }
                lock.lock();
                help_error_ = error;
                help_done_ = true;
                changed_.notify_all();
                continue;
            }
            std::size_t g = 0;
            bool frozen = false;
            for (; g < groups_.size(); ++g) {
                if (groups_[g].next < groups_[g].limit) break;
                if (groups_[g].limit < groups_[g].end) {  // waiting for resolve(): do not start a later group meanwhile
                    frozen = true;
                    break;
                }
            }
            if (frozen || g == groups_.size()) {
                if (!frozen && closed_) break;
                changed_.wait(lock);
                continue;
            }
            const std::size_t block = groups_[g].next++;
            ++groups_[g].busy;
            const bool failed = static_cast<bool>(error_);
            lock.unlock();
            std::exception_ptr error;
            if (!failed) {
                try {
                    read(g, block);
                } catch (...) {
                    error = std::current_exception();
                }
            }
            if (failed || error) clear(g, block);
            publish(g, block);
            lock.lock();
            if (error) error_ = error;
            groups_[g].open -= units_[g][block];
            --groups_[g].busy;
            changed_.notify_all();
            if (!failed && !error) {
                done_ += units_[g][block];
                done_at_ = std::chrono::steady_clock::now();
            }
        }
        if (error_) std::rethrow_exception(error_);
    }

    // The consumer thread, when `group` is about to be consumed: at most one more block of it starts until resolve().
    ReadProgress freeze(std::size_t group) {
        std::lock_guard<std::mutex> lock(mutex_);
        Group& g = groups_.at(group);
        g.limit = std::min(g.end, g.next + 1U);
        ReadProgress p;
        p.open = g.open;
        p.free_from = g.limit;
        for (std::size_t b = g.limit; b < g.end; ++b) p.free += units_[group][b];
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(done_at_ - start_).count();
        if (done_ != 0U && us > 0) p.rate = std::min<std::uint64_t>(done_ * 1000000U / static_cast<std::uint64_t>(us), (std::uint64_t{1} << 31U) - 1U);
        return p;
    }
    // Blocks [give_from, end) of the group are the peer's now (give_from = block count: none); the reader goes on.
    void resolve(std::size_t group, std::size_t give_from) {
        std::lock_guard<std::mutex> lock(mutex_);
        Group& g = groups_.at(group);
        if (give_from < g.limit || give_from > g.end) throw std::logic_error("balanced read: split inside started blocks");
        for (std::size_t b = give_from; b < g.end; ++b) g.open -= units_[group][b];
        g.limit = g.end = give_from;
        changed_.notify_all();
    }
    // Every block of the group this rank kept is read (after resolve()).
    void wait_own(std::size_t group) {
        std::unique_lock<std::mutex> lock(mutex_);
        const Group& g = groups_.at(group);
        changed_.wait(lock, [&] { return g.next == g.end && g.busy == 0U; });
    }
    // The peer's blocks [first, ...) of its group: read next, before more own blocks. One request at a time.
    void help(std::size_t group, std::size_t first) {
        std::lock_guard<std::mutex> lock(mutex_);
        help_ = std::pair<std::size_t, std::size_t>(group, first);
        help_done_ = false;
        changed_.notify_all();
    }
    void wait_help() {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return help_done_; });
        if (help_error_) std::rethrow_exception(help_error_);
    }
    // No further freeze / help: run() returns once the own blocks are done. Safe to call more than once.
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& g : groups_) g.limit = g.end;
        closed_ = true;
        changed_.notify_all();
    }

private:
    struct Group {
        std::size_t next, limit, end;  // next block to start; blocks the reader may start; blocks this rank reads
        std::uint64_t open;
        std::size_t busy;  // blocks being read
    };
    std::vector<std::vector<std::uint32_t>> units_;
    std::vector<Group> groups_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::optional<std::pair<std::size_t, std::size_t>> help_;
    bool help_done_ = false, closed_ = false;
    std::exception_ptr error_, help_error_;
    std::uint64_t done_ = 0;
    std::chrono::steady_clock::time_point start_{}, done_at_{};
};

}  // namespace strix::core
