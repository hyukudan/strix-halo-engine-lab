#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix {
class RegisteredPageLockError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// Preserve the automatic lock/off policy. Only an explicit =1 requires success.
struct RegisteredPageLockPolicy {
    bool enabled;
    bool required;
    explicit RegisteredPageLockPolicy(const char* setting)
        : enabled(!(setting && setting[0] == '0')),
          required(setting && std::string_view(setting) == "1") {}

    template<class Warn>
    void failed(const std::string& reason, Warn warn) {
        if (required) throw RegisteredPageLockError(reason);
        warn();
        enabled = false; // As before, stop locking subsequent ranges after failure.
    }
};
// Track a successful registration before querying its alias. An alias failure
// must unwind through the owner, never release an untracked live registration.
template<class Register, class Track, class Alias>
void acquire_registered_backing(Register reg, Track track, Alias alias) {
    reg();
    track();
    alias(); // A new registration always obtains a fresh device alias.
}
// ORDER (no VMA change next to a live registration): locking or unlocking a range changes its VMA flags, so the
// kernel splits the VMA at the range boundary. A boundary inside a PMD-mapped file folio splits that PMD, which
// invalidates the CPU mapping of the neighbouring range; if that neighbour is registered with the GPU, the driver
// evicts every queue of the process. Hence: lock every range before the first registration, and unregister every
// range before the first unlock. Same pages, flags and locked bytes as locking each range right after it is
// registered; only the order changes.
//
// lock(range) -> bool locked (may throw); acquire(range, locked, handed) registers one range and calls handed() once
// the owner tracks it (from then on the owner unlocks it). On any exception every locked range the owner does not
// track is unlocked here before rethrowing.
template<class Ranges, class Lock, class Unlock, class Acquire>
void acquire_locked_backings(const Ranges& ranges, Lock lock, Unlock unlock, Acquire acquire) {
    std::vector<char> locked(ranges.size(), 0);
    std::size_t owned = 0;  // ranges [0, owned) are tracked by the owner
    try {
        for (std::size_t i = 0; i < ranges.size(); ++i) locked[i] = lock(ranges[i]) ? 1 : 0;
        for (std::size_t i = 0; i < ranges.size(); ++i) acquire(ranges[i], locked[i] != 0, [&owned, i] { owned = i + 1; });
    } catch (...) {
        for (std::size_t i = owned; i < ranges.size(); ++i)
            if (locked[i]) unlock(ranges[i]);
        throw;
    }
}
// Teardown unlock in runs: an unregistered range can still be mapped to the GPU until it is unmapped (measured: unlocking
// each piece after every piece was unregistered still evicted the process's queues ~0.7-1.1 s), so unlock never splits a
// locked VMA. Address-ordered contiguous ranges are merged into one unlock call whose ends are the locked run's own VMA
// boundaries. add() order must be ascending for a run to merge; any other order only costs more calls. No allocation.
template<class Unlock>
class CoalescedUnlock {
public:
    explicit CoalescedUnlock(Unlock unlock) : unlock_(unlock) {}
    void add(std::uintptr_t base, std::uint64_t bytes) {
        if (bytes == 0) return;
        if (bytes_ != 0 && base == base_ + bytes_) { bytes_ += bytes; return; }
        flush();
        base_ = base;
        bytes_ = bytes;
    }
    void flush() {
        if (bytes_ != 0) unlock_(base_, bytes_);
        bytes_ = 0;
    }
private:
    Unlock unlock_;
    std::uintptr_t base_ = 0;
    std::uint64_t bytes_ = 0;
};
// Consumers must already be closed to new work. A failed drain/unregister
// forbids unlocking/discarding the affected backing; the owner retains it.
// Every unregister precedes the first unlock (ORDER above).
template<class Ranges, class Drain, class Unregister, class Unlock>
bool retire_registered_backings(const Ranges& ranges, Drain drain, Unregister unreg, Unlock unlock) {
    if (ranges.empty()) return true;
    drain();
    std::vector<char> unregistered;
    unregistered.reserve(ranges.size());
    for (const auto& range : ranges) unregistered.push_back(unreg(range) ? 1 : 0);
    bool complete = true;
    std::size_t index = 0;
    for (const auto& range : ranges) {
        if (unregistered[index++]) unlock(range);
        else complete = false;
    }
    return complete;
}
} // namespace strix
