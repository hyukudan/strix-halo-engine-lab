#pragma once
// The HIP runtime's own host memory, kept off the reclaim LRU (HIP backend core, host-only, no HIP headers).
//
// Why: on integrated GPUs every host allocation of the runtime (AQL queues, kernel arguments, signals, code objects,
// small hipHostMalloc buffers) is a KFD userptr BO over SHARED shmem pages ("rw-s /dev/zero (deleted)"). When reclaim
// or compaction moves ONE of those pages, the MMU notifier evicts every queue of the process until the restore worker
// revalidates them (measured: 9-770 ms inside timed MiniMax pair prefills, right after the load and once ~32 s later,
// while compaction ran). Engine buffers of at least 2 MiB are already out of reach (pinned_host.hpp: private anonymous pages with a
// long-term pin, which compaction skips). The runtime's shmem cannot take that path: compaction isolates a pinned
// shmem page and fires the notifier anyway. mlock moves those pages to the unevictable LRU: reclaim never takes them,
// and compaction skips them once the system runs with vm.compact_unevictable_allowed=0 (a system setting, not applied
// here; with the kernel default 1, compaction still migrates mlocked pages). MLOCK_ONFAULT locks the resident pages
// and later faults only: nothing is populated, no memory is committed that the runtime did not touch. No byte the GPU
// reads or writes changes: bitwise by construction.
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strix::host_map_lock {

struct Range {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

namespace detail {
[[nodiscard]] inline bool hex(std::string_view s, std::uintptr_t& v) {
    if (s.empty() || s.size() > 2 * sizeof(std::uintptr_t)) return false;
    v = 0;
    for (const char c : s) {
        unsigned d = 0;
        if (c >= '0' && c <= '9') d = static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') d = static_cast<unsigned>(c - 'a') + 10U;
        else if (c >= 'A' && c <= 'F') d = static_cast<unsigned>(c - 'A') + 10U;
        else return false;
        v = (v << 4U) | d;
    }
    return true;
}
// Next space-separated field of `line` from `pos` (leading spaces skipped); empty at the end.
[[nodiscard]] inline std::string_view field(std::string_view line, std::size_t& pos) {
    while (pos < line.size() && line[pos] == ' ') ++pos;
    const std::size_t start = pos;
    while (pos < line.size() && line[pos] != ' ') ++pos;
    return line.substr(start, pos - start);
}
}  // namespace detail

// One /proc/<pid>/maps line ("begin-end perms offset dev inode   path") -> its range when it is a readable, writable,
// SHARED shmem mapping: anonymous shared memory (/dev/zero (deleted), [anon_shmem:NAME]), memfd or SysV segments, i.e.
// the runtime's host allocations. Device files (/dev/kfd, /dev/dri/*), file mappings, private and read-only maps are
// not (the GPU's own BOs are device-file mappings of TTM pages, which never migrate).
[[nodiscard]] inline bool runtime_shared_line(std::string_view line, Range& out) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.remove_suffix(1);
    std::size_t pos = 0;
    const std::string_view range = detail::field(line, pos);
    const std::string_view perms = detail::field(line, pos);
    (void)detail::field(line, pos);  // offset
    (void)detail::field(line, pos);  // device
    if (detail::field(line, pos).empty()) return false;  // inode: a short line is not a maps entry
    while (pos < line.size() && line[pos] == ' ') ++pos;
    const std::string_view path = line.substr(pos);
    if (perms.size() != 4 || perms[0] != 'r' || perms[1] != 'w' || perms[3] != 's') return false;
    const bool shmem = path == "/dev/zero (deleted)" || path.substr(0, 7) == "/memfd:" || path.substr(0, 5) == "/SYSV" ||
                       path.substr(0, 12) == "[anon_shmem:";
    if (!shmem) return false;
    const std::size_t dash = range.find('-');
    if (dash == std::string_view::npos) return false;
    Range r;
    if (!detail::hex(range.substr(0, dash), r.begin) || !detail::hex(range.substr(dash + 1), r.end) || r.end <= r.begin)
        return false;
    out = r;
    return true;
}

// Every runtime shmem range of a whole maps text, in file order.
[[nodiscard]] inline std::vector<Range> runtime_shared_ranges(std::string_view maps) {
    std::vector<Range> ranges;
    while (!maps.empty()) {
        const std::size_t eol = maps.find('\n');
        const std::string_view line = maps.substr(0, eol);
        maps = eol == std::string_view::npos ? std::string_view{} : maps.substr(eol + 1);
        Range r;
        if (runtime_shared_line(line, r)) ranges.push_back(r);
    }
    return ranges;
}

// One pass: how many runtime shmem maps exist and how many bytes they span, how many were locked by this pass
// (new since the previous one) and how many refused the lock (RLIMIT_MEMLOCK, unmapped meanwhile).
struct Pass {
    std::size_t maps = 0;
    std::uint64_t bytes = 0;
    std::size_t fresh = 0;
    std::size_t refused = 0;
};

// Decides which ranges of a pass need the lock call: those not seen with the same extent in the previous pass (a
// mapping replaced at the same address is locked again; one that refused is not retried until it changes).
class Tracker {
public:
    template <class Lock>
    Pass apply(const std::vector<Range>& ranges, Lock&& lock) {
        Pass p;
        std::map<std::uintptr_t, std::uintptr_t> now;
        for (const Range& r : ranges) {
            ++p.maps;
            p.bytes += r.end - r.begin;
            now[r.begin] = r.end;
            const auto it = seen_.find(r.begin);
            if (it != seen_.end() && it->second == r.end) continue;
            if (lock(r)) ++p.fresh;
            else ++p.refused;
        }
        seen_ = std::move(now);
        return p;
    }

private:
    std::map<std::uintptr_t, std::uintptr_t> seen_;
};

#if defined(__linux__)
#ifndef MLOCK_ONFAULT
#define MLOCK_ONFAULT 0x01
#endif
// /proc/self/maps through plain read(2) (no iostreams: the background pass may still run while static objects of the
// process are destroyed at exit).
[[nodiscard]] inline std::string read_self_maps() {
    std::string text;
    const int fd = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return text;
    char buf[16384];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n <= 0) break;
        text.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fd);
    return text;
}

[[nodiscard]] inline bool lock_on_fault(const Range& r) noexcept {
#ifdef SYS_mlock2
    return ::syscall(SYS_mlock2, reinterpret_cast<void*>(r.begin), static_cast<std::size_t>(r.end - r.begin), MLOCK_ONFAULT) == 0;
#else
    (void)r;
    return false;
#endif
}

// One pass over this process: mlock2(MLOCK_ONFAULT) of every runtime shmem range new since the previous pass.
[[nodiscard]] inline Pass lock_runtime_shared_maps(Tracker& tracker) {
    return tracker.apply(runtime_shared_ranges(read_self_maps()), [](const Range& r) { return lock_on_fault(r); });
}
#endif

}  // namespace strix::host_map_lock
