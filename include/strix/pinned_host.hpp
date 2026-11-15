#pragma once
// Host memory the GPU maps, kept out of reach of page migration and reclaim (HIP backend core, host-only).
//
// Why: on this stack every hipHostMalloc is a KFD userptr BO over shmem pages. When compaction (kcompactd, or
// direct compaction under the page-cache churn of other CPU jobs) or reclaim moves ONE page of such a buffer, the
// MMU notifier evicts EVERY queue of the process and the restore worker revalidates the WHOLE BO before the queues
// run again: hundreds of ms for a 100-400 MB exchange buffer, inside timed prefills (wave4/next1112/KFDEVICT-RESULT.md).
// Pinning shmem does not help (compaction retries the unmovable pages and evicts again), so the lever allocates
// engine-owned PRIVATE anonymous pages instead: MADV_DONTFORK, mlock (off the reclaim LRU), a long-term pin through
// io_uring buffer registration (compaction skips anonymous pages with extra references), then hipHostRegister
// (fine-grained for hipHostMallocCoherent requests, coarse-grained otherwise). Same bytes, same addresses semantics
// for the caller: bitwise output. Off: the exact hipHostMalloc / hipHostFree calls.
//
// Lever STRIX_HOST_ANON_PIN (bitwise; opt-in until its in-app A/B lands as default). Only allocations of at least
// kMinBytes take the pinned path: small buffers are revalidated cheaply and keep hipHostMalloc.
#include "strix/lever.hpp"

#include <hip/hip_runtime.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace strix::host_pin {

inline constexpr std::string_view kLeverName = "host.anon_pin";
inline constexpr lever::Decl kLever{kLeverName, "STRIX_HOST_ANON_PIN", lever::Exactness::bitwise, false, {}, {},
                                    "wave4/next1112/KFDEVICT-RESULT.md (micro A/B under memory pressure)"};
inline constexpr std::size_t kMinBytes = std::size_t{2} << 20;

struct Pinned { std::size_t bytes = 0; int ring = -1; bool locked = false; };
struct Registry { std::mutex m; std::map<void*, Pinned> live; std::size_t bytes = 0, unpinned = 0; };
inline Registry& registry() { static Registry r; return r; }

[[nodiscard]] inline bool enabled() {
    static const bool on = lever::select(kLever, lever::Inputs{}).on;
    return on;
}

// io_uring fixed-buffer registration takes a FOLL_LONGTERM pin on every page (1 GiB per iovec). -1 when refused.
[[nodiscard]] inline int long_term_pin(void* base, std::size_t bytes) noexcept {
    io_uring_params p{};
    const int fd = static_cast<int>(::syscall(__NR_io_uring_setup, 1U, &p));
    if (fd < 0) return -1;
    std::vector<iovec> iov;
    for (std::size_t o = 0; o < bytes; o += std::size_t{1} << 30)
        iov.push_back({static_cast<char*>(base) + o, std::min(bytes - o, std::size_t{1} << 30)});
    if (::syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS, iov.data(), static_cast<unsigned>(iov.size())) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

// hipHostMalloc(host, bytes, flags) with the lever applied.
[[nodiscard]] inline hipError_t allocate(void** host, std::size_t bytes, unsigned flags) {
    if (!enabled() || bytes < kMinBytes) return hipHostMalloc(host, bytes, flags);
    const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t size = (bytes + page - 1) / page * page;
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) return hipErrorOutOfMemory;
    (void)::madvise(p, size, MADV_DONTFORK);
    Pinned rec{size, -1, ::mlock(p, size) == 0};
    rec.ring = long_term_pin(p, size);
    const unsigned reg = hipHostRegisterMapped | ((flags & hipHostMallocCoherent) ? 0U : hipExtHostRegisterCoarseGrained);
    if (const hipError_t e = hipHostRegister(p, size, reg); e != hipSuccess) {
        if (rec.ring >= 0) ::close(rec.ring);
        ::munmap(p, size);
        return e;
    }
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.m);
    if (rec.ring < 0 || !rec.locked) ++r.unpinned;
    r.bytes += size;
    r.live.emplace(p, rec);
    *host = p;
    return hipSuccess;
}

// hipHostFree for memory from allocate() (either path).
inline hipError_t release(void* host) noexcept {
    if (host == nullptr) return hipSuccess;
    Pinned rec;
    {
        auto& r = registry();
        std::lock_guard<std::mutex> lock(r.m);
        const auto it = r.live.find(host);
        if (it == r.live.end()) return hipHostFree(host);
        rec = it->second;
        r.bytes -= rec.bytes;
        r.live.erase(it);
    }
    const hipError_t e = hipHostUnregister(host);
    if (rec.ring >= 0) ::close(rec.ring);
    ::munmap(host, rec.bytes);
    return e;
}

// One receipt line (bytes held on the pinned path, how many lost the mlock or the pin).
inline void print_receipt(const char* who) {
    if (!enabled()) return;
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.m);
    std::printf("%s: host anon pin %zu buffers %.1f MiB (unpinned %zu)\n", who, r.live.size(), static_cast<double>(r.bytes) / 1048576.0, r.unpinned);
}

}  // namespace strix::host_pin
