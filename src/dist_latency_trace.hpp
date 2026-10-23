// Optional diagnostic trace. CPU observations use the same CLOCK_MONOTONIC as tracefs mono.
// Each thread buffers records and writes only at its exit; no file I/O in an exchange.
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace strix::dist::latency {
inline const char* path() {
    static const char* value = std::getenv("STRIX_DIST_LATENCY_TRACE");
    return value;
}
inline bool enabled() { return path() && *path(); }
struct Context { std::uint64_t sequence = UINT64_MAX; };
inline Context& context() { static thread_local Context value; return value; }
struct Record {
    std::uint64_t ns, sequence, value;
    int cpu;
    const char* phase;
};
struct Buffer {
    std::array<Record, 65536> records{};
    std::size_t used = 0, dropped = 0;
    ~Buffer() {
        if (!used) return;
        std::array<char, 4096> name{};
        const int n = std::snprintf(name.data(), name.size(), "%s.t%ld.csv", path(), ::syscall(SYS_gettid));
        if (n <= 0 || static_cast<std::size_t>(n) >= name.size()) return;
        if (auto* out = std::fopen(name.data(), "w")) {
            std::fprintf(out, "# clock=CLOCK_MONOTONIC tid=%ld dropped=%zu\nns,sequence,phase,cpu,value\n", ::syscall(SYS_gettid), dropped);
            for (std::size_t i = 0; i < used; ++i) {
                const auto& r = records[i];
                std::fprintf(out, "%llu,%llu,%s,%d,%llu\n", static_cast<unsigned long long>(r.ns),
                             static_cast<unsigned long long>(r.sequence), r.phase, r.cpu, static_cast<unsigned long long>(r.value));
            }
            std::fclose(out);
        } else {
            std::perror("dist latency trace");
        }
    }
};
inline void mark(const char* phase, std::uint64_t value = 0) {
    if (!enabled()) return;
    static thread_local Buffer buffer;
    if (buffer.used == buffer.records.size()) { ++buffer.dropped; return; }
    timespec ts{};
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    const auto ns = static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(ts.tv_nsec);
    buffer.records[buffer.used++] = Record{ns, context().sequence, value, ::sched_getcpu(), phase};
}
} // namespace strix::dist::latency
