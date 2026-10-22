#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#if defined(__linux__)
#include <time.h>
#endif

namespace strix {
// STRIX_CYCLE_TIMING=1: host time between consecutive named marks, totals printed at exit.
// STRIX_CYCLE_TIMING_SLOW=<ms>: also prints every single phase longer than <ms> as it happens, with its
// steady-clock and boot-clock end times in ns ("slow phase t=<ns> bt=<ns> <name> <ms>"), to place host stalls on a kernel trace.
struct HostPhaseTiming {
    double slow_ms = [] { const char* v = std::getenv("STRIX_CYCLE_TIMING_SLOW"); return v ? std::atof(v) : -1.0; }();
    bool enabled = std::getenv("STRIX_CYCLE_TIMING") != nullptr || slow_ms >= 0.0;
    std::chrono::steady_clock::time_point last{};
    std::map<std::string, std::pair<double, unsigned long long>> totals;
    ~HostPhaseTiming() {
        if (!enabled) return;
        for (const auto& [name, t] : totals)
            std::fprintf(stderr, "cycle host timing %-40s %8llu x %.4f ms\n", name.c_str(), t.second, t.first / static_cast<double>(t.second));
    }
};
inline HostPhaseTiming& host_phase_timing() { static HostPhaseTiming t; return t; }
// Records the time since the previous mark under `name` (a reset mark starts a new chain).
inline void host_phase_mark(const char* name, bool reset = false) {
    auto& t = host_phase_timing();
    if (!t.enabled) return;
    const auto now = std::chrono::steady_clock::now();
    if (!reset && t.last.time_since_epoch().count()) {
        auto& e = t.totals[name];
        const double ms = std::chrono::duration<double, std::milli>(now - t.last).count();
        e.first += ms; ++e.second;
        if (t.slow_ms >= 0.0 && ms > t.slow_ms)
        {
            long long boot = 0;
#if defined(__linux__)
            timespec ts{};
            if (clock_gettime(CLOCK_BOOTTIME, &ts) == 0) boot = static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
#endif
            std::fprintf(stderr, "slow phase t=%lld bt=%lld %s %.3f\n", static_cast<long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count()), boot, name, ms);
        }
    }
    t.last = now;
}
}  // namespace strix
