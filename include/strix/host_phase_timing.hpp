#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace strix {
// STRIX_CYCLE_TIMING=1: host time between consecutive named marks, totals printed at exit.
struct HostPhaseTiming {
    bool enabled = std::getenv("STRIX_CYCLE_TIMING") != nullptr;
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
        e.first += std::chrono::duration<double, std::milli>(now - t.last).count(); ++e.second;
    }
    t.last = now;
}
}  // namespace strix
