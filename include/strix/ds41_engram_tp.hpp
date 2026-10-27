#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strix::ds41 {

// Engram output-row TP (only with two ranks and an F8 engram_kv, i.e. STRIX_DS41_F16_AS_F8; steps of <= 8 rows): each
// rank computes half of the Engram projection's output rows (the same per-row kernel, so the same bits) and the halves are
// exchanged before the injection, instead of both ranks reading the whole ~160 MB weight. Default: on for every single-row
// DeviceForward / BatchForward step, decode or prompt (in-process A/B, exact: R1 -0.61 ms/step, -2.1 %; at R6 not
// significant); STRIX_DS41_ENGRAM_TP=1: on for every step of at most 8 rows; =0: off; other values: warning, default.
// set_engram_tp(0 / 1) overrides the environment (the apps' --aba-flags engtp); -1 restores it.
inline std::atomic<int>& engram_tp_override() {
    static std::atomic<int> mode{-1};
    return mode;
}
inline void set_engram_tp(int mode) { engram_tp_override().store(mode, std::memory_order_relaxed); }
inline bool engram_tp(std::uint32_t rows) {
    int mode = engram_tp_override().load(std::memory_order_relaxed);
    if (mode < 0) {
        static const int env = [] {
            const char* v = std::getenv("STRIX_DS41_ENGRAM_TP");
            if (v == nullptr || *v == '\0') return -1;
            if (std::strcmp(v, "0") == 0) return 0;
            if (std::strcmp(v, "1") == 0) return 1;
            std::fprintf(stderr, "STRIX_DS41_ENGRAM_TP=%s ignored (0 or 1): default\n", v);
            return -1;
        }();
        mode = env;
    }
    if (mode == 0) return false;
    return mode == 1 ? rows <= 8U : rows == 1U;
}

}  // namespace strix::ds41
