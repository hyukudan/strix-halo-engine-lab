#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strix::startup {
// Initialized before main; all translation units share this monotonic origin.
inline const auto process_start = std::chrono::steady_clock::now();
inline std::mutex output_mutex;
inline std::atomic<bool> tile_seen{false}, token_seen{false};
inline bool enabled() noexcept {
    const char* value = std::getenv("STRIX_STARTUP_MARKERS");
    return value && std::strcmp(value, "1") == 0;
}
inline void emit(const char* phase, const char* event, unsigned long long chunk = 0) noexcept {
    if (!enabled()) return;
    const std::lock_guard lock(output_mutex);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - process_start).count();
    std::fprintf(stderr, "strix-startup-marker phase=%s event=%s monotonic_ms=%llu.%03llu chunk=%llu\n",
        phase, event, static_cast<unsigned long long>(us / 1000),
        static_cast<unsigned long long>(us % 1000), chunk);
}
class Phase {
public:
    explicit Phase(const char* phase, unsigned long long chunk = 0) noexcept
        : phase_(enabled() ? phase : nullptr), chunk_(chunk) { if (phase_) emit(phase_, "begin", chunk_); }
    ~Phase() { end(); }
    void end() noexcept { if (phase_) { emit(phase_, "end", chunk_); phase_ = nullptr; } }
    Phase(const Phase&) = delete;
    Phase& operator=(const Phase&) = delete;
private:
    const char* phase_;
    unsigned long long chunk_;
};
inline void first_prefill_tile() noexcept {
    if (enabled() && !tile_seen.exchange(true)) emit("first_prefill_tile", "submit");
}
inline void first_token() noexcept {
    if (enabled() && !token_seen.exchange(true)) emit("first_token", "ready");
}
} // namespace strix::startup
