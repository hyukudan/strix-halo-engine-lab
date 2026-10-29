// Opt-in request-local trace. All timestamps use this one steady-clock epoch.
#pragma once
#include <chrono>
#include <atomic>
#include <utility>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
namespace strix::qwen_prefill_trace {
struct Event {
    std::string stage, component;
    std::uint64_t id = 0, first = 0, rows = 0, bytes = 0;
    double ms = 0;
};
class Trace {
public:
    using Clock = std::chrono::steady_clock;
    // Hard bound on recorded events; marks past it count as dropped.
    static constexpr std::size_t kMaxEvents = 1u << 20;
    Trace() : epoch_(Clock::now()) {}
    void mark(std::string stage, std::uint64_t id = 0, std::string component = {},
              std::uint64_t first = 0, std::uint64_t rows = 0, std::uint64_t bytes = 0) {
        const auto now = Clock::now();
        std::lock_guard<std::mutex> lock(m_);
        if (events_.size() >= kMaxEvents) { ++dropped_; return; }
        events_.push_back({std::move(stage), std::move(component), id, first, rows, bytes,
                          std::chrono::duration<double, std::milli>(now - epoch_).count()});
    }
    void drop() noexcept { ++dropped_; }
    std::uint64_t dropped() const noexcept { return dropped_.load(); }
    std::vector<Event> snapshot() const { std::lock_guard<std::mutex> lock(m_); return events_; }
private:
    std::atomic<std::uint64_t> dropped_{0};
    Clock::time_point epoch_;
    mutable std::mutex m_;
    std::vector<Event> events_;
};
} // namespace strix::qwen_prefill_trace
