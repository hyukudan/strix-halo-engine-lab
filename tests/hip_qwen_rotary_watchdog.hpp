#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace strix::qwen_rotary_diagnostic {
// Independent of HIP: a stuck runtime call must not prevent the timeout exit.
class SiteWatchdog {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::string site_;
    std::chrono::steady_clock::time_point deadline_;
    unsigned generation_ = 0;
    bool stop_ = false;
    std::thread worker_{[this] {
        std::unique_lock lock(mutex_);
        while (!stop_) {
            if (site_.empty()) {
                changed_.wait(lock, [this] { return stop_ || !site_.empty(); });
                continue;
            }
            const auto generation = generation_;
            if (!changed_.wait_until(lock, deadline_, [this, generation] {
                    return stop_ || generation_ != generation;
                })) {
                std::fprintf(stderr, "TIMEOUT site=%s\n", site_.c_str());
                std::fflush(stderr);
                // Do not run destructors that might wait on the stalled runtime.
                std::_Exit(124);
            }
        }
    }};
public:
    ~SiteWatchdog() {
        { const std::lock_guard lock(mutex_); stop_ = true; }
        changed_.notify_one();
        worker_.join();
    }
    void begin(const char* site, std::chrono::milliseconds timeout = std::chrono::seconds(120)) {
        {
            const std::lock_guard lock(mutex_);
            site_ = site;
            deadline_ = std::chrono::steady_clock::now() + timeout;
            ++generation_;
        }
        changed_.notify_one();
    }
};
inline SiteWatchdog& site_watchdog() { static SiteWatchdog watchdog; return watchdog; }
} // namespace strix::qwen_rotary_diagnostic
