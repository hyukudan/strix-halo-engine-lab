#pragma once
#include "byte_budget.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace strix::core {
// The one host-side pool. Engines do not own threads for request preprocessing or background work; they
// submit jobs here.
//   critical   : short latency-bound jobs (tokenizer, row folds, masks). At most two threads, which keep
//                polling for a short window after a job so back-to-back jobs do not pay a wake-up.
//   background : byte-heavy jobs (snapshot packing, checkpoint CRC, cache hashing). Sleeping threads; each
//                job draws the bytes it reads from budget(), which is capped while any session decodes.
enum class HostPriority { critical, background };

struct HostLaneConfig {
    bool enabled = true;                 // false: every job runs inline in submit()
    std::vector<int> cores;              // cores the lane may pin; empty: no pinning
    unsigned critical_threads = 1;       // clamped to [0, 2]
    unsigned background_threads = 2;
    double byte_cap_gbytes_ps = 10.0;    // background ceiling while any decode is active; <= 0: never capped
    std::chrono::microseconds critical_spin{200};
};

struct HostLaneStats {
    std::uint64_t critical_jobs = 0, background_jobs = 0, inline_jobs = 0;
    std::uint64_t critical_busy_ns = 0, background_busy_ns = 0;
    ByteBudget::Stats budget;
    double busy_ms() const { return static_cast<double>(critical_busy_ns + background_busy_ns) / 1e6; }
};

// "2-15", "0,2,4-6". Throws std::invalid_argument on anything else.
inline std::vector<int> parse_core_list(const std::string& text) {
    std::vector<int> cores;
    std::size_t at = 0;
    const auto number = [&]() {
        if (at >= text.size() || text[at] < '0' || text[at] > '9') throw std::invalid_argument("core list: expected a number");
        long value = 0;
        while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
            value = value * 10 + (text[at++] - '0');
            if (value > 4095) throw std::invalid_argument("core list: core index too large");
        }
        return static_cast<int>(value);
    };
    while (at < text.size()) {
        const int first = number();
        int last = first;
        if (at < text.size() && text[at] == '-') {
            ++at;
            last = number();
            if (last < first) throw std::invalid_argument("core list: descending range");
        }
        for (int core = first; core <= last; ++core) cores.push_back(core);
        if (at == text.size()) break;
        if (text[at] != ',' || at + 1 == text.size()) throw std::invalid_argument("core list: expected ','");
        ++at;
    }
    return cores;
}

// Best effort; false when the platform has no affinity call or the kernel refuses the set.
inline bool pin_current_thread(const std::vector<int>& cores) {
#if defined(__linux__)
    if (cores.empty()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int core : cores)
        if (core >= 0 && core < CPU_SETSIZE) CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cores;
    return false;
#endif
}

class HostLane {
public:
    // budget: null gives the lane its own budget from config.byte_cap_gbytes_ps; the process-wide lane shares
    // the process-wide budget.
    explicit HostLane(HostLaneConfig config = {}, ByteBudget* budget = nullptr)
        : config_(std::move(config)), own_budget_(budget ? 0.0 : config_.byte_cap_gbytes_ps * 1e9),
          budget_(budget ? budget : &own_budget_) {
        if (config_.critical_threads > 2) config_.critical_threads = 2;
        if (!config_.enabled) config_.critical_threads = config_.background_threads = 0;
        const std::size_t reserved = std::min<std::size_t>(config_.critical_threads, config_.cores.size());
        std::vector<int> rest(config_.cores.begin() + static_cast<std::ptrdiff_t>(reserved), config_.cores.end());
        if (rest.empty()) rest = config_.cores;
        for (unsigned i = 0; i < config_.critical_threads; ++i) {
            std::vector<int> own = i < config_.cores.size() ? std::vector<int>{config_.cores[i]} : config_.cores;
            threads_.emplace_back([this, own = std::move(own)] { pin_current_thread(own); critical_loop(); });
        }
        for (unsigned i = 0; i < config_.background_threads; ++i)
            threads_.emplace_back([this, rest] { pin_current_thread(rest); background_loop(); });
    }
    HostLane(const HostLane&) = delete;
    HostLane& operator=(const HostLane&) = delete;
    // Queued jobs are completed before the threads leave, so no future is abandoned.
    ~HostLane() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        critical_wake_.notify_all();
        background_wake_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    template<class F>
    auto submit(HostPriority priority, F&& job) -> std::future<std::invoke_result_t<std::decay_t<F>&>> {
        using Result = std::invoke_result_t<std::decay_t<F>&>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(job));
        auto result = task->get_future();
        const bool critical = priority == HostPriority::critical;
        if ((critical ? config_.critical_threads : config_.background_threads) == 0) {
            inline_jobs_.fetch_add(1, std::memory_order_relaxed);
            (*task)();
            return result;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) throw std::logic_error("host lane: submit after shutdown");
            (critical ? critical_ : background_).emplace_back([task] { (*task)(); });
            if (critical) critical_pending_.fetch_add(1, std::memory_order_release);
        }
        (critical ? critical_wake_ : background_wake_).notify_one();
        return result;
    }

    ByteBudget& budget() { return *budget_; }
    const HostLaneConfig& config() const { return config_; }
    HostLaneStats stats() const {
        HostLaneStats out;
        out.critical_jobs = critical_jobs_.load(std::memory_order_relaxed);
        out.background_jobs = background_jobs_.load(std::memory_order_relaxed);
        out.inline_jobs = inline_jobs_.load(std::memory_order_relaxed);
        out.critical_busy_ns = critical_busy_ns_.load(std::memory_order_relaxed);
        out.background_busy_ns = background_busy_ns_.load(std::memory_order_relaxed);
        out.budget = budget_->stats();
        return out;
    }
    // Receipt fields: busy time, bytes granted to background work, and how often the cap made a job wait.
    std::string receipt_json() const {
        const auto s = stats();
        return "{\"host_lane_busy_ms\":" + std::to_string(s.busy_ms()) +
               ",\"host_lane_jobs\":" + std::to_string(s.critical_jobs + s.background_jobs + s.inline_jobs) +
               ",\"host_lane_bytes_granted\":" + std::to_string(s.budget.bytes_granted) +
               ",\"host_lane_cap_hits\":" + std::to_string(s.budget.cap_hits) +
               ",\"host_lane_throttled_ms\":" + std::to_string(static_cast<double>(s.budget.throttled_ns) / 1e6) + "}";
    }

private:
    using Job = std::function<void()>;
    using Clock = std::chrono::steady_clock;

    static void run(Job& job, std::atomic<std::uint64_t>& jobs, std::atomic<std::uint64_t>& busy_ns) {
        jobs.fetch_add(1, std::memory_order_relaxed); // before the future becomes ready
        const auto begin = Clock::now();
        job(); // a packaged task: exceptions travel through the future
        busy_ns.fetch_add(static_cast<std::uint64_t>(
                              std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count()),
                          std::memory_order_relaxed);
    }
    void critical_loop() {
        auto spin_until = Clock::now();
        for (;;) {
            Job job;
            if (critical_pending_.load(std::memory_order_acquire) != 0) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!critical_.empty()) {
                    job = std::move(critical_.front());
                    critical_.pop_front();
                    critical_pending_.fetch_sub(1, std::memory_order_release);
                }
            }
            if (job) {
                run(job, critical_jobs_, critical_busy_ns_);
                spin_until = Clock::now() + config_.critical_spin;
                continue;
            }
            if (Clock::now() < spin_until) {
                std::this_thread::yield();
                continue;
            }
            std::unique_lock<std::mutex> lock(mutex_);
            critical_wake_.wait(lock, [&] { return stop_ || !critical_.empty(); });
            if (critical_.empty()) return; // stop requested and nothing queued
        }
    }
    void background_loop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                background_wake_.wait(lock, [&] { return stop_ || !background_.empty(); });
                if (background_.empty()) return;
                job = std::move(background_.front());
                background_.pop_front();
            }
            run(job, background_jobs_, background_busy_ns_);
        }
    }

    HostLaneConfig config_;
    ByteBudget own_budget_;
    ByteBudget* budget_;
    std::mutex mutex_;
    std::condition_variable critical_wake_, background_wake_;
    std::deque<Job> critical_, background_;
    std::atomic<std::uint64_t> critical_pending_{0};
    bool stop_ = false;
    std::atomic<std::uint64_t> critical_jobs_{0}, background_jobs_{0}, inline_jobs_{0};
    std::atomic<std::uint64_t> critical_busy_ns_{0}, background_busy_ns_{0};
    std::vector<std::thread> threads_;
};

// Configuration of the process-wide lane from the environment (the off switch and the declared cores):
//   STRIX_HOST_LANE=0                  run every job inline (previous behaviour)
//   STRIX_HOST_LANE_CORES=2-15         cores the lane may pin
//   STRIX_HOST_LANE_BACKGROUND=N       background threads (default 2)
//   STRIX_HOST_BYTE_CAP_GBPS=10        background ceiling during decode; 0 = never capped
inline HostLaneConfig host_lane_config_from_environment() {
    HostLaneConfig config;
    if (const char* value = host_environment("STRIX_HOST_LANE")) config.enabled = std::string(value) != "0";
    if (const char* value = host_environment("STRIX_HOST_LANE_CORES")) config.cores = parse_core_list(value);
    if (const char* value = host_environment("STRIX_HOST_LANE_BACKGROUND")) {
        const long threads = std::strtol(value, nullptr, 10);
        if (threads < 0 || threads > 64) throw std::invalid_argument("STRIX_HOST_LANE_BACKGROUND out of range");
        config.background_threads = static_cast<unsigned>(threads);
    }
    config.byte_cap_gbytes_ps = byte_cap_gbytes_ps_from_environment(config.byte_cap_gbytes_ps);
    return config;
}

// The process-wide lane. Created on first use; every engine and the serving layer share it.
inline HostLane& shared_host_lane() {
    static HostLane lane(host_lane_config_from_environment(), &shared_byte_budget());
    return lane;
}
} // namespace strix::core
