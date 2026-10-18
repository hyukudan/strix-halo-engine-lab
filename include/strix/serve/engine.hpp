#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace strix::serve {

struct SamplingParameters {
    double temperature = 0.0; // 0 = greedy (exact)
    double top_p = 1.0;
    std::uint32_t top_k = 0;
    std::uint64_t seed = 0;
    bool seeded = false;
};

// One generation on the engine. on_tokens runs on the engine thread after each completed decode
// step with the newly verified tokens (stop tokens included); returning false stops generation.
struct GenerationJob {
    std::vector<std::uint32_t> prompt;
    std::uint32_t max_tokens = 256;
    std::vector<std::uint32_t> stop_tokens;
    SamplingParameters sampling;
    std::function<bool(std::span<const std::uint32_t>)> on_tokens;
};

struct GenerationStats {
    std::uint64_t prompt_tokens = 0, cached_tokens = 0, generated_tokens = 0;
    std::uint64_t cycles = 0, proposed = 0, accepted = 0;
    double queue_ms = 0.0, prefill_ms = 0.0, decode_ms = 0.0;
    // "stop" (a stop token), "length" (max_tokens), "context" (context full), "cancelled".
    std::string finish = "length";
    std::string sampling = "greedy";
};

class GenerationEngine {
public:
    virtual ~GenerationEngine() = default;
    virtual GenerationStats generate(GenerationJob& job) = 0;
    [[nodiscard]] virtual std::uint64_t context_tokens() const = 0;
    // The sampling modes this engine executes; others are rejected or downgraded by the API.
    [[nodiscard]] virtual bool supports_sampling() const { return false; }
};

// A request in flight: the engine thread publishes tokens into it and the HTTP thread drains them.
class GenerationTicket {
public:
    // HTTP side.
    void cancel() noexcept { cancelled_.store(true); notify(); }
    [[nodiscard]] bool cancelled() const noexcept { return cancelled_.load(); }
    // Waits for new tokens or completion (or the timeout); returns the new tokens.
    std::vector<std::uint32_t> wait(std::chrono::milliseconds timeout, bool& done);
    [[nodiscard]] GenerationStats stats() const;
    [[nodiscard]] std::string error() const;
    // Engine side.
    bool publish(std::span<const std::uint32_t> tokens);
    void finish(GenerationStats stats, std::string error = {});
    [[nodiscard]] std::uint64_t queue_position() const noexcept { return queue_position_.load(); }
    std::atomic<std::uint64_t> queue_position_{0};
private:
    void notify();
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::uint32_t> pending_;
    GenerationStats stats_;
    std::string error_;
    bool done_ = false;
    std::atomic<bool> cancelled_{false};
};

struct QueueSnapshot {
    std::uint64_t queued = 0, running = 0, completed = 0, failed = 0, cancelled = 0;
    std::uint64_t prompt_tokens = 0, cached_tokens = 0, generated_tokens = 0;
    double busy_seconds = 0.0;
};

// FIFO of generation jobs executed one at a time on the thread that calls run().
class GenerationQueue {
public:
    explicit GenerationQueue(std::size_t max_queued = 64) : max_queued_(max_queued) {}
    // nullptr when the queue is full or shut down.
    std::shared_ptr<GenerationTicket> submit(GenerationJob job);
    // Runs jobs until shutdown(); `observe` (optional) sees every finished job's stats.
    void run(GenerationEngine& engine, const std::function<void(const GenerationStats&)>& observe = {});
    void shutdown();
    [[nodiscard]] QueueSnapshot snapshot() const;
private:
    struct Entry { GenerationJob job; std::shared_ptr<GenerationTicket> ticket; std::chrono::steady_clock::time_point submitted; };
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Entry> entries_;
    std::size_t max_queued_;
    bool stopping_ = false;
    QueueSnapshot totals_;
};

}  // namespace strix::serve
