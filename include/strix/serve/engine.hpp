#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strix::serve {

struct SamplingParameters {
    double temperature = 0.0; // 0 = greedy (exact)
    double top_p = 1.0;
    std::uint32_t top_k = 0;
    double min_p = 0.0;
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
    // Request field "ngram_draft": true/false forces the context n-gram draft extension on/off for this
    // request; unset = the engine's default (auto: on where admissible, its learned gate decides).
    std::optional<bool> ngram_draft;
    // Request field "draft_depth_max" (batched decoding, measurement): at most this many predictor drafts per
    // cycle for this request (0/absent = the engine's setting). Drafts only: never the request's output.
    std::uint32_t draft_depth_max = 0;
    // Request field "tool_draft" (batched decoding): strict tool-call frame proposals on/off for this request;
    // unset = the engine's setting (STRIX_BATCH_TOOL_DRAFT, off by default). Drafts only.
    std::optional<bool> tool_draft;
    // Request field "ngram_peers" (batched decoding): cross-request lookup on/off for this request's n-gram
    // extension; unset = the engine's setting (STRIX_BATCH_NGRAM_PEERS, off by default). Drafts only.
    std::optional<bool> ngram_peers;
    std::function<bool(std::span<const std::uint32_t>)> on_tokens;
};

struct GenerationStats {
    std::uint64_t prompt_tokens = 0, cached_tokens = 0, generated_tokens = 0;
    std::uint64_t cycles = 0, proposed = 0, accepted = 0;
    // Context n-gram extension rows proposed / accepted (0 when it did not run or never opened).
    std::uint64_t ngram_rows = 0, ngram_accepted = 0;
    std::uint64_t tool_rows = 0, tool_accepted = 0; // strict tool-call frame rows (batched decoding, opt-in)
    std::uint64_t source_shadow = 0, source_scored = 0, source_right = 0, frame_cycles = 0, frame_shadow = 0; // diagnostics
    double queue_ms = 0.0, prefill_ms = 0.0, decode_ms = 0.0;
    // "stop" (a stop token), "length" (max_tokens), "context" (context full), "cancelled".
    std::string finish = "length";
    std::string sampling = "greedy";
    // Engine receipts added to the response's "strix" object: (member name, JSON value text). Empty: none.
    std::vector<std::pair<std::string, std::string>> engine_extra;
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

// A job taken by a batching engine (GenerationQueue::take). on_tokens already publishes into the
// ticket; the engine reports the end with GenerationQueue::finish.
struct TakenJob {
    GenerationJob job;
    std::shared_ptr<GenerationTicket> ticket;
    std::chrono::steady_clock::time_point submitted, started;
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
    // Batching engines, which decode several jobs at once and take new ones between steps instead of
    // run(): the oldest job that is not cancelled, waiting for one when `wait` (nullopt when none is
    // queued, or after shutdown). Jobs cancelled while queued are finished here.
    std::optional<TakenJob> take(bool wait);
    // The end of a taken job: fills queue_ms, updates the totals and finishes its ticket.
    void finish(TakenJob& job, GenerationStats stats, std::string error = {},
        const std::function<void(const GenerationStats&)>& observe = {});
    [[nodiscard]] bool stopping() const;
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
