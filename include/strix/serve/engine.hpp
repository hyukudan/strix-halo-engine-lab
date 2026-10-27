#pragma once

#include <array>
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::serve {

class ConstraintProgram;
struct ThinkingPolicy {
    std::uint32_t max_thinking_tokens = 0, answer_room_tokens = 0, reasoning_limit = 0;
    std::vector<std::uint32_t> closure_ids;
};
struct LogitTransform {
    double presence = 0, frequency = 0, repetition = 1;
    std::vector<std::pair<std::uint32_t, double>> bias;
    bool active() const { return presence != 0 || frequency != 0 || repetition != 1 || !bias.empty(); }
};
struct TokenScore {
    std::uint32_t id = 0;
    std::optional<double> logprob;
    std::vector<std::pair<std::uint32_t, double>> top;
    bool forced_control = false;
};
// Serial host commits publish scores before token IDs; HTTP readers copy under this lock.
struct SelectionScores {
    mutable std::mutex mutex;
    std::vector<TokenScore> tokens;
};
struct ConstraintSpec {
    LogitTransform transform;
    bool logprobs = false;
    std::uint32_t top_logprobs = 0;
    std::vector<std::uint32_t> prompt_ids;
    std::shared_ptr<SelectionScores> scores = std::make_shared<SelectionScores>();
    // Null for a thinking-only policy: ordinary greedy selection in the answer.
    std::shared_ptr<const ConstraintProgram> program;
    std::vector<std::uint32_t> eos_ids;
    std::optional<std::uint32_t> end_thinking_id;
    bool starts_in_reasoning = false;
    std::optional<ThinkingPolicy> thinking;
    std::string identity; // "json-object" or "strix-json-schema-v1"
};

// Reported by the executing engine, never inferred from decoded output.
struct ConstraintVerdict {
    std::string status = "in_progress";
    bool accepting = false;
    std::string phase = "answer";
    std::string reason;
    std::string execution;
    std::string closure = "natural";
    std::uint32_t reasoning_tokens = 0;
    std::vector<std::uint32_t> forced_ids{};
    std::string execution_fallback{};
    // Host selection cost (wall time, every selected row including rejected window rows): mask build, select
    // (transforms, argmax, scores) and, on MTP windows, the whole row (readback + mask + select).
    std::uint64_t profile_rows = 0, profile_window_rows = 0, profile_known_rows = 0;
    double profile_mask_ms = 0, profile_select_ms = 0, profile_window_row_ms = 0;
};

// Armed instrumentation: one process-wide word, unarmed until a GET /metrics or GET /api/status?detail=1 arms it
// (ApiService). The arming expires: each arming request carries its service's expiry (ApiConfig::observe_expiry_seconds:
// default 60 s, STRIX_SERVE_OBSERVE_EXPIRY_S, 0 = never) and the armed period ends at the latest of its requests' times
// plus their expiries (never, once a request without an expiry armed it), when the process-wide expiry thread disarms;
// the next arming request arms again. Unarmed, every instrumentation path is this one relaxed load: the engine threads
// only read the word (none of them reads a clock or takes a lock for the expiry). What was observed stays across periods.
// The word counts the transitions: odd = armed. Arming (even to odd) and the expiry (odd to even) each add one under the
// arming lock, so it is written once per transition (later scrapes only read it: the line the engine threads read stays
// shared; the time of the last arming request lives elsewhere). A request is observed when it was submitted in an armed
// period and the word still holds that period when it finishes: an expiry (or an expiry and a new arming) in between
// changes it, and since every reader reads this one location, a thread that saw the instrumentation unarmed reads the
// changed word afterwards (no ordering between two variables is needed).
inline std::atomic<std::uint64_t> serve_observe_period_{0};
[[nodiscard]] inline std::uint64_t observe_period() noexcept { return serve_observe_period_.load(std::memory_order_relaxed); }
[[nodiscard]] inline bool observe_armed() noexcept { return (observe_period() & 1U) != 0U; }
// An arming request (HTTP threads): arms when unarmed; the period lasts at least `expiry_seconds` more (0: it does not
// expire). True: this call started a period.
bool observe_arm(double expiry_seconds);
// Permanent arming for services with process-lifetime instrumentation.
inline void observe_arm() { (void)observe_arm(0.0); }
// Ends the armed period now (tests; a service that wants the instrumentation off at once).
void observe_disarm();

struct SamplingParameters {
    double temperature = 0.0; // 0 = greedy (exact)
    double top_p = 1.0;
    std::uint32_t top_k = 0;
    double min_p = 0.0;
    std::uint64_t seed = 0;
    bool seeded = false;
};

// A coalesced host snapshot: computed prefill and restored cache counts are separate.
enum class GenerationPhase { queued, prefill, decode };
struct GenerationProgress {
    GenerationPhase phase = GenerationPhase::queued;
    std::uint64_t prefilled_tokens = 0, cached_tokens = 0, total_tokens = 0;
    bool operator==(const GenerationProgress&) const = default;
};

// One generation on the engine. on_tokens runs on the engine thread after each completed decode
// step with the newly verified tokens (stop tokens included); returning false stops generation.
struct GenerationJob {
    std::function<void(GenerationProgress)> on_progress; // optional, host-only completed prefill units
    std::shared_ptr<const ConstraintSpec> constraint; // null: existing route; otherwise JSON and/or thinking selection
    std::vector<std::uint32_t> prompt;
    // chat requests: the prompt's first tokens that render the system prompt and the tools (a prefix every turn of the
    // conversation, and often other conversations, repeat: a prefix cache may snapshot there); 0: none known
    std::uint32_t stable_prefix = 0;
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
    // Set by GenerationQueue::submit: the request was submitted after the instrumentation armed, so every armed
    // measurement of it (first token, drafts) is complete. Engines measure only these requests. When the armed period
    // expires before the request finishes, the queue reports it as not observed (GenerationTicket::observed), and an
    // engine reports its armed-only measurements only when observation_complete() still holds at its finish (else they
    // stopped at the expiry and are partial).
    bool observed = false;
    std::uint64_t observed_period = 0;  // set by submit: observe_period() then
    [[nodiscard]] bool observation_complete() const noexcept { return observed && observe_period() == observed_period; }
};

struct GenerationStats {
    std::optional<ConstraintVerdict> constraint;
    std::uint64_t prompt_tokens = 0, cached_tokens = 0, generated_tokens = 0;
    std::uint64_t cycles = 0, proposed = 0, accepted = 0;
    // Context n-gram extension rows proposed / accepted (0 when it did not run or never opened).
    std::uint64_t ngram_rows = 0, ngram_accepted = 0;
    std::uint64_t tool_rows = 0, tool_accepted = 0; // strict tool-call frame rows (batched decoding, opt-in)
    std::uint64_t source_shadow = 0, source_scored = 0, source_right = 0, frame_cycles = 0, frame_shadow = 0; // diagnostics
    double queue_ms = 0.0, prefill_ms = 0.0, decode_ms = 0.0;
    // first_token_ms, set by engines for observed requests (GenerationJob::observed): submit to the first published
    // token; engines on GenerationQueue::take/finish measure it from TakenJob::submitted (the queue adds nothing),
    // engines on GenerationQueue::run from their generate() entry (run() adds queue_ms). Absent: not reported.
    // imported_tokens, set by engines for every request: prompt tokens whose state was imported from another device;
    // computed prefill = prompt - cached - imported (the status's prefill rate uses it for every request; the field
    // and the C8 totals are reported for observed requests only).
    std::optional<double> first_token_ms;
    std::uint64_t imported_tokens = 0;
    // "stop" (a stop token), "length" (max_tokens), "context" (context full), "cancelled".
    std::string finish = "length";
    std::string sampling = "greedy";
    // Engine receipts added to the response's "strix" object: (member name, JSON value text). Empty: none.
    std::vector<std::pair<std::string, std::string>> engine_extra;
};

// A request-local failure preserves its terminal receipt without poisoning the engine.
class GenerationRequestError : public std::runtime_error {
public:
    GenerationRequestError(GenerationStats result, const std::string& message)
        : std::runtime_error(message), stats(std::move(result)) {}
    GenerationStats stats;
};

class GenerationEngine {
public:
    virtual ~GenerationEngine() = default;
    virtual GenerationStats generate(GenerationJob& job) = 0;
    [[nodiscard]] virtual std::uint64_t context_tokens() const = 0;
    // The sampling modes this engine executes; others are rejected or downgraded by the API.
    [[nodiscard]] virtual bool supports_sampling() const { return false; }
    [[nodiscard]] virtual bool supports_constraints() const { return false; }
    [[nodiscard]] virtual bool supports_thinking_budget() const { return false; }
    [[nodiscard]] virtual std::vector<std::string> constraint_execution_routes() const { return {}; }
};

// A request in flight: the engine thread publishes tokens into it and the HTTP thread drains them.
class GenerationTicket {
public:
    // HTTP side.
    void cancel() noexcept { cancelled_.store(true); notify(); }
    [[nodiscard]] bool cancelled() const noexcept { return cancelled_.load(); }
    // Waits for new tokens or completion (or the timeout); returns the new tokens.
    // wake_on_cancel false: a cancellation does not end the wait (a caller that cancelled waits for the finish)
    std::vector<std::uint32_t> wait(std::chrono::milliseconds timeout, bool& done, bool wake_on_cancel = true);
    [[nodiscard]] GenerationProgress progress() const;
    [[nodiscard]] GenerationStats stats() const;
    [[nodiscard]] std::string error() const;
    // Engine side. Progress replaces the snapshot without allocating or waking the HTTP thread.
    void report_progress(GenerationProgress progress);
    bool publish(std::span<const std::uint32_t> tokens);
    void finish(GenerationStats stats, std::string error = {});
    // Waiting jobs ahead; running jobs do not contribute. The immutable submission
    // snapshot is safe to expose even if the engine takes the job before headers.
    [[nodiscard]] std::uint64_t submitted_queue_position() const noexcept { return submitted_queue_position_; }
    [[nodiscard]] std::uint64_t queue_position() const noexcept { return queue_position_.load(); }
    // The job was submitted after the instrumentation armed (GenerationJob::observed; fixed before the job is queued) and
    // its armed period did not expire before it finished (the queue clears it then, before finishing the ticket).
    [[nodiscard]] bool observed() const noexcept { return observed_; }
    std::atomic<std::uint64_t> queue_position_{0};
private:
    void notify();
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::uint32_t> pending_;
    GenerationProgress progress_;
    GenerationStats stats_;
    std::string error_;
    bool done_ = false;
    std::atomic<bool> cancelled_{false};
    bool observed_ = false;
    std::uint64_t submitted_queue_position_ = 0;
    std::uint64_t period_ = 0;  // observe_period() at submit
    friend class GenerationQueue;
};

// A job taken by a batching engine (GenerationQueue::take). on_tokens already publishes into the
// ticket; the engine reports the end with GenerationQueue::finish.
struct TakenJob {
    GenerationJob job;
    std::shared_ptr<GenerationTicket> ticket;
    std::chrono::steady_clock::time_point submitted, started;
};

// Fixed-bucket histogram (cumulative only when rendered): counts[i] = observations <= bounds[i] and > bounds[i - 1];
// counts[N] = above the last bound.
template <std::size_t N>
struct ObserveHistogram {
    std::array<std::uint64_t, N + 1> counts{};
    double sum = 0.0;
    std::uint64_t count = 0;
    // the bucket of a value (N: above the last bound), N + 1 when it is not an observation (NaN, negative, absurd)
    static std::size_t bucket(double value, const std::array<double, N>& bounds) noexcept {
        if (!(value >= 0.0) || value > 1e12) return N + 1;
        std::size_t i = 0;
        while (i < N && value > bounds[i]) ++i;
        return i;
    }
    void add(std::size_t i, double value) noexcept {
        if (i > N) return;
        ++counts[i];
        sum += value;
        ++count;
    }
    void observe(double value, const std::array<double, N>& bounds) noexcept { add(bucket(value, bounds), value); }
};
inline constexpr std::array<double, 10> ttft_seconds_bounds{0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0};
inline constexpr std::array<double, 12> decode_tok_s_bounds{1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0};

// What the queue observed: observed requests (submitted after arming) that finished without an error; the prompt
// accounting only of those that produced a token (their prompt was read). It sums every armed period (an expiry keeps it).
struct ObserveSnapshot {
    std::uint64_t requests = 0, prompt_tokens = 0, cached_tokens = 0, imported_tokens = 0, computed_tokens = 0;
    ObserveHistogram<ttft_seconds_bounds.size()> ttft_seconds;            // from first_token_ms, when reported
    ObserveHistogram<decode_tok_s_bounds.size()> decode_tokens_per_second; // generated / decode time, >= 2 tokens
};

// One observed request's contribution, computed before the queue lock (which then only adds it). No member
// initializers: an unobserved request never initializes one; GenerationQueue::observe_delta() sets every member.
struct ObserveDelta {
    std::uint64_t requests, prompt_tokens, cached_tokens, imported_tokens, computed_tokens;
    std::size_t ttft_bucket, decode_bucket;  // N + 1: none
    double ttft_seconds, decode_tokens_per_second;
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
    [[nodiscard]] bool has_queued_work() const;
    // The end of a taken job: fills queue_ms, updates the totals and finishes its ticket.
    // Engines may call it inside their own timed regions (a batching runner that times a prompt unit and completes a
    // request in it). The observation it adds reads and writes only the queue's own counters, never an engine's
    // scheduling state; its cost is fixed: one branch unarmed, and armed a contribution computed from the job's stats
    // before the lock plus a few additions under it (no allocation, no syscall). A runner's timed region that encloses
    // the call therefore sees at most that constant, on top of the totals update it already contained.
    void finish(TakenJob& job, GenerationStats stats, std::string error = {},
        const std::function<void(const GenerationStats&)>& observe = {});
    [[nodiscard]] bool stopping() const;
    void shutdown();
    [[nodiscard]] QueueSnapshot snapshot() const;
    [[nodiscard]] ObserveSnapshot observed() const;
private:
    static bool still_observed(GenerationTicket& ticket) noexcept;
    static ObserveDelta observe_delta(const GenerationStats& stats, const std::string& error) noexcept;
    void observe_locked(const ObserveDelta& d) noexcept;
    struct Entry { GenerationJob job; std::shared_ptr<GenerationTicket> ticket; std::chrono::steady_clock::time_point submitted; };
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Entry> entries_;
    std::size_t max_queued_;
    bool stopping_ = false;
    QueueSnapshot totals_;
    ObserveSnapshot observed_;
};

}  // namespace strix::serve
