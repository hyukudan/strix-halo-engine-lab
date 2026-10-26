#pragma once

// Observability of strix-minimax-serve, rank 0's HTTP side: request ids, the recent-request ring, the latency
// histograms and the rendering of GET /api/status (schema strix-serve-status-v1) and GET /metrics (metrics text
// exposition format 0.0.4). This module runs on HTTP threads only and never takes an engine lock.
//
// Armed instrumentation: the serve layer's process-wide flag, set by the first GET /metrics or GET /api/status?detail=1
// and never cleared. GenerationQueue::submit loads it once per request and marks the request observed. Unarmed, the
// engine thread only tests that boolean at completion (in Lead::finish and in GenerationQueue::finish). For an observed
// request the scheduler copies its first-token time into the stats and the queue updates its own accounting under its
// existing mutex, both at completion; the HTTP thread then writes one ring record and the histograms here, under this
// module's own mutex. Lifetime totals are the engine queue's counters (GenerationQueue::snapshot), never counted twice.

#include "strix/serve/engine.hpp"
#include "strix/serve/json.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::minimax::serve {

// An inbound X-Request-Id is kept when it has 8-64 characters of [A-Za-z0-9._-]; anything else is ignored.
[[nodiscard]] bool valid_request_id(std::string_view id) noexcept;

// A query string (a=1&b=2) has key=1 or key=true.
[[nodiscard]] bool query_flag(std::string_view query, std::string_view key) noexcept;

// The process-wide armed flag: the serve layer's own (strix::serve::observe_armed / observe_arm; one-way).
[[nodiscard]] bool observe_armed() noexcept;
void observe_arm() noexcept;
void observe_reset_for_tests() noexcept;

// 16 lowercase hex digits from the system random source: one per process start (the counters' baseline key).
[[nodiscard]] std::string new_instance_id();

// One finished generation request, as the HTTP thread saw it.
struct RequestRecord {
    std::string request_id;     // the accepted inbound X-Request-Id, empty when none
    std::string completion_id;  // chatcmpl-N / cmpl-N
    std::string route;          // chat.completions | completions
    std::int64_t time = 0;      // unix seconds at arrival
    std::string status;         // done | cancelled | error
    std::string stop_reason;    // stop | length | context | cancelled ("" on error)
    std::string error;
    bool stream = false;
    std::uint64_t prompt_tokens = 0, cached_tokens = 0, imported_tokens = 0, output_tokens = 0;
    bool draft = false;  // the engine drafted for this request (cycles/proposed/accepted are meaningful)
    std::uint64_t cycles = 0, proposed = 0, accepted = 0;
    double queue_ms = 0.0, prefill_ms = 0.0, decode_ms = 0.0;
    double first_token_ms = -1.0;  // submit to the first published token; < 0 = not measured
    double max_gap_ms = -1.0;      // the engine's longest gap between two emitted tokens; < 0 = not measured

    [[nodiscard]] const std::string& id() const noexcept { return request_id.empty() ? completion_id : request_id; }
    [[nodiscard]] std::uint64_t computed_tokens() const noexcept;
    [[nodiscard]] double decode_tok_s() const noexcept;  // tokens after the first per second of decode; 0 when unknown
};

// Fixed-bucket histogram (upper bounds ascending, +Inf implicit).
class Histogram {
public:
    explicit Histogram(std::vector<double> bounds);
    void observe(double value) noexcept;
    [[nodiscard]] const std::vector<double>& bounds() const noexcept { return bounds_; }
    [[nodiscard]] std::uint64_t cumulative(std::size_t bucket) const noexcept;  // bucket == bounds().size(): +Inf
    [[nodiscard]] double sum() const noexcept { return sum_; }
    [[nodiscard]] std::uint64_t count() const noexcept { return count_; }

private:
    std::vector<double> bounds_;
    std::vector<std::uint64_t> counts_;  // per bucket, the last one = above every bound
    double sum_ = 0.0;
    std::uint64_t count_ = 0;
};

// HELP and TYPE of a legacy value line of /metrics (its name and value line stay as they were).
struct LegacyMetric {
    std::string name, type, help;
};
// HELP/TYPE of the mm_* lines: the queue's seven and the serve scheduler's eleven (step, prefill, draft, idle and
// stall seconds; steps, step rows, prefill units, draft cycles, drafted and accepted tokens).
[[nodiscard]] const std::vector<LegacyMetric>& legacy_metrics();
// The legacy value lines, each preceded by its HELP and TYPE (a line whose name is not in the table: TYPE untyped).
[[nodiscard]] std::string with_help(std::string_view legacy, const std::vector<LegacyMetric>& table);

// Static facts of the serving process.
struct ObserverInfo {
    std::string engine = "strix-minimax-serve";
    std::string model;
    std::uint64_t context_tokens = 0;
    std::uint64_t slots = 0;
    std::string transport;
    double load_seconds = 0.0;
    std::uint64_t device_used_bytes = 0;
    std::string instance_id;      // new_instance_id() when empty
    bool draft = false;           // a drafter is loaded (capabilities.draft)
    std::vector<LegacyMetric> legacy;  // HELP/TYPE of the legacy lines; empty = legacy_metrics()
    std::int64_t start_time = 0;  // unix seconds; now when 0
};

class Observer {
public:
    explicit Observer(ObserverInfo info, std::size_t ring_size = 50);

    // The end of a request submitted while armed (the caller checks the flag once, at submission): one ring entry and
    // the histograms (TTFT when first_token_ms >= 0; decode rate for completed requests with more than one token).
    void record(RequestRecord r);

    // GET /api/status: queue counters from the engine queue's snapshot; detail adds the latency histograms.
    [[nodiscard]] strix::serve::Json status(const strix::serve::QueueSnapshot& q, bool stopping, bool detail) const;
    // GET /metrics: the legacy lines (default legacy_metric_lines(q)) with HELP/TYPE, then the strix_* families.
    [[nodiscard]] std::string metrics(const strix::serve::QueueSnapshot& q) const;
    [[nodiscard]] std::string metrics(const strix::serve::QueueSnapshot& q, std::string_view legacy) const;

    [[nodiscard]] const ObserverInfo& info() const noexcept { return info_; }

    static const std::vector<double>& ttft_bounds();
    static const std::vector<double>& decode_bounds();

private:
    ObserverInfo info_;
    std::size_t ring_size_;
    std::chrono::steady_clock::time_point started_;
    mutable std::mutex mutex_;
    std::deque<RequestRecord> recent_;
    Histogram ttft_, decode_;
};

// The queue's seven mm_* value lines (the serve app appends its scheduler's lines with the same format; every value
// line is byte-identical to the text /metrics served before HELP/TYPE were added).
[[nodiscard]] std::string legacy_metric_lines(const strix::serve::QueueSnapshot& q);

// The "strix" object of a reply: {"token_ids": [...]}.
[[nodiscard]] strix::serve::Json reply_extension(std::span<const std::uint32_t> token_ids);

}  // namespace strix::minimax::serve
