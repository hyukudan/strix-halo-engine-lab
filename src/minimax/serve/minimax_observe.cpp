#include "minimax/serve/minimax_observe.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <random>
#include <utility>

namespace strix::minimax::serve {
namespace {

namespace sv = strix::serve;


sv::Json number(double v) { return sv::Json(std::isfinite(v) ? v : 0.0); }
sv::Json count(std::uint64_t v) { return sv::Json(v); }

double rate(std::uint64_t tokens, double ms) {
    return ms > 0.0 ? 1000.0 * static_cast<double>(tokens) / ms : 0.0;
}

std::string format_u64(std::uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%" PRIu64, v);
    return b;
}
std::string format_double(double v) {
    if (!std::isfinite(v)) v = 0.0;
    char b[64];
    std::snprintf(b, sizeof b, "%.10g", v);
    return b;
}
std::string format_bound(double v) { return std::isinf(v) ? std::string("+Inf") : format_double(v); }

// Label value escaping of the metrics text exposition format: backslash, double quote and line feed.
std::string label_value(std::string_view v) {
    std::string out;
    out.reserve(v.size());
    for (const char c : v) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

void family(std::string& m, std::string_view name, std::string_view type, std::string_view help) {
    m.append("# HELP ").append(name).append(" ").append(help).append("\n");
    m.append("# TYPE ").append(name).append(" ").append(type).append("\n");
}
void sample(std::string& m, std::string_view name, const std::string& value, std::string_view labels = {}) {
    m.append(name);
    if (!labels.empty()) m.append("{").append(labels).append("}");
    m.append(" ").append(value).append("\n");
}
void scalar(std::string& m, std::string_view name, std::string_view type, std::string_view help, const std::string& value) {
    family(m, name, type, help);
    sample(m, name, value);
}

void histogram_text(std::string& m, std::string_view name, std::string_view help, const Histogram& h) {
    family(m, name, "histogram", help);
    const std::string bucket = std::string(name) + "_bucket";
    for (std::size_t i = 0; i <= h.bounds().size(); ++i) {
        const double le = i < h.bounds().size() ? h.bounds()[i] : INFINITY;
        sample(m, bucket, format_u64(h.cumulative(i)), "le=\"" + format_bound(le) + "\"");
    }
    sample(m, std::string(name) + "_sum", format_double(h.sum()));
    sample(m, std::string(name) + "_count", format_u64(h.count()));
}

sv::Json histogram_json(const Histogram& h) {
    sv::Json le = sv::Json::array(), counts = sv::Json::array();
    for (std::size_t i = 0; i <= h.bounds().size(); ++i) {
        // +Inf is not a JSON number: the last cumulative count has no bound (it equals count).
        if (i < h.bounds().size()) le.push(sv::Json(h.bounds()[i]));
        counts.push(count(h.cumulative(i)));
    }
    sv::Json out = sv::Json::object();
    out.set("le", std::move(le));
    out.set("counts", std::move(counts));
    out.set("sum", number(h.sum()));
    out.set("count", count(h.count()));
    return out;
}

}  // namespace

bool valid_request_id(std::string_view id) noexcept {
    if (id.size() < 8U || id.size() > 64U) return false;
    for (const char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                        c == '-';
        if (!ok) return false;
    }
    return true;
}

bool query_flag(std::string_view query, std::string_view key) noexcept {
    std::size_t at = 0;
    while (at <= query.size()) {
        std::size_t end = query.find('&', at);
        if (end == std::string_view::npos) end = query.size();
        const std::string_view kv = query.substr(at, end - at);
        if (kv.size() > key.size() && kv.substr(0, key.size()) == key && kv[key.size()] == '=') {
            const std::string_view v = kv.substr(key.size() + 1U);
            if (v == "1" || v == "true") return true;
        }
        at = end + 1U;
    }
    return false;
}

// The one process-wide flag is the serve layer's (strix::serve::observe_armed): the queue reads it at submission.
bool observe_armed() noexcept { return sv::observe_armed(); }
void observe_arm() noexcept { sv::observe_arm(); }
void observe_reset_for_tests() noexcept { sv::observe_disarm(); }

std::string new_instance_id() {
    std::random_device rd;
    const std::uint64_t v = (static_cast<std::uint64_t>(rd()) << 32U) ^ static_cast<std::uint64_t>(rd());
    char b[17];
    std::snprintf(b, sizeof b, "%016" PRIx64, v);
    return b;
}

std::uint64_t RequestRecord::computed_tokens() const noexcept {
    const std::uint64_t reused = cached_tokens + imported_tokens;
    return prompt_tokens > reused ? prompt_tokens - reused : 0U;
}
double RequestRecord::decode_tok_s() const noexcept {
    return rate(output_tokens > 0U ? output_tokens - 1U : 0U, decode_ms);
}

Histogram::Histogram(std::vector<double> bounds) : bounds_(std::move(bounds)), counts_(bounds_.size() + 1U, 0U) {}

void Histogram::observe(double value) noexcept {
    if (!std::isfinite(value)) return;
    std::size_t i = 0;
    while (i < bounds_.size() && value > bounds_[i]) ++i;
    ++counts_[i];
    sum_ += value;
    ++count_;
}

std::uint64_t Histogram::cumulative(std::size_t bucket) const noexcept {
    std::uint64_t n = 0;
    for (std::size_t i = 0; i <= bucket && i < counts_.size(); ++i) n += counts_[i];
    return n;
}

const std::vector<double>& Observer::ttft_bounds() {
    static const std::vector<double> b{0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0};
    return b;
}
const std::vector<double>& Observer::decode_bounds() {
    static const std::vector<double> b{1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100};
    return b;
}

Observer::Observer(ObserverInfo info, std::size_t ring_size)
    : info_(std::move(info)), ring_size_(std::max<std::size_t>(ring_size, 1U)), started_(std::chrono::steady_clock::now()),
      ttft_(ttft_bounds()), decode_(decode_bounds()) {
    if (info_.instance_id.empty()) info_.instance_id = new_instance_id();
    if (info_.start_time == 0) info_.start_time = static_cast<std::int64_t>(std::time(nullptr));
}

void Observer::record(RequestRecord r) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (r.first_token_ms >= 0.0) ttft_.observe(r.first_token_ms / 1000.0);
    if (r.status == "done" && r.output_tokens > 1U && r.decode_ms > 0.0) decode_.observe(r.decode_tok_s());
    recent_.push_front(std::move(r));
    while (recent_.size() > ring_size_) recent_.pop_back();
}

sv::Json Observer::status(const sv::QueueSnapshot& q, bool stopping, bool detail) const {
    sv::Json out = sv::Json::object();
    out.set("schema", sv::Json("strix-serve-status-v1"));
    out.set("engine", sv::Json(info_.engine));
    out.set("model", sv::Json(info_.model));
    out.set("context_tokens", count(info_.context_tokens));
    out.set("uptime_seconds", number(std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count()));
    sv::Json instance = sv::Json::object();
    instance.set("id", sv::Json(info_.instance_id));
    instance.set("start_time", sv::Json(info_.start_time));  // the shared layer's name
    instance.set("started_at", sv::Json(info_.start_time));  // the same value, the name status collectors read
    out.set("instance", std::move(instance));
    out.set("armed", sv::Json(observe_armed()));
    sv::Json caps = sv::Json::object();
    caps.set("prefix_cache", sv::Json(false));
    caps.set("state_import", sv::Json(false));
    caps.set("draft", sv::Json(info_.draft));
    caps.set("return_token_ids", sv::Json(true));
    caps.set("request_ids", sv::Json(true));
    out.set("capabilities", std::move(caps));
    out.set("state", sv::Json(stopping ? "stopping" : q.running ? "generating" : q.queued ? "queued" : "idle"));
    sv::Json queue = sv::Json::object();
    queue.set("queued", count(q.queued));
    queue.set("running", count(q.running));
    queue.set("completed", count(q.completed));
    queue.set("failed", count(q.failed));
    queue.set("cancelled", count(q.cancelled));
    out.set("queue", std::move(queue));
    const std::uint64_t busy = std::min<std::uint64_t>(q.running, info_.slots);
    sv::Json slots = sv::Json::object();
    slots.set("total", count(info_.slots));
    slots.set("busy", count(busy));
    slots.set("free", count(info_.slots - busy));
    slots.set("context_tokens", count(info_.context_tokens));
    out.set("slots", std::move(slots));
    sv::Json group = sv::Json::object();
    group.set("rank", sv::Json(0));
    group.set("world", sv::Json(2));
    group.set("transport", sv::Json(info_.transport));
    out.set("group", std::move(group));
    sv::Json load = sv::Json::object();
    load.set("seconds", number(info_.load_seconds));
    load.set("device_used_bytes", count(info_.device_used_bytes));
    out.set("load", std::move(load));

    // lifetime totals: the engine queue is the one authority (this engine imports no state: imported is 0)
    sv::Json totals = sv::Json::object();
    totals.set("requests", count(q.completed + q.failed + q.cancelled));
    totals.set("prompt_tokens", count(q.prompt_tokens));
    totals.set("cached_tokens", count(q.cached_tokens));
    totals.set("imported_tokens", count(0U));
    totals.set("computed_tokens", count(q.prompt_tokens > q.cached_tokens ? q.prompt_tokens - q.cached_tokens : 0U));
    totals.set("generated_tokens", count(q.generated_tokens));
    totals.set("busy_seconds", number(q.busy_seconds));
    out.set("totals", std::move(totals));

    const std::lock_guard<std::mutex> lock(mutex_);
    sv::Json list = sv::Json::array();
    for (const auto& r : recent_) {
        sv::Json e = sv::Json::object();
        e.set("id", sv::Json(r.id()));
        e.set("request_id", r.request_id.empty() ? sv::Json() : sv::Json(r.request_id));
        e.set("completion_id", sv::Json(r.completion_id));
        e.set("route", sv::Json(r.route));
        e.set("time", sv::Json(r.time));
        e.set("status", sv::Json(r.status));
        e.set("stop_reason", sv::Json(r.stop_reason));
        if (!r.error.empty()) e.set("error", sv::Json(r.error));
        e.set("stream", sv::Json(r.stream));
        e.set("prompt_tokens", count(r.prompt_tokens));
        e.set("cached_tokens", count(r.cached_tokens));
        e.set("imported_tokens", count(r.imported_tokens));
        e.set("computed_tokens", count(r.computed_tokens()));
        e.set("reused_tokens", count(r.cached_tokens));
        e.set("output_tokens", count(r.output_tokens));
        e.set("queue_ms", number(r.queue_ms));
        e.set("prefill_ms", number(r.prefill_ms));
        e.set("decode_ms", number(r.decode_ms));
        e.set("duration_ms", number(r.queue_ms + r.prefill_ms + r.decode_ms));
        e.set("prefill_tok_s", number(rate(r.computed_tokens(), r.prefill_ms)));
        e.set("decode_tok_s", number(r.decode_tok_s()));
        e.set("first_token_ms", r.first_token_ms >= 0.0 ? number(r.first_token_ms) : sv::Json());
        e.set("max_gap_ms", r.max_gap_ms >= 0.0 ? number(r.max_gap_ms) : sv::Json());
        e.set("mtp_cycles", count(r.draft ? r.cycles : 0U));
        e.set("mtp_accepted", count(r.draft ? r.accepted : 0U));
        e.set("mtp_proposed", count(r.draft ? r.proposed : 0U));
        if (r.draft) {
            sv::Json d = sv::Json::object();
            d.set("cycles", count(r.cycles));
            d.set("proposed", count(r.proposed));
            d.set("accepted", count(r.accepted));
            e.set("draft", std::move(d));
        }
        list.push(std::move(e));
    }
    out.set("requests", std::move(list));
    if (detail) {
        sv::Json latency = sv::Json::object();
        latency.set("ttft_seconds", histogram_json(ttft_));
        latency.set("decode_tok_s", histogram_json(decode_));
        out.set("latency", std::move(latency));
    }
    return out;
}

std::string legacy_metric_lines(const sv::QueueSnapshot& s) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "mm_queued %llu\nmm_running %llu\nmm_completed %llu\nmm_failed %llu\nmm_prompt_tokens %llu\n"
                  "mm_generated_tokens %llu\nmm_busy_seconds %.3f\n",
                  static_cast<unsigned long long>(s.queued), static_cast<unsigned long long>(s.running),
                  static_cast<unsigned long long>(s.completed), static_cast<unsigned long long>(s.failed),
                  static_cast<unsigned long long>(s.prompt_tokens), static_cast<unsigned long long>(s.generated_tokens),
                  s.busy_seconds);
    return buf;
}

const std::vector<LegacyMetric>& legacy_metrics() {
    static const std::vector<LegacyMetric> t{
        {"mm_queued", "gauge", "Generations waiting in the queue (same value as strix_queue_waiting)."},
        {"mm_running", "gauge", "Generations running (same value as strix_queue_running)."},
        {"mm_completed", "counter", "Generations completed (same value as strix_requests_completed_total)."},
        {"mm_failed", "counter", "Generations that failed (same value as strix_requests_failed_total)."},
        {"mm_prompt_tokens", "counter", "Prompt tokens of finished generations (same value as strix_prompt_tokens_total)."},
        {"mm_generated_tokens", "counter", "Generated tokens (same value as strix_generated_tokens_total)."},
        {"mm_busy_seconds", "counter", "Seconds the engine spent on generations (same value as strix_engine_busy_seconds_total)."},
        // the serve scheduler's counters (zeroed by POST /admin/sched with reset_counters: a counter reset)
        {"mm_step_seconds", "counter", "Seconds the scheduler spent in decode steps."},
        {"mm_prefill_seconds", "counter", "Seconds the scheduler spent in prefill units."},
        {"mm_draft_seconds", "counter", "Seconds the scheduler spent drafting."},
        {"mm_idle_seconds", "counter", "Seconds the scheduler loop waited with no work."},
        {"mm_decode_stall_seconds", "counter", "Decoding-request seconds stalled by prefill units (summed over the decoding requests)."},
        {"mm_steps", "counter", "Decode steps."},
        {"mm_step_rows", "counter", "Rows of all decode steps."},
        {"mm_prefill_units", "counter", "Prefill units (a whole prefill is one unit)."},
        {"mm_draft_cycles", "counter", "Decode cycles with drafts."},
        {"mm_drafted", "counter", "Draft tokens proposed."},
        {"mm_accepted", "counter", "Draft tokens accepted."},
    };
    return t;
}

std::string with_help(std::string_view legacy, const std::vector<LegacyMetric>& table) {
    std::string m;
    std::size_t at = 0;
    while (at < legacy.size()) {
        std::size_t end = legacy.find('\n', at);
        if (end == std::string_view::npos) end = legacy.size();
        const std::string_view line = legacy.substr(at, end - at);
        if (!line.empty()) {
            const std::string_view name = line.substr(0, std::min(line.find(' '), line.find('{')));
            const LegacyMetric* known = nullptr;
            for (const auto& l : table)
                if (l.name == name) known = &l;
            if (known) family(m, known->name, known->type, known->help);
            else family(m, name, "untyped", "Serve metric.");
            m.append(line).append("\n");
        }
        at = end + 1U;
    }
    return m;
}

std::string Observer::metrics(const sv::QueueSnapshot& q) const { return metrics(q, legacy_metric_lines(q)); }

std::string Observer::metrics(const sv::QueueSnapshot& q, std::string_view legacy) const {
    std::string m = with_help(legacy, info_.legacy.empty() ? legacy_metrics() : info_.legacy);
    m.reserve(m.size() + 6144U);
    const std::uint64_t busy = std::min<std::uint64_t>(q.running, info_.slots);
    scalar(m, "strix_requests_completed_total", "counter", "Generations completed.", format_u64(q.completed));
    scalar(m, "strix_requests_failed_total", "counter", "Generations that failed.", format_u64(q.failed));
    scalar(m, "strix_requests_cancelled_total", "counter", "Generations cancelled (client gone).", format_u64(q.cancelled));
    scalar(m, "strix_prompt_tokens_total", "counter", "Prompt tokens of finished generations.", format_u64(q.prompt_tokens));
    scalar(m, "strix_prompt_tokens_cached_total", "counter", "Prompt tokens restored from a prefix cache (this engine has none).",
           format_u64(q.cached_tokens));
    scalar(m, "strix_generated_tokens_total", "counter", "Generated tokens.", format_u64(q.generated_tokens));
    scalar(m, "strix_engine_busy_seconds_total", "counter", "Seconds the engine spent on generations.", format_double(q.busy_seconds));
    scalar(m, "strix_queue_waiting", "gauge", "Generations waiting in the queue.", format_u64(q.queued));
    scalar(m, "strix_queue_running", "gauge", "Generations running.", format_u64(q.running));
    scalar(m, "strix_slots_busy", "gauge", "Sequence slots in use.", format_u64(busy));
    scalar(m, "strix_slots_total", "gauge", "Sequence slots.", format_u64(info_.slots));
    family(m, "strix_engine_info", "gauge", "Serving process identity (value 1).");
    sample(m, "strix_engine_info", "1",
           "engine=\"minimax\",model=\"" + label_value(info_.model) + "\",instance=\"" + label_value(info_.instance_id) + "\"");
    scalar(m, "strix_engine_start_time_seconds", "gauge", "Process start, unix seconds.",
           format_u64(static_cast<std::uint64_t>(std::max<std::int64_t>(info_.start_time, 0))));
    scalar(m, "strix_instrumentation_armed", "gauge", "1 once the armed instrumentation is on.", observe_armed() ? "1" : "0");

    scalar(m, "strix_prompt_tokens_imported_total", "counter", "Prompt tokens whose state was imported (this engine imports none).", "0");
    scalar(m, "strix_prompt_tokens_computed_total", "counter", "Prompt tokens prefilled by this engine (prompt - cached - imported).",
           format_u64(q.prompt_tokens > q.cached_tokens ? q.prompt_tokens - q.cached_tokens : 0U));
    const std::lock_guard<std::mutex> lock(mutex_);
    if (observe_armed()) {
        histogram_text(m, "strix_ttft_seconds", "Submit to first published token (engine scheduler), requests submitted while armed.",
                       ttft_);
        histogram_text(m, "strix_decode_tokens_per_second",
                       "Per-request decode rate (tokens after the first per decode second), completed requests submitted while armed.",
                       decode_);
    }
    return m;
}

sv::Json reply_extension(std::span<const std::uint32_t> token_ids) {
    sv::Json ids = sv::Json::array();
    for (const auto id : token_ids) ids.push(sv::Json(static_cast<std::int64_t>(id)));
    sv::Json out = sv::Json::object();
    out.set("token_ids", std::move(ids));
    return out;
}

}  // namespace strix::minimax::serve
