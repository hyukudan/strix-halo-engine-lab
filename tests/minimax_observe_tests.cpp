// strix-minimax-serve observability, host only: request ids, the armed flag, the recent-request ring, prompt
// accounting, status JSON and the metrics text of GET /metrics.
#include "minimax/serve/minimax_observe.hpp"
#include "strix/serve/json.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using strix::serve::Json;
namespace mo = strix::minimax::serve;

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << "\n";
    }
}

const Json& at(const Json& j, std::string_view key) {
    static const Json missing;
    const auto* v = j.find(key);
    return v ? *v : missing;
}

strix::serve::QueueSnapshot snapshot() {
    strix::serve::QueueSnapshot q;
    q.queued = 2;
    q.running = 3;
    q.completed = 41;
    q.failed = 1;
    q.cancelled = 4;
    q.prompt_tokens = 123456789012ULL;
    q.cached_tokens = 0;
    q.generated_tokens = 987654;
    q.busy_seconds = 1234.5678;
    return q;
}

mo::ObserverInfo info() {
    mo::ObserverInfo i;
    i.model = "minimax \"m3.1\"\\flash\n";
    i.context_tokens = 8256;
    i.slots = 8;
    i.transport = "usb4-host:test";
    i.load_seconds = 45.5;
    i.device_used_bytes = 98676645478ULL;
    i.instance_id = "0123456789abcdef";
    i.start_time = 1761350400;
    return i;
}

mo::RequestRecord record(const std::string& completion_id, std::uint64_t prompt, std::uint64_t output) {
    mo::RequestRecord r;
    r.completion_id = completion_id;
    r.route = "chat.completions";
    r.time = 1761350500;
    r.status = "done";
    r.stop_reason = "stop";
    r.prompt_tokens = prompt;
    r.output_tokens = output;
    r.queue_ms = 2.0;
    r.prefill_ms = 500.0;
    r.decode_ms = 1000.0;
    r.first_token_ms = 520.0;
    return r;
}

void test_request_ids() {
    check(mo::valid_request_id("abcdefgh"), "8 characters");
    check(mo::valid_request_id(std::string(64, 'a')), "64 characters");
    check(mo::valid_request_id("0123456789abcdef.1"), "router form id.attempt");
    check(mo::valid_request_id("Az09._-x"), "every allowed class");
    check(!mo::valid_request_id("abcdefg"), "7 characters");
    check(!mo::valid_request_id(std::string(65, 'a')), "65 characters");
    check(!mo::valid_request_id(""), "empty");
    for (const char* bad : {"abcd efgh", "abcd/efgh", "abcd:efgh", "abcd\refgh", "abcd\nefgh", "abcd\"efgh", "abcd\xc3\xa9" "efgh",
                            "abcd%20efgh", "abcd+efgh"})
        check(!mo::valid_request_id(bad), std::string("rejected ") + bad);
}

void test_query_flag() {
    check(mo::query_flag("detail=1", "detail"), "detail=1");
    check(mo::query_flag("a=2&detail=true", "detail"), "detail=true after another key");
    check(!mo::query_flag("detail=0", "detail"), "detail=0");
    check(!mo::query_flag("", "detail"), "empty query");
    check(!mo::query_flag("xdetail=1", "detail"), "other key with the name as suffix");
    check(!mo::query_flag("detail", "detail"), "key without value");
    check(!mo::query_flag("detailx=1", "detail"), "other key with the name as prefix");
}

void test_instance_id() {
    const auto a = mo::new_instance_id(), b = mo::new_instance_id();
    bool hex = a.size() == 16U;
    for (const char c : a) hex = hex && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    check(hex, "instance id is 16 lowercase hex");
    check(a != b, "instance ids differ");
}

void test_ring_and_status() {
    mo::observe_reset_for_tests();
    mo::Observer obs(info(), 50);
    auto first = record("chatcmpl-1", 100, 11);
    obs.record(first);
    for (int i = 2; i <= 60; ++i) {
        auto r = record("chatcmpl-" + std::to_string(i), 10U + static_cast<std::uint64_t>(i), 5);
        if (i == 60) {
            r.request_id = "0123456789abcdef.2";
            r.first_token_ms = -1.0;
        }
        obs.record(r);
    }
    const auto q = snapshot();
    const auto s = strix::serve::parse_json(obs.status(q, false, false).dump());
    check(at(s, "schema").as_string() == "strix-serve-status-v1", "schema");
    check(at(s, "engine").as_string() == "strix-minimax-serve", "engine");
    check(at(s, "armed").is_bool() && !at(s, "armed").as_bool(), "unarmed");
    check(at(s, "state").as_string() == "generating", "state from running");
    check(at(at(s, "instance"), "id").as_string() == "0123456789abcdef", "instance id");
    check(at(at(s, "instance"), "start_time").as_int() == 1761350400, "instance start");
    check(at(at(s, "instance"), "started_at").as_int() == 1761350400, "instance started_at (same value)");
    check(at(at(s, "queue"), "cancelled").as_int() == 4, "queue counters");
    check(at(at(s, "totals"), "requests").as_int() == 46, "totals.requests = the queue's finished requests");
    check(at(at(s, "totals"), "prompt_tokens").as_int() == 123456789012LL, "totals.prompt_tokens from the queue");
    check(at(at(s, "totals"), "computed_tokens").as_int() == 123456789012LL, "computed = prompt - cached - imported");
    check(at(at(s, "totals"), "imported_tokens").as_int() == 0, "imported");
    check(at(at(s, "slots"), "busy").as_int() == 3 && at(at(s, "slots"), "free").as_int() == 5, "slots");
    check(at(at(s, "load"), "device_used_bytes").as_int() == 98676645478LL, "load bytes");
    check(!s.find("latency"), "no latency without detail");
    const auto& list = at(s, "requests");
    check(list.is_array() && list.size() == 50U, "ring keeps 50");
    const auto& newest = list.as_array().front();
    check(at(newest, "id").as_string() == "0123456789abcdef.2", "id = accepted inbound request id");
    check(at(newest, "request_id").as_string() == "0123456789abcdef.2", "request_id");
    check(at(newest, "completion_id").as_string() == "chatcmpl-60", "completion id");
    check(at(newest, "first_token_ms").is_null(), "first_token_ms null when not measured");
    check(at(list.as_array()[1], "first_token_ms").as_double() == 520.0, "first_token_ms kept");
    const auto& second = list.as_array()[1];
    check(at(second, "id").as_string() == "chatcmpl-59", "id falls back to the completion id");
    check(at(second, "request_id").is_null(), "request_id null without an inbound id");
    check(at(list.as_array().back(), "completion_id").as_string() == "chatcmpl-11", "oldest kept = 11th");
    // per-record fields as the engine adapter reads them
    for (const char* k : {"prompt_tokens", "reused_tokens", "output_tokens", "mtp_cycles", "mtp_accepted", "mtp_proposed", "time",
                          "cached_tokens", "imported_tokens", "computed_tokens"})
        check(at(newest, k).is_number() && at(newest, k).number_source().find_first_of(".eE") == std::string::npos,
              std::string("integer field ") + k);
    for (const char* k : {"queue_ms", "prefill_ms", "decode_ms", "duration_ms", "prefill_tok_s", "decode_tok_s"})
        check(at(newest, k).is_number(), std::string("number field ") + k);
    for (const char* k : {"id", "route", "status", "stop_reason"}) check(at(newest, k).is_string(), std::string("string field ") + k);
    check(std::abs(at(newest, "duration_ms").as_double() - 1502.0) < 1e-9, "duration = queue + prefill + decode");
    check(std::abs(at(newest, "prefill_tok_s").as_double() - 140.0) < 1e-9, "prefill rate from computed tokens");
    check(std::abs(at(newest, "decode_tok_s").as_double() - 4.0) < 1e-9, "decode rate = (output - 1) / decode s");
    check(!newest.find("draft") && !newest.find("error"), "no draft / error keys when absent");
    check(at(newest, "stream").is_bool(), "stream flag");
}

void test_histograms() {
    mo::observe_reset_for_tests();
    mo::Observer obs(info());
    auto a = record("chatcmpl-2", 10, 21);
    a.first_token_ms = 75.0;  // 0.075 s -> le 0.1
    obs.record(a);
    auto b = record("chatcmpl-3", 10, 1);  // one token: no decode rate
    b.first_token_ms = 120000.0;              // 120 s -> +Inf
    obs.record(b);
    auto c = record("chatcmpl-4", 10, 0);
    c.status = "error";
    c.stop_reason.clear();
    c.error = "prompt does not fit";
    c.first_token_ms = -1.0;
    obs.record(c);
    auto d = record("chatcmpl-5", 10, 30);
    d.draft = true;
    d.cycles = 7;
    d.proposed = 21;
    d.accepted = 13;
    d.first_token_ms = 0.0;  // boundary: le 0.05
    obs.record(d);
    mo::observe_arm();
    const auto s = strix::serve::parse_json(obs.status(snapshot(), true, true).dump());
    check(at(s, "state").as_string() == "stopping", "stopping state");
    check(at(s, "armed").as_bool(), "armed in status");
    const auto& ttft = at(at(s, "latency"), "ttft_seconds");
    check(at(ttft, "count").as_int() == 3, "ttft count (records with a first token)");
    const auto& counts = at(ttft, "counts").as_array();
    check(at(ttft, "le").size() == 10U && counts.size() == 11U, "ttft buckets: finite bounds + the +Inf count");
    check(counts[0].as_int() == 1 && counts[1].as_int() == 2 && counts[9].as_int() == 2 && counts[10].as_int() == 3, "ttft cumulative");
    check(std::abs(at(ttft, "sum").as_double() - 120.075) < 1e-9, "ttft sum seconds");
    const auto& dec = at(at(s, "latency"), "decode_tok_s");
    check(at(dec, "count").as_int() == 2, "decode count: done with > 1 token");
    const auto& list = at(s, "requests").as_array();
    check(at(list[0], "first_token_ms").as_double() == 0.0, "first_token_ms 0 kept");
    check(at(at(list[0], "draft"), "accepted").as_int() == 13 && at(list[0], "mtp_proposed").as_int() == 21, "draft counts");
    check(at(list[1], "status").as_string() == "error" && at(list[1], "error").as_string() == "prompt does not fit", "error record");
    check(at(list[1], "first_token_ms").is_null(), "no first token on error");
    check(at(list[3], "first_token_ms").as_double() == 75.0, "record keeps first_token_ms");
}

// ---- a small validator of the metrics text exposition format ----
struct Family {
    std::string type;
    bool help = false;
    int samples = 0;
};
bool metric_name(const std::string& n) {
    if (n.empty()) return false;
    for (std::size_t i = 0; i < n.size(); ++i) {
        const char c = n[i];
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == ':' || (i > 0 && c >= '0' && c <= '9');
        if (!ok) return false;
    }
    return true;
}
std::string family_of(const std::string& sample, const std::map<std::string, Family>& families) {
    if (families.count(sample)) return sample;
    for (const char* suffix : {"_bucket", "_sum", "_count"}) {
        const std::string s(suffix);
        if (sample.size() > s.size() && sample.compare(sample.size() - s.size(), s.size(), s) == 0) {
            const auto base = sample.substr(0, sample.size() - s.size());
            if (families.count(base) && families.at(base).type == "histogram") return base;
        }
    }
    return {};
}
bool allow_untyped = false;
std::map<std::string, Family> validate(const std::string& text, std::map<std::string, std::string>& values) {
    std::map<std::string, Family> families;
    std::istringstream in(text);
    std::string line;
    std::set<std::string> seen_samples;
    check(!text.empty() && text.back() == '\n', "metrics end with a newline");
    while (std::getline(in, line)) {
        check(!line.empty(), "no blank lines");
        if (line.rfind("# HELP ", 0) == 0) {
            const auto rest = line.substr(7);
            const auto name = rest.substr(0, rest.find(' '));
            check(metric_name(name) && !families[name].help && families[name].samples == 0, "HELP once before samples: " + name);
            families[name].help = true;
            continue;
        }
        if (line.rfind("# TYPE ", 0) == 0) {
            std::istringstream ls(line.substr(7));
            std::string name, type;
            ls >> name >> type;
            check(families[name].type.empty() && families[name].samples == 0, "TYPE once before samples: " + name);
            check(type == "counter" || type == "gauge" || type == "histogram" || (allow_untyped && type == "untyped"), "known type " + type);
            families[name].type = type;
            continue;
        }
        check(line[0] != '#', "no other comments");
        const auto space = line.rfind(' ');
        const auto key = line.substr(0, space), value = line.substr(space + 1);
        const auto brace = key.find('{');
        const auto name = key.substr(0, brace);
        check(metric_name(name), "sample name " + name);
        check(seen_samples.insert(key).second, "unique sample " + key);
        if (brace != std::string::npos) check(key.back() == '}', "label set closed " + key);
        char* end = nullptr;
        const double v = std::strtod(value.c_str(), &end);
        check(end && *end == '\0' && std::isfinite(v) && v >= 0.0, "sample value " + line);
        const auto fam = family_of(name, families);
        check(!fam.empty() && families[fam].help && !families[fam].type.empty(), "sample has HELP and TYPE: " + name);
        if (!fam.empty()) ++families[fam].samples;
        values[key] = value;
    }
    for (const auto& [name, f] : families) check(f.samples > 0, "family has samples: " + name);
    return families;
}

std::map<std::string, Family> validate_any(const std::string& text, std::map<std::string, std::string>& values) {
    allow_untyped = true;
    auto f = validate(text, values);
    allow_untyped = false;
    return f;
}

void test_metrics() {
    mo::observe_reset_for_tests();
    mo::Observer obs(info());
    const auto q = snapshot();
    std::map<std::string, std::string> v;
    auto fam = validate(obs.metrics(q), v);
    // legacy lines byte-identical and in their order
    const auto legacy = mo::legacy_metric_lines(q);
    check(legacy == "mm_queued 2\nmm_running 3\nmm_completed 41\nmm_failed 1\nmm_prompt_tokens 123456789012\n"
                    "mm_generated_tokens 987654\nmm_busy_seconds 1234.568\n",
          "legacy format");
    {
        std::istringstream in(obs.metrics(q));
        std::string line, kept;
        while (std::getline(in, line))
            if (line.rfind("mm_", 0) == 0) kept += line + "\n";
        check(kept == legacy, "legacy value lines unchanged in /metrics");
    }
    check(fam["mm_completed"].type == "counter" && fam["mm_queued"].type == "gauge", "legacy types");
    check(v["strix_prompt_tokens_total"] == "123456789012", "large counter exact");
    check(v["strix_generated_tokens_total"] == "987654", "generated");
    check(v["strix_prompt_tokens_computed_total"] == "123456789012", "computed");
    check(v["strix_prompt_tokens_imported_total"] == "0", "imported");
    check(v["strix_slots_busy"] == "3" && v["strix_slots_total"] == "8", "slots");
    check(v["strix_engine_start_time_seconds"] == "1761350400", "start time");
    check(v["strix_instrumentation_armed"] == "0", "armed gauge before arming (direct render)");
    check(v.count("strix_engine_info{engine=\"minimax\",model=\"minimax \\\"m3.1\\\"\\\\flash\\n\",instance=\"0123456789abcdef\"}") == 1U,
          "info labels escaped");
    check(!fam.count("strix_ttft_seconds"), "no histograms unarmed");
    for (const char* name : {"strix_requests_completed_total", "strix_requests_failed_total", "strix_requests_cancelled_total",
                             "strix_prompt_tokens_cached_total", "strix_engine_busy_seconds_total", "strix_queue_waiting",
                             "strix_queue_running"})
        check(fam.count(name) == 1U, std::string("shared family ") + name);

    mo::observe_arm();
    auto r = record("chatcmpl-1", 10, 11);
    r.first_token_ms = 300.0;  // 0.3 s -> le 0.5
    obs.record(r);
    std::map<std::string, std::string> w;
    fam = validate(obs.metrics(q), w);
    check(fam["strix_ttft_seconds"].type == "histogram" && fam["strix_decode_tokens_per_second"].type == "histogram", "armed histograms");
    check(w["strix_instrumentation_armed"] == "1", "armed gauge");
    check(w["strix_ttft_seconds_bucket{le=\"0.25\"}"] == "0" && w["strix_ttft_seconds_bucket{le=\"0.5\"}"] == "1" &&
              w["strix_ttft_seconds_bucket{le=\"+Inf\"}"] == "1" && w["strix_ttft_seconds_count"] == "1",
          "ttft buckets cumulative, +Inf == count");
    check(w["strix_ttft_seconds_sum"] == "0.3", "ttft sum");
    check(w["strix_decode_tokens_per_second_bucket{le=\"10\"}"] == "1" && w["strix_decode_tokens_per_second_bucket{le=\"5\"}"] == "0",
          "decode bucket (10 tok/s)");
    // cumulative monotone
    std::uint64_t prev = 0;
    for (const double b : mo::Observer::ttft_bounds()) {
        std::ostringstream k;
        k << "strix_ttft_seconds_bucket{le=\"" << b << "\"}";
        const auto it = w.find(k.str());
        check(it != w.end(), "bucket present " + k.str());
        if (it != w.end()) {
            const auto n = std::stoull(it->second);
            check(n >= prev, "monotone " + k.str());
            prev = n;
        }
    }
}

void test_port_extensions() {
    mo::observe_reset_for_tests();
    auto i = info();
    i.draft = true;
    i.legacy = {{"mm_queued", "gauge", "Queued."}, {"mm_steps", "counter", "Decode steps."}};
    mo::Observer obs(i);
    auto r = record("chatcmpl-1", 10, 5);
    r.max_gap_ms = 61.5;
    obs.record(r);
    auto r2 = record("chatcmpl-2", 10, 5);
    obs.record(r2);
    const auto s = strix::serve::parse_json(obs.status(snapshot(), false, false).dump());
    check(at(at(s, "capabilities"), "draft").as_bool(), "capabilities.draft from the configuration");
    const auto& list = at(s, "requests").as_array();
    check(at(list[1], "max_gap_ms").as_double() == 61.5 && at(list[0], "max_gap_ms").is_null(), "max_gap_ms number or null");
    // a build with more legacy lines passes its own text and table; unknown names are untyped
    const std::string legacy = "mm_queued 2\nmm_steps 77\nmm_other{a=\"b\"} 1.5\n";
    std::map<std::string, std::string> v;
    auto fam = validate_any(obs.metrics(snapshot(), legacy), v);
    check(fam["mm_steps"].type == "counter" && fam["mm_queued"].type == "gauge" && fam["mm_other"].type == "untyped", "legacy table types");
    check(v["mm_steps"] == "77" && v["mm_other{a=\"b\"}"] == "1.5", "legacy values unchanged");
    check(mo::with_help("mm_queued 1\n", mo::legacy_metrics()) ==
              "# HELP mm_queued Generations waiting in the queue (same value as strix_queue_waiting).\n# TYPE mm_queued gauge\nmm_queued 1\n",
          "with_help line");
}

void test_reply_extension() {
    const std::vector<std::uint32_t> ids{200019, 0, 4294967295U, 7};
    const auto j = strix::serve::parse_json(mo::reply_extension(ids).dump());
    const auto& arr = at(j, "token_ids").as_array();
    check(arr.size() == 4U && arr[0].as_int() == 200019 && arr[1].as_int() == 0 && arr[2].as_int() == 4294967295LL && arr[3].as_int() == 7,
          "token ids kept exactly");
    check(at(strix::serve::parse_json(mo::reply_extension({}).dump()), "token_ids").size() == 0U, "empty ids");
}

}  // namespace

int main() {
    test_request_ids();
    test_query_flag();
    test_instance_id();
    test_ring_and_status();
    test_histograms();
    test_metrics();
    test_port_extensions();
    test_reply_extension();
    std::cout << (failures ? "FAIL" : "PASS") << " strix-minimax-observe-tests: " << checks - failures << "/" << checks << " checks\n";
    return failures ? 1 : 0;
}
