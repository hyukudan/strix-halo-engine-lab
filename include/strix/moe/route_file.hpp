#pragma once
// MoE route record / replay / null tap: the engine side of the FAST gate v2 (docs: tools/strix-gate/README.md,
// "FAST gate v2"). Backend-neutral host code. A backend binds it in one place: after its router wrote the selected
// expert ids and weights of a call, it copies them to host, calls RouteTap::on_call and copies them back when that
// returns true. With no switch set the tap is inactive and the backend copies nothing.
//
//   STRIX_MOE_ROUTES_OUT=FILE   record every call's ids and weights (as this arm computed them)
//   STRIX_MOE_ROUTES_IN=FILE    replay: overwrite ids and weights with the recorded ones, call by call; the layer,
//                               row count and top-k of every call must match the record (else std::runtime_error)
//   STRIX_MOE_ROUTE_NULL=L:SEED one-ulp null control: every selected weight of layer L moves by one FP32 ulp, up or
//                               down by a seeded hash (applied after replay)
//   STRIX_MOE_ROUTE_STATS=1     statistics only (never changes ids or weights): per layer the calls, rows, assignments,
//                               distinct experts per call (u_sum) and the rows a one-expert-per-tile grouped GEMM pads
//                               at tile heights 16 and 64; per call width (rows) the calls, u_sum and assignments
//                               (at most kRouteStatsMaxWidths distinct widths, the rest fold into "other"). Adds a
//                               "stats":{...} object to the exit line below.
//
// File strix-route-file/v1 (little-endian): 16-byte header "STXROUT1", u32 version 1, u32 0; then per call
// u32 layer, u32 rows, u32 top_k, u32 0, u32 ids[rows * top_k], f32 weights[rows * top_k] (row-major).
// At exit the process tap prints one line `strix-route-file/v1 {json}` on stderr: records written / consumed,
// local route flips per layer (the arm's own ids vs the replayed ones) and the null's nudge count; with
// STRIX_MOE_ROUTE_STATS=1 also "stats":{"layers":[{"layer":L,"calls":..,"rows":..,"assignments":..,"u_sum":..,
// "pad16":..,"pad64":..}],"widths":[{"rows":M,"calls":..,"assignments":..,"u_sum":..}],"other":{...}}.
//
// Sequence segments (one process, several prompts): when STRIX_MOE_ROUTES_OUT or _IN contains "{seq}", the engine calls
// route_tap_sequence(i) before prompt i (0, 1, ...). Each sequence is its own segment: "{seq}" becomes i in both paths,
// every counter (records, flips, null index, stats) restarts and the closing segment prints its own receipt line. A
// process that runs N sequences writes the same files and receipts as N processes that each run one of them with the
// path written out. Without "{seq}" route_tap_sequence() changes nothing (one file and one receipt per process).
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/env_flags.hpp"

namespace strix::moe {

inline constexpr char kRouteFileMagic[8] = {'S', 'T', 'X', 'R', 'O', 'U', 'T', '1'};
inline constexpr std::uint32_t kRouteFileVersion = 1;
inline constexpr std::uint32_t kRouteMaxTopK = 64;
inline constexpr std::size_t kRouteStatsMaxWidths = 64;

struct RouteRecordHeader {
    std::uint32_t layer = 0, rows = 0, top_k = 0, flags = 0;
};

// Ids of the set a[0..k) that are missing from b[0..k) (order-free; both sets of one row).
inline std::uint32_t route_set_diff(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t k) {
    std::uint32_t missing = 0;
    for (std::uint32_t i = 0; i < k; ++i)
        if (std::find(b, b + k, a[i]) == b + k) ++missing;
    return missing;
}

inline std::uint64_t route_mix64(std::uint64_t x) {  // splitmix64 finaliser
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

// One FP32 ulp up or down (sign from the seeded hash of index); finite in, finite out; zero moves to +-denorm min.
inline float route_ulp_nudge(float w, std::uint64_t seed, std::uint64_t index) {
    const bool up = (route_mix64(seed ^ route_mix64(index)) & 1U) != 0U;
    const float to = up ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    const float out = std::nextafter(w, to);
    return std::isfinite(out) ? out : w;
}

// One router call: distinct experts and the rows a grouped tile (one expert per tile) pads at heights 16 and 64.
struct RouteCallStats {
    std::uint32_t assignments = 0, u = 0, pad16 = 0, pad64 = 0;
};

// Pure: ids is rows x top_k row-major. scratch is reused between calls (the tap keeps one; no allocation after warm-up).
inline RouteCallStats route_call_stats(const std::uint32_t* ids, std::uint32_t rows, std::uint32_t top_k,
                                       std::vector<std::uint32_t>& scratch) {
    RouteCallStats r;
    const std::size_t n = static_cast<std::size_t>(rows) * top_k;
    r.assignments = static_cast<std::uint32_t>(n);
    scratch.assign(ids, ids + n);
    std::sort(scratch.begin(), scratch.end());
    for (std::size_t i = 0; i < n;) {
        std::size_t j = i;
        while (j < n && scratch[j] == scratch[i]) ++j;
        const auto c = static_cast<std::uint32_t>(j - i);
        ++r.u;
        r.pad16 += (16U - c % 16U) % 16U;
        r.pad64 += (64U - c % 64U) % 64U;
        i = j;
    }
    return r;
}
inline RouteCallStats route_call_stats(const std::uint32_t* ids, std::uint32_t rows, std::uint32_t top_k) {
    std::vector<std::uint32_t> scratch;
    return route_call_stats(ids, rows, top_k, scratch);
}

// Accumulated sums (plain data). Layer rows carry pad16 / pad64; width rows are keyed by the call's row count.
struct RouteStatsSums {
    std::uint64_t calls = 0, rows = 0, assignments = 0, u_sum = 0, pad16 = 0, pad64 = 0;
};
struct RouteStats {
    std::map<std::uint32_t, RouteStatsSums> layers;
    std::map<std::uint32_t, RouteStatsSums> widths;  // keyed by rows (at most kRouteStatsMaxWidths entries)
    RouteStatsSums other;                            // widths beyond the cap
};

struct RouteTapConfig {
    std::string out_path, in_path;
    bool null_on = false;
    bool stats_on = false;
    std::uint32_t null_layer = 0;
    std::uint64_t null_seed = 0;
    bool active() const { return !out_path.empty() || !in_path.empty() || null_on || stats_on; }
};

// "L:SEED" (or "L", seed 0); refuses anything else.
inline void parse_route_null(const std::string& text, RouteTapConfig& c) {
    const auto colon = text.find(':');
    const auto layer = text.substr(0, colon);
    const auto seed = colon == std::string::npos ? std::string("0") : text.substr(colon + 1);
    const auto digits = [](const std::string& s) {
        return !s.empty() && s.size() <= 19 && std::all_of(s.begin(), s.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
    };
    if (!digits(layer) || !digits(seed) || std::stoull(layer) > 0xFFFFFFFFULL)
        throw std::invalid_argument("STRIX_MOE_ROUTE_NULL must be LAYER or LAYER:SEED (decimal), got '" + text + "'");
    c.null_on = true;
    c.null_layer = static_cast<std::uint32_t>(std::stoull(layer));
    c.null_seed = std::stoull(seed);
}

inline RouteTapConfig route_tap_config_from_env() {
    RouteTapConfig c;
    if (const char* v = envcfg::text("STRIX_MOE_ROUTES_OUT"); v && *v) c.out_path = v;
    if (const char* v = envcfg::text("STRIX_MOE_ROUTES_IN"); v && *v) c.in_path = v;
    if (const char* v = envcfg::text("STRIX_MOE_ROUTE_NULL"); v && *v) parse_route_null(v, c);
    c.stats_on = envcfg::flag("STRIX_MOE_ROUTE_STATS", false, EnvFlagMatch::exact);
    if (!c.out_path.empty() && c.out_path == c.in_path)
        throw std::invalid_argument("STRIX_MOE_ROUTES_OUT and STRIX_MOE_ROUTES_IN name the same file");
    return c;
}

class RouteTap {
public:
    explicit RouteTap(RouteTapConfig config)
        : out_pattern_(config.out_path), in_pattern_(config.in_path), config_(std::move(config)) {
        open_segment(0);
    }
    RouteTap(const RouteTap&) = delete;
    RouteTap& operator=(const RouteTap&) = delete;

    bool active() const { return config_.active(); }
    const RouteTapConfig& config() const { return config_; }
    // True when a path names "{seq}": route_tap_sequence() then starts a new segment per prompt.
    bool segmented() const { return has_seq(out_pattern_) || has_seq(in_pattern_); }
    std::uint64_t sequence() const { return seq_; }

    // Starts sequence `seq` (call before the prompt's first router call). Segmented: the current segment prints its
    // receipt line (the same line a process prints at exit) and every counter restarts on the new paths. Not
    // segmented, inactive or the current sequence: nothing changes. `sink` receives the closing receipt.
    template <class Sink> void sequence(std::uint64_t seq, Sink&& sink) {
        if (!active() || !segmented() || seq == seq_) return;
        sink(receipt());
        std::lock_guard<std::mutex> lock(mutex_);
        if (out_.is_open()) {
            out_.close();
            if (!out_) throw std::runtime_error("route tap: close failed on " + config_.out_path);
        }
        if (in_.is_open()) in_.close();
        out_ = std::ofstream();
        in_ = std::ifstream();
        out_records_ = out_rows_ = in_total_ = in_consumed_ = in_rows_ = nudged_ = null_index_ = 0;
        flips_.clear();
        stats_ = RouteStats{};
        seq_ = seq;
        open_segment(seq);
    }
    const RouteStats& stats() const { return stats_; }  // not locked: read after the calls

    // ids / weights: host copies of one router call, rows x top_k row-major. Returns true when they were changed
    // (replay or null): the backend must copy them back before the experts read them.
    bool on_call(std::uint32_t layer, std::uint32_t rows, std::uint32_t top_k, std::uint32_t* ids, float* weights) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active()) return false;
        if (rows == 0 || top_k == 0 || top_k > kRouteMaxTopK)
            throw std::invalid_argument("route tap: call with zero rows or an unsupported top-k");
        const std::size_t n = static_cast<std::size_t>(rows) * top_k;
        const RouteRecordHeader h{layer, rows, top_k, 0};
        if (out_.is_open()) {
            out_.write(reinterpret_cast<const char*>(&h), sizeof(h));
            out_.write(reinterpret_cast<const char*>(ids), static_cast<std::streamsize>(n * sizeof(std::uint32_t)));
            out_.write(reinterpret_cast<const char*>(weights), static_cast<std::streamsize>(n * sizeof(float)));
            if (!out_) throw std::runtime_error("route tap: write failed on " + config_.out_path);
            ++out_records_;
            out_rows_ += rows;
        }
        bool changed = false;
        if (in_.is_open()) {
            RouteRecordHeader r{};
            in_.read(reinterpret_cast<char*>(&r), sizeof(r));
            if (!in_)
                throw std::runtime_error("route replay: " + config_.in_path + " exhausted at call " + std::to_string(in_consumed_) +
                                         " (layer " + std::to_string(layer) + "): the arm runs more router calls than the reference");
            if (r.layer != layer || r.rows != rows || r.top_k != top_k)
                throw std::runtime_error("route replay: call " + std::to_string(in_consumed_) + " is layer " + std::to_string(layer) +
                                         " rows " + std::to_string(rows) + " top-k " + std::to_string(top_k) + ", the record has layer " +
                                         std::to_string(r.layer) + " rows " + std::to_string(r.rows) + " top-k " + std::to_string(r.top_k));
            replay_ids_.resize(n);
            replay_weights_.resize(n);
            in_.read(reinterpret_cast<char*>(replay_ids_.data()), static_cast<std::streamsize>(n * sizeof(std::uint32_t)));
            in_.read(reinterpret_cast<char*>(replay_weights_.data()), static_cast<std::streamsize>(n * sizeof(float)));
            if (!in_) throw std::runtime_error("route replay: truncated record " + std::to_string(in_consumed_));
            auto& f = flips_[layer];
            for (std::uint32_t row = 0; row < rows; ++row) {
                const auto d = route_set_diff(replay_ids_.data() + std::size_t{row} * top_k, ids + std::size_t{row} * top_k, top_k);
                f.slots += top_k;
                f.ids += d;
                f.rows_changed += d ? 1U : 0U;
            }
            std::memcpy(ids, replay_ids_.data(), n * sizeof(std::uint32_t));
            std::memcpy(weights, replay_weights_.data(), n * sizeof(float));
            ++in_consumed_;
            in_rows_ += rows;
            changed = true;
        }
        if (config_.null_on && layer == config_.null_layer) {
            for (std::size_t i = 0; i < n; ++i) {
                weights[i] = route_ulp_nudge(weights[i], config_.null_seed, null_index_ + i);
                ++nudged_;
            }
            null_index_ += n;
            changed = true;
        }
        if (config_.stats_on) {  // the ids the experts will read (after replay)
            const auto c = route_call_stats(ids, rows, top_k, stats_scratch_);
            const auto add = [&](RouteStatsSums& t) {
                ++t.calls; t.rows += rows; t.assignments += c.assignments; t.u_sum += c.u; t.pad16 += c.pad16; t.pad64 += c.pad64;
            };
            add(stats_.layers[layer]);
            const auto w = stats_.widths.find(rows);
            if (w != stats_.widths.end()) add(w->second);
            else if (stats_.widths.size() < kRouteStatsMaxWidths) add(stats_.widths[rows]);
            else add(stats_.other);
        }
        return changed;
    }

    // One JSON object (no newline): the replay / null bookkeeping the gate reads back.
    std::string receipt() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (out_.is_open()) out_.flush();
        std::string s = "{\"version\":1";
        if (!config_.out_path.empty())
            s += ",\"out\":{\"path\":" + quote(config_.out_path) + ",\"records\":" + std::to_string(out_records_) +
                 ",\"rows\":" + std::to_string(out_rows_) + "}";
        if (!config_.in_path.empty()) {
            std::uint64_t slots = 0, ids = 0, rows_changed = 0;
            std::string per;
            for (const auto& [layer, f] : flips_) {
                slots += f.slots; ids += f.ids; rows_changed += f.rows_changed;
                per += (per.empty() ? "" : ",") + std::string("[") + std::to_string(layer) + "," + std::to_string(f.slots) + "," +
                       std::to_string(f.ids) + "]";
            }
            s += ",\"in\":{\"path\":" + quote(config_.in_path) + ",\"records\":" + std::to_string(in_total_) +
                 ",\"consumed\":" + std::to_string(in_consumed_) + ",\"rows\":" + std::to_string(in_rows_) +
                 ",\"complete\":" + (in_consumed_ == in_total_ ? "true" : "false") + "}";
            s += ",\"local_flips\":{\"slots\":" + std::to_string(slots) + ",\"ids\":" + std::to_string(ids) +
                 ",\"rows_changed\":" + std::to_string(rows_changed) + ",\"per_layer\":[" + per + "]}";
        }
        if (config_.null_on)
            s += ",\"null\":{\"layer\":" + std::to_string(config_.null_layer) + ",\"seed\":" + std::to_string(config_.null_seed) +
                 ",\"nudged\":" + std::to_string(nudged_) + "}";
        if (config_.stats_on) s += ",\"stats\":" + stats_json();
        return s + "}";
    }

    // The process tap (environment switches, read once). Prints its receipt on stderr at exit when active.
    static RouteTap& instance();

private:
    static bool has_seq(const std::string& p) { return p.find("{seq}") != std::string::npos; }
    static std::string segment_path(std::string p, std::uint64_t seq) {
        for (auto at = p.find("{seq}"); at != std::string::npos; at = p.find("{seq}", at)) {
            const auto v = std::to_string(seq);
            p.replace(at, 5, v);
            at += v.size();
        }
        return p;
    }
    // Resolves "{seq}" for sequence `seq` and opens the segment's files (the constructor's former body).
    void open_segment(std::uint64_t seq) {
        config_.out_path = segment_path(out_pattern_, seq);
        config_.in_path = segment_path(in_pattern_, seq);
        if (!config_.out_path.empty()) {
            out_.open(config_.out_path, std::ios::binary | std::ios::trunc);
            if (!out_) throw std::runtime_error("route tap: cannot create " + config_.out_path);
            const std::uint32_t head[2] = {kRouteFileVersion, 0};
            out_.write(kRouteFileMagic, 8);
            out_.write(reinterpret_cast<const char*>(head), sizeof(head));
        }
        if (!config_.in_path.empty()) {
            in_.open(config_.in_path, std::ios::binary);
            if (!in_) throw std::runtime_error("route tap: cannot open " + config_.in_path);
            char magic[8] = {};
            std::uint32_t head[2] = {};
            in_.read(magic, 8);
            in_.read(reinterpret_cast<char*>(head), sizeof(head));
            if (!in_ || std::memcmp(magic, kRouteFileMagic, 8) != 0 || head[0] != kRouteFileVersion)
                throw std::runtime_error("route tap: " + config_.in_path + " is not a strix-route-file/v1");
            in_total_ = count_records(config_.in_path);
        }
    }

    static std::string sums_json(const RouteStatsSums& t, bool layer_form) {
        std::string s = "\"calls\":" + std::to_string(t.calls);
        if (layer_form) s += ",\"rows\":" + std::to_string(t.rows);
        s += ",\"assignments\":" + std::to_string(t.assignments) + ",\"u_sum\":" + std::to_string(t.u_sum);
        if (layer_form) s += ",\"pad16\":" + std::to_string(t.pad16) + ",\"pad64\":" + std::to_string(t.pad64);
        return s;
    }
    std::string stats_json() const {
        std::string l, w;
        for (const auto& [layer, t] : stats_.layers)
            l += (l.empty() ? "" : ",") + std::string("{\"layer\":") + std::to_string(layer) + "," + sums_json(t, true) + "}";
        for (const auto& [rows, t] : stats_.widths)
            w += (w.empty() ? "" : ",") + std::string("{\"rows\":") + std::to_string(rows) + "," + sums_json(t, false) + "}";
        return "{\"layers\":[" + l + "],\"widths\":[" + w + "],\"other\":{" + sums_json(stats_.other, false) + "}}";
    }

    struct Flips { std::uint64_t slots = 0, ids = 0, rows_changed = 0; };

    static std::string quote(const std::string& v) {
        std::string q = "\"";
        for (const char ch : v) {
            if (ch == '"' || ch == '\\') q += '\\';
            if (static_cast<unsigned char>(ch) < 0x20U) continue;
            q += ch;
        }
        return q + "\"";
    }

    static std::uint64_t count_records(const std::string& path) {
        // seekg() may succeed beyond EOF: use the actual file length to reject truncated bodies and partial headers
        // before a replay starts. Otherwise the receipt can claim a route file is complete when it is corrupt.
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("route tap: cannot inspect " + path);
        const auto end = f.tellg();
        if (end < static_cast<std::streamoff>(16)) throw std::runtime_error("route tap: short header in " + path);
        f.seekg(16, std::ios::beg);
        std::uint64_t n = 0;
        RouteRecordHeader h{};
        while (f.tellg() < end) {
            const auto left = end - f.tellg();
            if (left < static_cast<std::streamoff>(sizeof(h)))
                throw std::runtime_error("route tap: partial record header in " + path);
            f.read(reinterpret_cast<char*>(&h), sizeof(h));
            if (!f || h.rows == 0 || h.top_k == 0 || h.top_k > kRouteMaxTopK || h.flags != 0)
                throw std::runtime_error("route tap: corrupt record in " + path);
            const auto body = std::uint64_t{h.rows} * h.top_k * 8U;
            const auto remaining = end - f.tellg();
            if (remaining < 0 || body > static_cast<std::uint64_t>(remaining))
                throw std::runtime_error("route tap: truncated record body in " + path);
            f.seekg(static_cast<std::streamoff>(body), std::ios::cur);
            if (!f) throw std::runtime_error("route tap: corrupt record offset in " + path);
            ++n;
        }
        return n;
    }

    std::string out_pattern_, in_pattern_;  // as configured ("{seq}" unresolved)
    RouteTapConfig config_;                  // the current segment's resolved paths
    std::uint64_t seq_ = 0;
    std::mutex mutex_;
    std::ofstream out_;
    std::ifstream in_;
    std::uint64_t out_records_ = 0, out_rows_ = 0, in_total_ = 0, in_consumed_ = 0, in_rows_ = 0, nudged_ = 0, null_index_ = 0;
    std::map<std::uint32_t, Flips> flips_;
    std::vector<std::uint32_t> replay_ids_;
    std::vector<float> replay_weights_;
    RouteStats stats_;
    std::vector<std::uint32_t> stats_scratch_;
};

namespace detail {
struct RouteTapProcess {
    RouteTap tap{route_tap_config_from_env()};
    RouteTapProcess() = default;
    RouteTapProcess(const RouteTapProcess&) = delete;
    RouteTapProcess& operator=(const RouteTapProcess&) = delete;
    ~RouteTapProcess() {
        if (tap.active()) print_receipt(tap.receipt());
    }
    static void print_receipt(const std::string& receipt) { std::fprintf(stderr, "strix-route-file/v1 %s\n", receipt.c_str()); }
};
}  // namespace detail

inline RouteTap& RouteTap::instance() {
    static detail::RouteTapProcess process;
    return process.tap;
}

// Engine hook: prompt `seq` of a prompt list starts (see "Sequence segments" above). No-op unless the process tap is
// active and segmented.
inline void route_tap_sequence(std::uint64_t seq) {
    RouteTap& tap = RouteTap::instance();
    tap.sequence(seq, [](const std::string& receipt) { detail::RouteTapProcess::print_receipt(receipt); });
}

}  // namespace strix::moe
