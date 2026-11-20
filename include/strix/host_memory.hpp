#pragma once
// Host memory readings, the admission floor, the recovery wait and the pressure level: one home for every engine (host
// only, no device headers, no engine dependency).
//
//   floor      every load or pool admission leaves max(4 GiB, MemTotal / 32) of MemAvailable untouched.
//   pressure   PressureGovernor turns PSI averages into none / warning / critical (rise at once, fall after calm).
//   readings   one parser for the kernel's "Key:   N kB" counter files (/proc/meminfo, /proc/self/status) and one for
//              the memory pressure file (/proc/pressure/memory). Engines read through these instead of private parsers.
//   recovery   a process that is still exiting (the previous run of a back-to-back sequence) gives its memory back over
//              a few seconds; an admission taken in that window refuses a load that would fit a moment later.
//              await_memory() re-reads while the caller's own fit rule says "no" AND MemAvailable keeps rising, bounded
//              in time; it never waits on memory that is not coming back. The caller's admission then decides on the
//              last reading.
// Nothing here changes a system setting or needs a privilege; the wait is a pure function of its reads (host test).
#include "strix/env_flags.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>

namespace strix::host_memory {

inline constexpr std::uint64_t kGiB = 1ULL << 30;
inline constexpr std::uint64_t kMiB = 1ULL << 20;

// ---- Readings ----------------------------------------------------------------------------------------------------
// Every "Key: N [unit]" line -> bytes when the unit is kB, N otherwise (HugePages_Total, counts). Lines without a key,
// without a number or with a value that would overflow are skipped; the text is never trusted beyond that.
using KbFields = std::map<std::string, std::uint64_t>;

inline KbFields parse_kb_fields(std::string_view text) {
    KbFields out;
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t end = text.find('\n', at);
        if (end == std::string_view::npos) end = text.size();
        const std::string_view line = text.substr(at, end - at);
        at = end + 1U;
        const std::size_t colon = line.find(':');
        if (colon == std::string_view::npos || colon == 0U) continue;
        std::string_view rest = line.substr(colon + 1U);
        while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.remove_prefix(1U);
        std::uint64_t value = 0;
        std::size_t digits = 0;
        bool overflow = false;
        while (digits < rest.size() && rest[digits] >= '0' && rest[digits] <= '9') {
            const auto digit = static_cast<std::uint64_t>(rest[digits] - '0');
            if (value > (UINT64_MAX - digit) / 10U) { overflow = true; break; }
            value = value * 10U + digit;
            ++digits;
        }
        if (overflow || digits == 0U) continue;
        rest.remove_prefix(digits);
        while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.remove_prefix(1U);
        if (rest.substr(0, 2U) == "kB") {
            if (value > UINT64_MAX / 1024U) continue;
            value *= 1024U;
        }
        out[std::string(line.substr(0, colon))] = value;
    }
    return out;
}

inline std::string read_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) return {};
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

inline KbFields read_kb_fields(const std::string& path) { return parse_kb_fields(read_text(path)); }

struct Reading {
    std::uint64_t total = 0;      // MemTotal, bytes
    std::uint64_t free = 0;       // MemFree
    std::uint64_t available = 0;  // MemAvailable
    bool ok = false;              // MemFree and MemAvailable were both read
};

inline Reading reading_from(const KbFields& f) {
    Reading r;
    const auto get = [&](const char* k, std::uint64_t& v) {
        const auto it = f.find(k);
        if (it == f.end()) return false;
        v = it->second;
        return true;
    };
    get("MemTotal", r.total);
    const bool has_free = get("MemFree", r.free);
    const bool has_available = get("MemAvailable", r.available);
    r.ok = has_free && has_available;
    return r;
}

inline Reading read_meminfo(const std::string& path = "/proc/meminfo") { return reading_from(read_kb_fields(path)); }

// /proc/pressure/memory: "some avg10=0.00 avg60=0.12 avg300=0.50 total=2328134194" and the same for "full" (percent of
// wall time some / every non-idle task stalled on memory).
struct Psi {
    double some10 = 0, some60 = 0, full10 = 0, full60 = 0;
    bool ok = false;  // at least one "some" or "full" line parsed
};

inline Psi parse_psi(std::string_view text) {
    Psi p;
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t end = text.find('\n', at);
        if (end == std::string_view::npos) end = text.size();
        const std::string line(text.substr(at, end - at));
        at = end + 1U;
        std::istringstream is(line);
        std::string kind, kv;
        if (!(is >> kind) || (kind != "some" && kind != "full")) continue;
        double* avg10 = kind == "some" ? &p.some10 : &p.full10;
        double* avg60 = kind == "some" ? &p.some60 : &p.full60;
        while (is >> kv) {
            const std::size_t eq = kv.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = kv.substr(0, eq);
            const double v = std::strtod(kv.c_str() + eq + 1, nullptr);
            if (key == "avg10") { *avg10 = v; p.ok = true; }
            else if (key == "avg60") *avg60 = v;
        }
    }
    return p;
}

inline Psi read_psi(const std::string& path = "/proc/pressure/memory") { return parse_psi(read_text(path)); }

inline std::string gib_str(std::uint64_t bytes) {
    char b[32];
    std::snprintf(b, sizeof b, "%.2f", static_cast<double>(bytes) / static_cast<double>(kGiB));
    return b;
}

// ---- Floor and admission -------------------------------------------------------------------------------------------
// Floor kept free under every load or pool admission. On a unified-memory node (the GPU allocates host RAM) a process
// that drives MemAvailable to zero stalls the whole node in reclaim; the floor is max(4 GiB, MemTotal / 32), i.e.
// 4 GiB on a 128 GiB node. An unknown MemTotal (0) still keeps 4 GiB.
inline constexpr std::uint64_t floor_bytes(std::uint64_t mem_total) {
    return std::max<std::uint64_t>(4U * kGiB, mem_total / 32U);
}
// Bytes a new consumer may take: MemAvailable - floor, saturating at 0 (an unreadable reading gives 0).
inline std::uint64_t usable_above_floor(const Reading& r) {
    const std::uint64_t f = floor_bytes(r.total);
    return r.ok && r.available > f ? r.available - f : 0;
}

struct Admission {
    bool admitted = false;
    std::uint64_t need = 0, available = 0, floor = 0;
    std::string reason;
};

// need <= MemAvailable - floor, by subtraction (an oversized need cannot wrap into a fit). An unreadable meminfo admits
// nothing: the caller decides whether that is fatal.
inline Admission admit(std::uint64_t need, const Reading& r) {
    Admission a;
    a.need = need;
    a.available = r.available;
    a.floor = floor_bytes(r.total);
    if (!r.ok) { a.reason = "host memory unreadable (/proc/meminfo)"; return a; }
    a.admitted = need <= usable_above_floor(r);
    a.reason = std::string(a.admitted ? "fits: need " : "short: need ") + gib_str(need) + " GiB " +
               (a.admitted ? "<=" : ">") + " MemAvailable " + gib_str(r.available) + " GiB - floor " + gib_str(a.floor) +
               " GiB";
    return a;
}

// ---- Recovery wait -------------------------------------------------------------------------------------------------
struct WaitPolicy {
    double max_wait_s = 30;          // STRIX_HOST_MEMORY_WAIT_S (0 = never wait)
    double poll_s = 0.5;
    double stall_s = 3;              // give up when MemAvailable has not risen by min_rise for this long
    std::uint64_t min_rise = 256U * kMiB;
    static WaitPolicy from_env() {
        WaitPolicy p;
        p.max_wait_s = static_cast<double>(std::max(0, envcfg::integer("STRIX_HOST_MEMORY_WAIT_S", 30)));
        return p;
    }
};

struct WaitResult {
    bool fits = false;
    double waited_s = 0;
    int reads = 0;
    std::uint64_t first_available = 0, last_available = 0;
    std::string why;  // "fits", "no wait", "not recovering", "timeout"
};

// read() -> R (any snapshot), available(R) -> MemAvailable bytes, fits(R) -> bool, sleep(seconds) advances time.
// Re-reads while fits() is false and MemAvailable rose by >= min_rise within the last stall_s, up to max_wait_s. The
// last snapshot is left in `last` so the caller admits (and reports) on exactly what was decided.
template <class R, class Read, class Available, class Fits, class Sleep>
WaitResult await_memory(const WaitPolicy& p, Read&& read, Available&& available, Fits&& fits, Sleep&& sleep, R& last) {
    WaitResult w;
    last = read();
    w.reads = 1;
    w.first_available = w.last_available = available(last);
    if (fits(last)) { w.fits = true; w.why = "fits"; return w; }
    if (p.max_wait_s <= 0 || p.poll_s <= 0) { w.why = "no wait"; return w; }
    std::uint64_t mark = w.first_available;  // MemAvailable at the last counted rise
    double since_rise = 0;
    while (w.waited_s < p.max_wait_s) {
        sleep(p.poll_s);
        w.waited_s += p.poll_s;
        since_rise += p.poll_s;
        last = read();
        ++w.reads;
        w.last_available = available(last);
        if (fits(last)) { w.fits = true; w.why = "fits"; return w; }
        if (w.last_available >= mark + p.min_rise) { mark = w.last_available; since_rise = 0; }
        else if (since_rise >= p.stall_s) { w.why = "not recovering"; return w; }
    }
    w.why = "timeout";
    return w;
}

inline std::string describe(const WaitResult& w) {
    return "host-memory wait: " + w.why + " after " + std::to_string(w.reads) + " reads, " +
           std::to_string(static_cast<long>(w.waited_s * 1000.0)) + " ms, MemAvailable " + gib_str(w.first_available) +
           " -> " + gib_str(w.last_available) + " GiB";
}

// ---- Pressure governor ---------------------------------------------------------------------------------------------
// PSI averages -> none / warning / critical. A rise is taken at once; a fall only after calm_s of readings below the
// current level (then straight to the highest level seen during the calm spell). Pool owners stop growing at critical.
enum class Pressure { none, warning, critical };
inline const char* pressure_name(Pressure p) {
    return p == Pressure::critical ? "critical" : p == Pressure::warning ? "warning" : "none";
}

struct PressureThresholds {
    double warn_some10 = 10;  // some task stalled on memory 10 % of the last 10 s: reclaim is visible
    double warn_full10 = 2;   // every task stalled 2 % of the time: the node is reclaiming hard
    double crit_full10 = 25;  // thrash: the level the GLM load watchdog aborts at
    double calm_s = 30;
};

inline Pressure classify(const Psi& p, const PressureThresholds& t) {
    if (!p.ok) return Pressure::none;
    if (t.crit_full10 > 0 && p.full10 >= t.crit_full10) return Pressure::critical;
    if ((t.warn_full10 > 0 && p.full10 >= t.warn_full10) || (t.warn_some10 > 0 && p.some10 >= t.warn_some10))
        return Pressure::warning;
    return Pressure::none;
}

class PressureGovernor {
public:
    explicit PressureGovernor(PressureThresholds t = {}) : t_(t) {}
    Pressure update(const Psi& p, double now_s) {
        const Pressure seen = classify(p, t_);
        if (seen >= level_) {
            level_ = seen;
            calm_since_ = -1;
            calm_max_ = Pressure::none;
            return level_;
        }
        if (calm_since_ < 0) { calm_since_ = now_s; calm_max_ = seen; }
        calm_max_ = std::max(calm_max_, seen);
        if (now_s - calm_since_ >= t_.calm_s) {
            level_ = calm_max_;
            calm_since_ = -1;
            calm_max_ = Pressure::none;
        }
        return level_;
    }
    [[nodiscard]] Pressure level() const { return level_; }

private:
    PressureThresholds t_;
    Pressure level_ = Pressure::none;
    double calm_since_ = -1;
    Pressure calm_max_ = Pressure::none;
};

}  // namespace strix::host_memory
