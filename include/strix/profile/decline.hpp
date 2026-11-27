#pragma once
// One record for every lever a run asked for and did not run (a decline). A lever is asked for by its tier column, an
// explicit setting or --all-on; it can be declined by the shared resolver (a ladder row that does not cover the
// regime, a pending column), at load (the weight image is not resident for the instance's extent, or the memory
// budget does not hold it), per request (a resident lever outside the request's measured regime), by a conflict with
// another lever, or by the backend. Every decline is written once in the same form to the startup stream and recorded
// here, so the receipt (--print-effective-config, the request's "strix" object, /v1/capabilities) states it:
//
//   strix-decline/v1 lever=NAME requested=VALUE effective=VALUE origin=ORIGIN regime="..." reason="..."
//
// A decline without a reason is a defect of the caller (record_decline throws): no lever is off in silence.
// Header only, so an engine library records declines without linking the resolver.
#include <cstdint>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::profile {

enum class DeclineOrigin : std::uint8_t { resolver, load, request, memory, conflict, backend };

[[nodiscard]] constexpr std::string_view decline_origin_name(DeclineOrigin origin) noexcept {
    switch (origin) {
        case DeclineOrigin::resolver: return "resolver";
        case DeclineOrigin::load: return "load";
        case DeclineOrigin::request: return "request";
        case DeclineOrigin::memory: return "memory";
        case DeclineOrigin::conflict: return "conflict";
        case DeclineOrigin::backend: return "backend";
    }
    return "resolver";
}

struct Decline {
    std::string lever;      // lever name (or the switch an engine admission gates)
    std::string requested;  // what the column / explicit setting / --all-on asked for
    std::string effective;  // what runs instead
    DeclineOrigin origin = DeclineOrigin::resolver;
    std::string regime;     // decline_regime() spelling of the regime the decision saw (empty: not regime-bound)
    std::string reason;     // why (never empty)
    bool operator==(const Decline&) const = default;
};

// "phase=decode context=16384 lanes=1 device=gfx1151": context 0 is "unknown", lanes 0 "auto" (left to the memory
// budget), an empty device is omitted.
[[nodiscard]] inline std::string decline_regime(std::string_view phase, std::uint64_t context, std::uint32_t lanes,
                                                std::string_view device = {}) {
    std::string out = "phase=" + std::string(phase);
    out += " context=" + (context == 0U ? std::string("unknown") : std::to_string(context));
    out += " lanes=" + (lanes == 0U ? std::string("auto") : std::to_string(lanes));
    if (!device.empty()) out += " device=" + std::string(device);
    return out;
}

namespace decline_detail {
inline std::string escaped(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        if (c == '\n' || c == '\r') { out += ' '; continue; }
        out += c;
    }
    return out;
}
inline std::string json_text(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (byte < 0x20U) {
            static constexpr char hex[] = "0123456789abcdef";
            out += "\\u00";
            out += hex[byte >> 4U];
            out += hex[byte & 15U];
        } else out += c;
    }
    return out + "\"";
}
inline std::string value_text(std::string_view value) { return value.empty() ? std::string("-") : std::string(value); }
}  // namespace decline_detail

[[nodiscard]] inline std::string decline_line(const Decline& decline) {
    using decline_detail::escaped;
    using decline_detail::value_text;
    return "strix-decline/v1 lever=" + value_text(decline.lever) + " requested=" + value_text(decline.requested) +
           " effective=" + value_text(decline.effective) + " origin=" +
           std::string(decline_origin_name(decline.origin)) + " regime=\"" + escaped(decline.regime) + "\" reason=\"" +
           escaped(decline.reason) + "\"";
}

[[nodiscard]] inline std::string decline_json(const Decline& decline) {
    using decline_detail::json_text;
    return "{\"lever\":" + json_text(decline.lever) + ",\"requested\":" + json_text(decline.requested) +
           ",\"effective\":" + json_text(decline.effective) + ",\"origin\":" +
           json_text(decline_origin_name(decline.origin)) + ",\"regime\":" + json_text(decline.regime) +
           ",\"reason\":" + json_text(decline.reason) + "}";
}

[[nodiscard]] inline std::string declines_json(const std::vector<Decline>& declines) {
    std::string out = "[";
    for (std::size_t i = 0; i < declines.size(); ++i) out += (i == 0U ? "" : ",") + decline_json(declines[i]);
    return out + "]";
}

// A decline without its lever or reason is the silent decline this record exists to prevent.
inline void require_recordable(const Decline& decline) {
    if (decline.lever.empty() || decline.reason.empty())
        throw std::invalid_argument("decline of '" + decline.lever + "' without " +
                                    (decline.lever.empty() ? "a lever" : "a reason") +
                                    " (strix-decline/v1 requires both)");
}

// The process's load-time declines (the instance's startup record; /v1/capabilities states them). Identical records
// collapse: an instance that probes the same decision twice declines once.
class DeclineLog {
public:
    void record(Decline decline) {
        require_recordable(decline);
        const std::lock_guard lock(mutex_);
        for (const auto& row : rows_)
            if (row == decline) return;
        rows_.push_back(std::move(decline));
    }
    [[nodiscard]] std::vector<Decline> snapshot() const {
        const std::lock_guard lock(mutex_);
        return rows_;
    }
    void clear() {
        const std::lock_guard lock(mutex_);
        rows_.clear();
    }

private:
    mutable std::mutex mutex_;
    std::vector<Decline> rows_;
};

[[nodiscard]] inline DeclineLog& process_declines() {
    static DeclineLog log;
    return log;
}

// Writes the decline's one line to `out` and records it in the process log (load / startup decisions). Request-time
// declines go to the request's own receipt instead (serve::GenerationStats::declined) and are written with
// decline_line() only.
inline void record_decline(Decline decline, std::ostream& out) {
    require_recordable(decline);
    out << decline_line(decline) << '\n';
    process_declines().record(std::move(decline));
}

}  // namespace strix::profile
