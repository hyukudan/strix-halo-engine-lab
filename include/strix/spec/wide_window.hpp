#pragma once
// Wide verify windows for one lone request (backend-neutral pricing, host only).
//
// A wide window verifies a copied span of rows - 1 proposals plus the pending token in one forward of `rows` rows,
// where the narrow path verifies the same tokens in windows of `narrow_rows`. Every backend measures its own verify
// walls (one request, rows -> ms) and states what a rejected row costs it (keep_stride):
//   1  the accepted prefix is committed row by row (a rollback window),
//   s  the accepted prefix is committed down to a multiple of s rows (state checkpoints every s rows),
//   0  all-or-nothing: a rejected row discards the whole pass, the narrow path then emits the same tokens.
// Tokens never change with the decision (every committed row is a verified target row); only the cost does.
//
// Arithmetic (narrow cost per committed token c = narrow_ms / narrow_rows, a fully accepted narrow window commits
// narrow_rows tokens): a fully accepted wide window saves rows * c - ms; a miss after j accepted proposals keeps
// kept(j) rows (saving kept(j) * c) and pays ms. With the miss position uniform over j = 0..rows - 2, the break-even
// probability that the whole span is accepted is p* = (ms - K c) / (rows c - K c), K = mean kept rows of a miss.
// A size is admitted when a fully accepted window beats the narrow one by min_gain and p* is at most the span
// acceptance the backend measured for its copy proposals (full_accept).
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strix::spec {

struct WideWindowCost {
    std::uint32_t rows = 0;  // > narrow_rows
    double ms = 0.0;         // measured verify wall of one lone request
};

struct WideWindowTable {
    std::uint32_t narrow_rows = 8;
    double narrow_ms = 0.0;               // measured wall of the narrow window
    std::vector<WideWindowCost> wide;     // descending rows
    std::uint32_t keep_stride = 0;        // 0 all-or-nothing, 1 per row, s: multiples of s
    double full_accept = 0.0;             // measured share of copy proposals accepted to the last row
    double min_gain = 0.0;                // fully accepted wide tokens/ms over narrow tokens/ms - 1
};

// Rows a miss after `accepted` accepted proposals keeps.
[[nodiscard]] inline std::uint32_t wide_window_kept(std::uint32_t accepted, std::uint32_t keep_stride) noexcept {
    if (keep_stride == 0U) return 0U;
    return accepted / keep_stride * keep_stride;
}

// Mean kept rows of a miss, the miss position uniform over 0..rows - 2 accepted proposals.
[[nodiscard]] inline double wide_window_mean_kept(std::uint32_t rows, std::uint32_t keep_stride) noexcept {
    if (rows < 2U) return 0.0;
    double sum = 0.0;
    for (std::uint32_t j = 0; j + 1U < rows; ++j) sum += wide_window_kept(j, keep_stride);
    return sum / static_cast<double>(rows - 1U);
}

// A fully accepted wide window commits more tokens per ms than a fully accepted narrow one, by min_gain:
// rows / ms > (1 + min_gain) * narrow_rows / narrow_ms (compared without division).
[[nodiscard]] inline bool wide_window_full_pays(std::uint32_t rows, double ms, std::uint32_t narrow_rows, double narrow_ms,
    double min_gain = 0.0) noexcept {
    if (rows <= narrow_rows || !(ms > 0.0) || !(narrow_ms > 0.0)) return false;
    return static_cast<double>(rows) * narrow_ms > (1.0 + min_gain) * static_cast<double>(narrow_rows) * ms;
}

// Break-even full-span acceptance p* (above 1: never pays; 0 or below: pays even when every span misses).
[[nodiscard]] inline double wide_window_break_even(const WideWindowCost& w, const WideWindowTable& t) noexcept {
    if (w.rows <= t.narrow_rows || !(t.narrow_ms > 0.0) || !(w.ms > 0.0)) return 2.0;
    const double c = t.narrow_ms / static_cast<double>(t.narrow_rows);
    const double kept = wide_window_mean_kept(w.rows, t.keep_stride) * c;
    const double full = static_cast<double>(w.rows) * c;
    return full > kept ? (w.ms - kept) / (full - kept) : 2.0;
}

// The admitted sizes (descending), with one reason line per listed size.
[[nodiscard]] inline std::vector<std::uint32_t> wide_window_sizes(const WideWindowTable& t, std::string* reason = nullptr) {
    std::vector<std::uint32_t> out;
    std::string why;
    for (const auto& w : t.wide) {
        const double p = wide_window_break_even(w, t);
        const bool full = wide_window_full_pays(w.rows, w.ms, t.narrow_rows, t.narrow_ms, t.min_gain);
        const bool on = full && p <= t.full_accept;
        if (on) out.push_back(w.rows);
        if (reason) {
            char line[512];
            std::snprintf(line, sizeof line, "%s%u rows %.1f ms vs %u rows %.1f ms: break-even span acceptance %.2f %s measured %.2f (keep %u) -> %s",
                why.empty() ? "" : "; ", w.rows, w.ms, t.narrow_rows, t.narrow_ms, p, p <= t.full_accept ? "<=" : ">",
                t.full_accept, t.keep_stride, on ? "on" : full ? "off (misses cost more than full spans save)" : "off (slower when fully accepted)");
            why += line;
        }
    }
    if (reason) *reason = why.empty() ? "no wide size measured" : why;
    return out;
}

}  // namespace strix::spec
