#pragma once
// Prompt windows of a qwen35 dense CLI (pure host planning, no device types). One plan for every entry point:
//   - engine options: the engine's own defaults (EXACT wide windows, HipEngineOptions::prefill_rows / wide_exact)
//     unless a width is given explicitly (--wide / --wide-exact / --prefill-rows [--prefill-exact]);
//   - windows: chunked by the width the engine RESOLVED (HipEngine::prefill_rows(): a requested EXACT wide width
//     falls back to 16 when a large projection is not IU4), never by the requested one;
//   - logits: only the prompt's last window asks for them (LogitRows::last); intermediate windows ask none (no LM head,
//     no host read). Diagnostics keep their rows: --dump-logits (row path: every row of every window; wide path with
//     --dump-window: every row of the last wide window), --tf-rows N (the last N prompt tokens as <= max_rows windows on
//     the decode row path, logits of every row when dumping), --rows N (row-path windows of N rows).
// The rows a window computes do not depend on whether logits are asked (logits are a read of the final hidden rows), and
// the engine's windows of any width are bitwise equal to row-by-row decode (hip_engine.hpp), so the plan changes the
// work, never the committed state.
// Home: shared (host shared, qwen35 family: HIP decode + HIP spec CLIs bound; the CUDA SpecEngine::prefill callers chunk
// the same way and can bind it, no backend type here). Host test: tests/qwen35_prefill_windows_tests.cpp.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "strix/qwen35/engine_types.hpp"

namespace strix::qwen35 {

// What the command line asked for (0 / false = not given).
struct PrefillFlags {
    std::uint32_t rows = 0;    // --rows N: row-path windows of N rows (1..max_rows)
    std::uint32_t width = 0;   // --wide N / --wide-exact N / --prefill-rows N: explicit forward_commit width
    bool exact = false;        // with width: EXACT class (--wide-exact / --prefill-exact); else FAST above 16 rows
    bool dump = false;         // --dump-logits: logits of every prompt row the path computes
    bool dump_window = false;  // wide path + dump: every row of the last wide window (else its last row)
    std::uint32_t tf_rows = 0; // wide path: the last N prompt tokens on the decode row path

    [[nodiscard]] bool row_path() const noexcept { return !width && (rows || dump); }
};

// Engine options from the flags: untouched (engine defaults) unless an explicit width is given; the row path needs no
// wide scratch (16-row windows).
template<class Options>
void apply_prefill_flags(const PrefillFlags& f, std::uint32_t max_rows, Options& opt) {
    if (f.width) {
        opt.prefill_rows = f.width;
        opt.wide_exact = f.exact;
    } else if (f.row_path()) {
        opt.prefill_rows = max_rows;
    }
}

struct PrefillWindow {
    std::size_t at = 0;                    // first prompt position
    std::uint32_t rows = 0;                // window rows
    LogitRows want = LogitRows::none;      // logits asked of forward_commit
    [[nodiscard]] std::uint32_t tops() const noexcept {  // argmax rows to read back (0 = no host read)
        return want == LogitRows::all ? rows : want == LogitRows::last ? 1U : 0U;
    }
};

struct PrefillPlan {
    bool row_path = false;         // --rows / --dump-logits without a width: decode-row windows
    std::uint32_t requested = 0;   // the width the options asked (0 = engine default)
    std::uint32_t width = 0;       // the width the windows use (resolved / row width)
    std::vector<PrefillWindow> windows;
};

// `resolved` = the engine's prefill_rows() after load; `max_rows` = its decode window limit (16).
[[nodiscard]] inline PrefillPlan plan_prefill(const PrefillFlags& f, std::size_t prompt, std::uint32_t resolved,
                                              std::uint32_t max_rows) {
    if (prompt == 0U) throw std::invalid_argument("prefill: empty prompt");
    if (max_rows == 0U || resolved == 0U) throw std::invalid_argument("prefill: window widths must be positive");
    if (f.rows > max_rows) throw std::invalid_argument("prefill: --rows exceeds the decode window limit");
    PrefillPlan p;
    p.requested = f.width;
    p.row_path = f.row_path();
    const auto push = [&](std::size_t at, std::size_t end, std::uint32_t w, bool tail) {
        while (at < end) {
            const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(w, end - at));
            const bool last = at + n == prompt;
            LogitRows want = LogitRows::none;
            if (f.dump && (p.row_path || tail)) want = LogitRows::all;
            else if (last) want = f.dump && f.dump_window ? LogitRows::all : LogitRows::last;
            p.windows.push_back({at, n, want});
            at += n;
        }
    };
    if (p.row_path) {
        p.width = f.rows ? f.rows : 1U;  // --dump-logits alone keeps the one-row windows of its references
        push(0U, prompt, p.width, false);
        return p;
    }
    p.width = resolved;
    const std::size_t wide_end = prompt - std::min<std::size_t>(f.tf_rows, prompt - 1U);
    push(0U, wide_end, p.width, false);
    push(wide_end, prompt, max_rows, true);
    return p;
}

}  // namespace strix::qwen35
