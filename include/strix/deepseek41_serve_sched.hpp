#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Decode-priority scheduling of the two-rank serve (strix-deepseek41-serve, --prefill-path device), STRIX_DS41_SERVE_SCHED:
//
//   off (unset, default): an admitted prompt's device prefill runs at once and the decoding requests wait for all of it.
//   defer: deferred admission. A request whose device prefill would stall decoding requests waits in an admission
//     queue (no slot, no KV pool room) and is admitted when no request is decoding, or when it has waited
//     STRIX_DS41_SERVE_SCHED_STARVE_MS (default 30000), or when the cost rule says so: admitting now costs the others
//     N_decoding x T_dev(P) (each one stalls for the prefill), waiting costs this request W (the time until the decoding
//     requests end, estimated as the largest remaining max_tokens x the recent step time); admit now when
//     N x T_dev(P) <= W (J = own TTFT + N x delay is then the lower). T_dev(P) = a + b P seconds
//     (STRIX_DS41_SERVE_SCHED_TDEV="a,b", default the measured 0.17 s + 1.69 ms/token). A state imported for the request
//     while it waits (POST /admin/import-state, e.g. the prompt prefilled on another device) is taken at once and the
//     local prefill never runs.
//   interleave: the prefill runs as a PrefillTask in bounded units between decode steps: every scheduler tick runs one
//     batch step of the decoding requests, then one unit of the pending prefill, until it completes; then the request
//     joins the batch. With no request decoding, units run back to back at full size (the prefill's default pieces).
//     The unit today is one DevicePrefill run of at most STRIX_DS41_SERVE_SCHED_UNIT_ROWS rows (default 256, a multiple
//     of 8; extend() gives the bits of one run of the whole prompt); a resumable DevicePrefill plugs a finer unit in
//     through PrefillTask::step's budget: STRIX_DS41_SERVE_SCHED_UNIT_MS (rank 0 picks the number of resumable units
//     predicted to fit, DevicePrefill PrefillTask::units_within) or STRIX_DS41_SERVE_SCHED_UNIT_BUDGET (fixed units per
//     tick); then the pieces are the default ones and a unit is a layer part (STRIX_DS41_SERVE_SCHED_MOE_CHUNK rows per
//     MoE chunk, default 2048).
//     Prompts shorter than STRIX_DS41_SERVE_SCHED_MIN_ROWS (default 64) take the direct path (whole prefill at once).
//     Prompts resumed from a state imported while the task is pending abort the task and take the state.
//
// Every decision is rank 0's; rank 1 runs what the control channel tells it (the same commands in the same order, so
// the collectives of the two ranks stay in the same order). Pure functions here (unit tests: deepseek41_serve_sched_tests).
namespace strix::deepseek41::sched {

enum class Mode : std::uint32_t { off = 0, defer = 1, interleave = 2 };

struct Options {
    Mode mode = Mode::off;
    double starve_seconds = 30.0;
    double tdev_a = 0.17, tdev_b = 0.00169;  // T_dev(P) = a + b P seconds (measured on the two-node pair)
    std::uint32_t unit_rows = 256;           // interleave: rows of one unit while requests decode
    // interleave: prompts shorter than this run their whole prefill at once (no task). On the pair T_dev(64) ~ 0.28 s:
    // one such pause is close to a few ticks of the interleave, while a task would add one decode step per unit to its
    // TTFT (measured: 22-26-token prompts 0.18 s at once vs 0.6-0.78 s as 10-unit tasks).
    std::uint32_t min_rows = 64;
    std::uint32_t unit_budget = 0;           // interleave: PrefillTask::step budget per unit (0: the whole piece)
    // interleave with the resumable DevicePrefill (phase B): the units of a tick fit in unit_ms (rank 0's prediction,
    // DevicePrefill PrefillTask::units_within), else unit_budget units per tick; pieces stay the default ones.
    double unit_ms = 0.0;
    // the resumable task's MoE chunk (bounds a MoE unit; same on both ranks; 0: the prefill's own 4096). Smaller chunks
    // run on row views of the prefill's own MoE scratch (no extra memory); 2048 = DevicePrefill's pair-gated task default
    std::uint32_t moe_chunk_rows = 2048;
    [[nodiscard]] bool resumable() const noexcept { return unit_ms > 0.0 || unit_budget > 0U; }
    bool log = false;                        // one stderr line per decision
};

// The mode of `env` (STRIX_DS41_SERVE_SCHED's value) for a serve whose prefill path is the device (true) or rows:
// unset = interleave with the device prefill (default since P17.2), off with rows; defer / interleave need the device
// prefill (rejected for rows, before anything connects or loads).
[[nodiscard]] inline Mode mode_for(const std::string& m, bool device_prefill) {
    if (m == "0" || m == "off") return Mode::off;
    if (m.empty()) return device_prefill ? Mode::interleave : Mode::off;
    Mode mode;
    if (m == "defer") mode = Mode::defer;
    else if (m == "interleave" || m == "1") mode = Mode::interleave;
    else throw std::invalid_argument("STRIX_DS41_SERVE_SCHED is off, defer or interleave, not '" + m + "'");
    if (!device_prefill) throw std::invalid_argument("STRIX_DS41_SERVE_SCHED=" + m + " needs --prefill-path device");
    return mode;
}

inline Options options_from_env(bool device_prefill) {
    Options o;
    const auto env = [](const char* k) -> std::string {
        const char* v = std::getenv(k);
        return v ? std::string(v) : std::string();
    };
    // default interleave with the device prefill since P17.2 (decode stalls <= ~1 s while a long prompt prefills)
    o.mode = mode_for(env("STRIX_DS41_SERVE_SCHED"), device_prefill);
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_STARVE_MS"); !s.empty()) o.starve_seconds = std::stod(s) / 1e3;
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_TDEV"); !s.empty()) {
        const auto c = s.find(',');
        if (c == std::string::npos) throw std::invalid_argument("STRIX_DS41_SERVE_SCHED_TDEV is \"a,b\" (seconds, seconds per token)");
        o.tdev_a = std::stod(s.substr(0, c));
        o.tdev_b = std::stod(s.substr(c + 1));
    }
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_UNIT_ROWS"); !s.empty()) o.unit_rows = static_cast<std::uint32_t>(std::stoul(s));
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_MIN_ROWS"); !s.empty()) o.min_rows = static_cast<std::uint32_t>(std::stoul(s));
    if (o.unit_rows < 16U || o.unit_rows % 8U != 0U) throw std::invalid_argument("STRIX_DS41_SERVE_SCHED_UNIT_ROWS: a multiple of 8, at least 16");
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_UNIT_BUDGET"); !s.empty()) o.unit_budget = static_cast<std::uint32_t>(std::stoul(s));
    if (const auto v = env("STRIX_DS41_SERVE_SCHED_UNIT_MS"); !v.empty()) o.unit_ms = std::stod(v);
    if (const auto v = env("STRIX_DS41_SERVE_SCHED_MOE_CHUNK"); !v.empty()) o.moe_chunk_rows = static_cast<std::uint32_t>(std::stoul(v));
    o.log = o.mode != Mode::off && env("STRIX_DS41_SERVE_SCHED_LOG") != "0";
    return o;
}

// The numeric identity and the collective protocol both serve ranks must share, compared in the hello before READY:
// the weights' content identity (prefix_namespace_from_env: overlay, CB4, trellis packs / mix / sites, SHEXP_TP, EP map,
// the decode identity EP / expert TP) and the DSpark configuration (drafter, its ranks, window policy, vocabulary).
[[nodiscard]] inline std::string serve_identity_text(const std::string& namespace_text, std::uint64_t dspark_bytes, std::uint32_t dspark_ranks,
                                                     const std::string& dspark_window, const std::string& dspark_vocab) {
    std::string t = namespace_text + "dspark=";
    if (dspark_bytes == 0U) return t + "-;";
    return t + std::to_string(dspark_bytes) + ":ranks=" + std::to_string(dspark_ranks) + ":window=" + dspark_window + ":vocab=" + dspark_vocab + ";";
}

[[nodiscard]] inline const char* mode_name(Mode m) {
    return m == Mode::defer ? "defer" : m == Mode::interleave ? "interleave" : "off";
}

[[nodiscard]] inline double tdev_seconds(const Options& o, std::uint64_t prompt_tokens) {
    return o.tdev_a + o.tdev_b * static_cast<double>(prompt_tokens);
}

enum class Admit : std::uint32_t { defer = 0, idle = 1, starved = 2, cost = 3, draining = 4 };
[[nodiscard]] inline const char* admit_name(Admit a) {
    switch (a) {
        case Admit::idle: return "idle";
        case Admit::starved: return "starved";
        case Admit::cost: return "cost";
        case Admit::draining: return "draining";
        default: return "defer";
    }
}

// Deferred admission of a request whose device prefill would stall `decoding` requests. waited: seconds since it was
// taken from the queue; wait_estimate: seconds until no request decodes (W).
[[nodiscard]] inline Admit admit(const Options& o, std::uint64_t prompt_tokens, std::uint32_t decoding, double waited, double wait_estimate,
                                 bool draining = false) {
    if (draining) return Admit::draining;
    if (decoding == 0U) return Admit::idle;
    if (waited >= o.starve_seconds) return Admit::starved;
    if (static_cast<double>(decoding) * tdev_seconds(o, prompt_tokens) <= wait_estimate) return Admit::cost;
    return Admit::defer;
}

// W: the time until no request decodes, from each decoding request's remaining tokens (max_tokens - generated; an upper
// bound, a stop token ends it earlier) and the recent seconds per batch step.
[[nodiscard]] inline double wait_estimate(std::span<const std::uint64_t> remaining_tokens, double step_seconds) {
    std::uint64_t most = 0;
    for (const auto r : remaining_tokens) most = std::max(most, r);
    return static_cast<double>(most) * step_seconds;
}

// interleave: the end of the next unit of a prefill at row `at` (rows so far), given the default piece ends of the
// prompt (plan_prefill_pieces: they stay piece ends, so the prefix cache inserts at the same boundaries) and a row
// budget (0: up to the next default end). A unit ends on a multiple of 8 (extend() continues from there) and leaves
// at least min_rows rows to the next default end; its own rows are at least min_rows too.
[[nodiscard]] inline std::uint32_t next_unit_end(std::uint32_t at, std::span<const std::uint32_t> default_ends, std::uint32_t budget_rows,
                                                 std::uint32_t min_rows) {
    const auto it = std::upper_bound(default_ends.begin(), default_ends.end(), at);
    if (it == default_ends.end()) throw std::invalid_argument("next_unit_end: the prefill is complete");
    const std::uint32_t d = *it;
    if (budget_rows == 0U || d - at <= budget_rows) return d;
    std::uint32_t end = at + budget_rows / 8U * 8U;
    if (d - end < min_rows) end = d >= min_rows ? (d - min_rows) / 8U * 8U : at;
    if (end <= at || end - at < min_rows) return d;
    return end;
}

// A request's prompt prefill as a sequence of bounded units (both ranks hold one, created by the same command and
// stepped by the same commands, so its collectives run in the same order on both ranks). The default implementation
// is the serve's (SlotPrefill::Task): one unit = one DevicePrefill run of [at, end) rows, end chosen by rank 0
// (next_unit_end) and sent to rank 1; `budget` is passed through for a resumable DevicePrefill (0: the whole piece).
// Contract for an implementation:
//  - step() returns with no collective of the prefill in flight (a decode step's collectives follow it);
//  - the prefill's state, scratch and row buffers are its own (DevicePrefill's), never the BatchForward's slot or
//    workspace; the request's BatchForward slot is written only by finish() (BatchForward::import_state);
//  - the bits of every row and of the final state do not depend on how the prompt was cut into units.
class PrefillTask {
public:
    virtual ~PrefillTask() = default;
    // Advances the prefill to row `end` (a unit; resumable implementations may stop earlier inside the piece within
    // `budget` and return false with rows() unchanged); true when rows() == the prompt's length.
    virtual bool step(std::uint32_t end, std::uint32_t budget) = 0;
    [[nodiscard]] virtual std::uint32_t rows() const noexcept = 0;   // rows of the prompt done (at)
    [[nodiscard]] virtual std::uint32_t total() const noexcept = 0;  // the prompt's length
    [[nodiscard]] virtual bool done() const noexcept { return rows() == total(); }
    [[nodiscard]] virtual bool mid_piece() const noexcept { return false; }  // a resumable unit stopped inside a piece
    [[nodiscard]] virtual std::uint32_t piece_end() const noexcept { return rows(); }  // mid_piece: the end the piece runs to
};

}  // namespace strix::deepseek41::sched
