#pragma once

#include "strix/ds41_pipeline_wire.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// strix/ds41-pipeline-v1, speculative extension (DSpark across the cut): the TAIL runs the drafter and the window
// controller next to the target's last layers (the drafter reads the inputs of the last three layers of committed rows);
// the FRONT runs the verify windows' first layers and applies the TAIL's decisions. One request / response per window,
// no extra blocking acknowledgement:
//
//   FRONT                                         TAIL (rank 0)
//   RESET, prompt rows: STEP kStepFill ...  --->  every row committed, its hidden appended to the drafter; RESULT
//   SPEC_BEGIN {epoch, position, max_new, eos}  --->  vocabulary from the context, first draft + window choice
//                                           <---  SPEC_ACCEPT {window 0: nothing verified; plan of window 1}
//   CUDA step of the plan's R rows
//   STEP kStepVerify, request = (epoch, w)  --->  verify, accepted prefix a, kept = a + 1 (EOS / max_new cut it),
//                                                 commit locally, append the kept rows, next draft + choice
//                                           <---  SPEC_ACCEPT {window w: argmax, kept, next token, done; plan of w + 1}
//   commit_window(kept, next token) locally, publish the tokens, run the plan ...
//   SPEC_COMMIT {epoch, last window}         --->  (end of the sequence or a stop) ACK with the TAIL's position
//
// Commit watermark: a VERIFY of window w is sent only after the FRONT applied decision w - 1 (its rows are the plan of
// that decision, its position the committed position after it), so every VERIFY is the acknowledgement of the previous
// decision. Decisions are idempotent: the TAIL keeps the last decision's message and answers a VERIFY of the same
// (epoch, window) with it again without running anything (a FRONT that timed out resends the same STEP); an older window
// or another epoch is answered with status stale and changes nothing. The FRONT accepts exactly one decision per window
// (the one whose epoch and window it waits for) and drops duplicates. Fallback: a TAIL without the drafter answers
// SPEC_BEGIN with status off; the FRONT then decodes with plain one-row STEPs of the same numeric identity.
namespace strix::ds41::pipe {

// STEP payload flags (with kStepWantLogits = 1, kStepWantRowFnv = 2)
inline constexpr std::uint32_t kStepVerify = 4U;  // a speculative verify window (header.request = spec_request(epoch, window))
inline constexpr std::uint32_t kStepFill = 8U;    // context rows: every row committed by the TAIL, appended to the drafter

[[nodiscard]] constexpr std::uint64_t spec_request(std::uint32_t epoch, std::uint32_t window) noexcept {
    return (std::uint64_t{epoch} << 32U) | window;
}
[[nodiscard]] constexpr std::uint32_t spec_epoch(std::uint64_t request) noexcept { return static_cast<std::uint32_t>(request >> 32U); }
[[nodiscard]] constexpr std::uint32_t spec_window(std::uint64_t request) noexcept { return static_cast<std::uint32_t>(request); }

enum class SpecStatus : std::uint32_t { ok = 0, stale = 1, off = 2, refused = 3 };
enum class SpecDone : std::uint32_t { more = 0, eos = 1, max_new = 2 };

struct SpecBegin {
    std::uint32_t epoch = 0;
    std::uint32_t position = 0;      // the FRONT's committed position (== the TAIL's)
    std::uint32_t next_token = 0;    // the target's token after position - 1 (the first window's input)
    std::uint32_t max_new = 0;       // tokens to publish from here (next_token counts as the first)
    std::uint32_t flags = 0;         // kStepWantRowFnv: every decision carries its rows' full-logits FNV (checks)
    std::vector<std::uint32_t> eos;  // end-of-sequence ids (decoding stops after publishing one)
};

struct SpecAccept {
    std::uint32_t epoch = 0, window = 0;
    SpecStatus status = SpecStatus::ok;
    // the verified window (window > 0): rows at position .. position + rows - 1, the target's argmax of every row
    std::uint32_t position = 0, rows = 0, kept = 0;
    std::vector<std::uint32_t> argmax;
    std::vector<std::uint64_t> row_fnv;  // rows (0 when not requested)
    // after the decision: the committed position, the next token (input of the next window), what was published
    std::uint32_t committed = 0, next_token = 0;
    std::vector<std::uint32_t> published;  // argmax[0..kept) cut at EOS / max_new (window 0: [next_token])
    SpecDone done = SpecDone::more;
    // the plan of window + 1 (rows 1..8; empty when done): [next_token, draft_1, .., draft_k]
    std::vector<std::uint32_t> plan;
    float draft_ms = 0.0F, control_ms = 0.0F;  // TAIL-side times of this decision (diagnostics)
};

[[nodiscard]] std::vector<std::uint8_t> encode(const SpecBegin& m);
[[nodiscard]] std::vector<std::uint8_t> encode(const SpecAccept& m);
[[nodiscard]] SpecBegin decode_spec_begin(std::span<const std::uint8_t> p);
[[nodiscard]] SpecAccept decode_spec_accept(std::span<const std::uint8_t> p, std::uint32_t max_rows = 8U);
// SPEC_COMMIT payload: epoch, last window applied
[[nodiscard]] std::vector<std::uint8_t> encode_spec_commit(std::uint32_t epoch, std::uint32_t window);
[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> decode_spec_commit(std::span<const std::uint8_t> p);

// The verify rule of greedy speculative decoding: inputs [t, d_1..d_k], argmax g[0..k]: drafts are accepted while
// d_{j+1} == g[j]. Returns a (0..k).
[[nodiscard]] std::uint32_t accepted_prefix(std::span<const std::uint32_t> inputs, std::span<const std::uint32_t> argmax);

// What a verified window commits and publishes: kept = a + 1 rows, published = argmax[0..a]; cut after the first EOS
// in published (kept = its index + 1) or at `remaining` published tokens. remaining >= 1.
struct SpecOutcome {
    std::uint32_t accepted = 0, kept = 0;
    std::vector<std::uint32_t> published;
    SpecDone done = SpecDone::more;
};
[[nodiscard]] SpecOutcome spec_outcome(std::span<const std::uint32_t> inputs, std::span<const std::uint32_t> argmax, std::uint32_t remaining,
                                       std::span<const std::uint32_t> eos);

// TAIL side: the last decision of the session (idempotent answers).
class SpecDecisionLog {
public:
    enum class Kind { fresh, duplicate, stale };
    void begin(std::uint32_t epoch) {
        epoch_ = epoch;
        window_ = 0;
        last_.clear();
        active_ = true;
        finished_ = false;
    }
    void end() noexcept { active_ = false; }
    // The sequence is done (EOS / max_new): no further window, the last decision is still answered again.
    void finish() noexcept { finished_ = true; }
    [[nodiscard]] bool finished() const noexcept { return finished_; }
    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] std::uint32_t epoch() const noexcept { return epoch_; }
    [[nodiscard]] std::uint32_t window() const noexcept { return window_; }  // the last decided window
    // A VERIFY with this request: the next window (fresh), the last decided one again (duplicate), anything else (stale).
    [[nodiscard]] Kind classify(std::uint64_t request) const noexcept;
    void record(std::uint32_t window, std::vector<std::uint8_t> message) {
        window_ = window;
        last_ = std::move(message);
    }
    [[nodiscard]] const std::vector<std::uint8_t>& last() const noexcept { return last_; }

private:
    std::uint32_t epoch_ = 0, window_ = 0;
    std::vector<std::uint8_t> last_;
    bool active_ = false, finished_ = false;
};

// FRONT side: which decision it waits for; anything else is dropped (duplicates, other epochs).
struct SpecFrontState {
    std::uint32_t epoch = 0, window = 0;  // waiting for the decision of `window` (0: the BEGIN's answer)
    [[nodiscard]] bool expected(const SpecAccept& a) const noexcept { return a.epoch == epoch && a.window == window; }
};

// Window costs of a split decode, from a JSON table {"V": {"1": ms, ..}, "D": ms, "A": ms} (the hetero cost tables):
// verify[R] in seconds (0 = not in the table), draft and extra (append + control) in seconds.
struct SpecCostTable {
    std::vector<double> verify;  // [0..8]
    double draft = 0.0, extra = 0.0;
    std::string label;
};
[[nodiscard]] SpecCostTable read_spec_cost_table(const std::string& path);

}  // namespace strix::ds41::pipe
