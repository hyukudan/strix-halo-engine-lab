#pragma once

#include "strix/mtp_context_draft.hpp"
#include "strix/mtp_selection_transaction.hpp"
#include "strix/serve/cycle_receipt.hpp"
#include "strix/serve/prefill_group.hpp"
#include "strix/serve/prompt_piece.hpp"
#include "strix/serve/row_budget.hpp"
#include "strix/serve/step_planner.hpp"
#include "strix/token_sampling.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Shared multi-request decoder: several requests decode in ONE forward per step.
//
// One weight read serves every row of a step, so at decode the shared bytes (weights) are paid once and each extra
// row only pays its own K/V and compute. The decoder is the backend-neutral contract that turns that into aggregate
// throughput: requests join and leave at step boundaries, prompts are read in slices between decode rounds
// (StepPlanner: decode first, short-first with aging, prompt debt), and every step's rows are priced by plan_rows:
// each running request holds its certain row (the pending token) and draft rows are bought only while their chance of
// yielding a token per measured millisecond beats the step's rate. Step cost comes from a least-squares fit of the
// observed steps (fixed + row * rows + lane * lanes), seeded by the engine's prior.
//
// A row's chance is the lane's kept histogram: per draft row index, the rows the target verified on this request and
// the ones it kept (row 0 also from offers the plan did not buy: the target's next token tells whether it was right),
// with the drafter's claim as the prior mean (accept_weight rows). While no request waits, the cohort drains and ends
// with its slowest lane, so plan_rows paces the lanes (RowLimits::drain): a lane that ends first anyway buys no drafts.
//
// Greedy requests verify shared context drafts (merged with the backend's device drafts when it has a drafter)
// against the target's greedy choice; a per-lane selection transaction (JSON, thinking budget) may cut the window. Sampled requests use a single
// row per lane per step. Every request's ids equal its solo run under the backend's batch invariance contract.
//
// Pure host code: no device types. Engines bind CohortBackend; the arithmetic lives here.
namespace strix::serve {

// One request's rows in a forward: `rows` consecutive rows of one lane (its pending token, then its drafts).
// reads_scores: whether anything may read the lane's final scores (read_final_logits, sample_final). Default true =
// every group is read, the contract before the flag existed. Only a caller that proves the lane has no score consumer
// (CohortDecoder: a greedy lane with no sampler and no logit-reading selection) passes false; a backend may then serve
// the forward with a pick form that writes no scores. A read of a group passed as false must still return the row's
// exact final logits (a backend recovers them); the flag is a cost hint, never a correctness input.
struct CohortGroup {
    std::uint32_t lane = 0;
    std::uint32_t rows = 1;
    bool reads_scores = true;
};

// One member of a grouped prompt pass: `tokens` appended to `lane` as prefill(lane, tokens, first) would.
struct CohortPrompt {
    std::uint32_t lane = 0;
    std::span<const std::uint32_t> tokens = {};
    bool first = true;
};

// Engine side of the contract. Lanes are independent sessions held by the backend.
// The greedy id and final logits of each row depend only on that lane's committed state and the row's inputs.
class CohortBackend {
public:
    virtual ~CohortBackend() = default;
    [[nodiscard]] virtual std::uint32_t lanes() const = 0;             // sessions the backend holds
    [[nodiscard]] virtual std::uint32_t rows_per_forward() const = 0;  // physical rows of one forward, all groups
    // Positive only when verify() can run a rectangle of consecutive rows
    // per lane in one physical pass and commit its window prefixes independently.
    [[nodiscard]] virtual std::uint32_t max_batch_verify_rows() const { return 0U; }
    [[nodiscard]] virtual std::uint32_t max_positions() const = 0;     // positions one lane can hold
    // Starts or continues the lane's prompt: appends `tokens` to the lane's committed state (no logits needed).
    // first = true resets the lane before the rows.
    virtual void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) = 0;
    // For profile policies based on the whole prompt even when serving yields between slices.
    virtual void prefill_scoped(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first,
                                std::uint64_t /*prompt_tokens*/) {
        prefill(lane, tokens, first);
    }
    // One forward over `groups` (tokens in group order). top[i] = greedy id after row i. The rows stay a pending
    // window until commit(). A greedy-only backend may already commit the
    // guaranteed row inside verify() when every group has exactly one row.
    virtual void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                        std::span<std::uint32_t> top) = 0;
    [[nodiscard]] virtual bool supports_final_logits() const { return false; }
    // Between verify and commit, reads the final row of a sampled lane from the pending window.
    // The returned span is valid until the next backend call.
    virtual std::span<const float> read_final_logits(std::uint32_t lane);
    // Device sampling: the sampled id of a lane is decided on the device (e.g. after an on-device vocabulary
    // exchange), not by the host sampler over read_final_logits. When true, wherever the decoder would call
    // sampler.decide(read_final_logits(lane), position, nullptr) it calls sample_final(lane, sampler, position)
    // instead (same points, same order). sample_final returns the id decide() would return for that row, drawing
    // sampler.uniforms(position, false).plain, and records the decision with sampler.count(decision) exactly once, as
    // decide() does, so the sampler's stats and receipt stay the host path's. Default false: never called.
    [[nodiscard]] virtual bool samples_on_device() const { return false; }
    virtual std::uint32_t sample_final(std::uint32_t lane, TokenSampler& sampler, std::uint64_t position);
    // Keeps accept[g] (1..rows) leading rows of group g of the last verify, discards the rest.
    virtual void commit(std::span<const std::uint32_t> accept) = 0;
    virtual void release(std::uint32_t lane) = 0;  // the lane is free

    // Optional hooks. Each default keeps the behaviour of a backend that does not bind it.

    // Device drafter (MTP head, EAGLE, ...): the proposal the backend's last forward made for the lane's next window,
    // at most `room` draft rows after the pending token; probability[r] = the drafter's chance of row r given the rows
    // before it (empty or short = the decoder's prior). Asked for greedy lanes before each decode round. The decoder
    // merges it with its context drafter (MtpContextDraft::begin_cycle: a long exact copy span may replace it) and
    // plan_rows buys the rows; the outcome reaches the backend through verify / commit as for any draft.
    [[nodiscard]] virtual bool device_drafts() const { return false; }
    virtual void propose(std::uint32_t lane, std::uint32_t room, std::vector<std::uint32_t>& tokens,
                         std::vector<double>& probability);
    // Retained prefixes (the backend's own prefix store). Before a lane's first prompt rows: restores the longest
    // retained prefix of `prompt` into the lane and returns its rows, at most prompt.size() - 1 (0 = none: the lane then
    // starts with prefill(first = true)). The lane's state after restore(n) equals prefill(prompt[0, n), first = true).
    // When prefill_emits_first() is true it may return prompt.size(): the whole prompt is restored, no prefill runs, and
    // prefill_top / read_final_logits then serve the prompt's last row as if prefill had just read it.
    [[nodiscard]] virtual std::uint64_t restore(std::uint32_t lane, std::span<const std::uint32_t> prompt);
    // A finished request's committed tokens (its prompt, then its emitted ids but the last), before release(lane); the
    // backend may retain them for a later restore. Not called for refused or cancelled requests.
    virtual void capture(std::uint32_t lane, std::span<const std::uint32_t> committed);
    // Grouped prompt service: N short prompts that are ready together share ONE pass (serve::plan_prefill_group).
    // capacity: the backend's limits for a group pass (members < 2 = no grouping, the default).
    // route_class: the backend's route class of a prompt of `rows` run alone (R1: a group carries one class).
    // prefill_group: one pass over the members; each member's state and bits equal prefill(lane, tokens, first).
    [[nodiscard]] virtual PrefillGroupCapacity prompt_group_capacity() const;
    [[nodiscard]] virtual std::uint32_t prompt_route_class(std::uint64_t rows) const;
    virtual void prefill_group(std::span<const CohortPrompt> members);
    // First token from the prefill. Default false: the last prompt token is the lane's first decode row. True: the
    // lane's WHOLE prompt is read by prefill / prefill_group (the last slice ends with the last prompt token), and the
    // first generated id is decided from that prefill's last row: prefill_top(lane) = its greedy id, and
    // read_final_logits(lane) reads its final logits (sampled or logit-reading lanes), both valid until the next
    // backend call. For an engine whose prefill row is not bit-equal to a decode row of the same position, this keeps
    // the first id the prefill's own. A one-token prompt is then admissible.
    [[nodiscard]] virtual bool prefill_emits_first() const { return false; }
    // Rows beyond one prompt slice that a prefill call takes in the same pass (the final slice of a prompt absorbs a
    // remainder of up to this many rows instead of running a short extra slice). 0 = slices are exact.
    [[nodiscard]] virtual std::uint32_t prompt_tail_rows() const { return 0U; }
    // Bounded prompt slices while other lanes decode (CohortConfig::prompt_slice_decoding). prompt_piece_rule: where
    // one of the backend's prefill chunks may be cut without changing any row's bits (default: nowhere, bounded slices
    // refused). prompt_call_chunk: rows of the FIRST chunk the backend runs for a prefill call of `rows` rows (default:
    // the call is one chunk). The decoder cuts only inside the chunks the unbounded schedule runs, so the bits of a
    // prompt do not depend on whether other lanes were decoding.
    [[nodiscard]] virtual PromptPieceRule prompt_piece_rule() const { return {}; }
    [[nodiscard]] virtual std::uint64_t prompt_call_chunk(std::uint64_t rows) const { return rows; }
    [[nodiscard]] virtual std::uint32_t prefill_top(std::uint32_t lane);
    // The most rows (the pending token plus drafts) the lane's next verify group may carry; at least 1. Default:
    // unbounded (the decoder's own limits apply). For a device rule that holds a lane to fewer rows at some positions.
    [[nodiscard]] virtual std::uint32_t max_window(std::uint32_t lane) const;
    // Per-request engine data. At admission, before restore / prefill, the decoder calls bind_lane(lane, request id,
    // request.traits) once per request whose traits are non-empty (a request without traits makes no bind call, so a
    // follower rank that predates bind_lane serves it unchanged). traits is an engine-defined word list (route class, sampled flag, whole prompt
    // length, pool reservation, ...); it is the only per-request data a follower rank receives, so a backend whose
    // prefill depends on the request reads it from here. Default: no-op.
    virtual void bind_lane(std::uint32_t lane, std::uint64_t request_id, std::span<const std::uint32_t> traits);
    // An engine control message (sessions, drafter taps, imports): an engine-defined word list the engine's own owner
    // passes through CohortDecoder::control. Under ReplicatedCohort every rank receives the same message at the same
    // point between backend calls. Default: no-op.
    virtual void control(std::span<const std::uint32_t> words);
};

struct CohortRequest {
    std::uint64_t id = 0;
    // >= 2 tokens: the last one is the first decode row (>= 1 token when the backend's prefill emits the first id)
    std::vector<std::uint32_t> prompt;
    std::uint32_t max_tokens = 128;
    std::vector<std::uint32_t> stop;    // ids that end the request (emitted, then the request leaves)
    TokenSamplingPolicy sampling{};
    // The sampler's options for a sampled request (request identity, acceptance). An engine that replaces its own
    // job sampler passes the options that sampler would have had, so a seeded request keeps its draws.
    TokenSamplerOptions sampler_options{};
    std::function<bool()> cancelled = {};
    std::function<void(std::uint64_t)> prefilled = {}; // completed prompt rows before the anchor
    // Called after the backend commits, with the emitted prefix through stop.
    // False releases the lane at this boundary.
    std::function<bool(std::span<const std::uint32_t>)> publish = {};
    // Retained prefixes: reuse = may start from the backend's retained prefix (restore); retain = the finished
    // request's committed tokens are offered to the backend (capture).
    bool reuse = true, retain = true;
    // Output selection (JSON constraint, thinking budget): a shared selection transaction (null = plain greedy or
    // sampling). selection_reads_logits = false: the policy is decided from the rows' greedy ids
    // (select_greedy_window), the lane keeps its drafts. true: it reads the lane's final logits, so the lane runs one
    // row per step and the backend must read final logits. Greedy requests only. A policy refusal
    // (MtpSelectionError) ends the request with that error.
    std::shared_ptr<MtpSelectionTransaction> selection = {};
    bool selection_reads_logits = false;
    // (Appended last, so positional initialisers of the leading fields stay valid.)
    // The engine's context ceiling: the most ids the lane can emit before its context is full (0 = none: a request
    // with prompt + max_tokens > max_positions is refused). When set, the request runs to min(max_tokens,
    // context_tokens) ids and finishes "context" when the ceiling, not max_tokens, ended it (a tie is "length"). The
    // ceiling must fit the lane: prompt + context_tokens - 1 <= max_positions (the last id is never a row).
    std::uint32_t context_tokens = 0;
    std::vector<std::uint32_t> traits = {};  // engine-defined per-request words, passed to CohortBackend::bind_lane
};

struct CohortResult {
    std::uint64_t id = 0;
    std::vector<std::uint32_t> ids;     // generated ids
    std::string error;                  // non-empty: refused (never ran) or ended by a selection refusal
    std::uint64_t restored = 0;         // prompt rows restored from a retained prefix (not prefilled)
    bool first_from_prefill = false;    // the first id was decided from the prefill's last row
    double submit_ms = 0, first_ms = 0, done_ms = 0;  // decoder clock
    std::uint32_t steps = 0;            // decode steps the request ran in
    std::uint64_t drafted = 0, accepted = 0;
    bool cancelled = false;
    std::string_view sampling = "greedy";
    std::string sampling_receipt;
    // (Appended last, so positional initialisers of the leading fields stay valid.)
    // Why the request ended: "stop", "length", "context" (CohortRequest::context_tokens), "cancelled" or "error".
    std::string_view finish = "error";
};

struct CohortConfig;
// The draft rows a lane may carry while `live` lanes hold a request (CohortConfig::max_draft_by_live).
[[nodiscard]] std::uint32_t max_draft_for_live(const CohortConfig& config, std::uint32_t live) noexcept;

struct CohortConfig {
    static MtpContextDraftConfig default_draft() {
        MtpContextDraftConfig c;
        c.copy_edit = true;
        return c;
    }
    bool context_draft = true;
    MtpContextDraftConfig draft = default_draft();  // n-gram suffix drafts plus long copy spans (copy_edit)
    double min_p = 0.0;                 // the drafter's min_probability
    std::uint32_t max_draft = 7;        // draft rows per lane per step (window = 1 + drafts)
    // Draft rows per lane by live lanes (the engine's data): entry n - 1 caps max_draft while n lanes hold a request
    // (prompting or decoding); lanes beyond the table take its last entry; empty = max_draft at every width. A prompt
    // that arrives raises the live count at once, so a lane that drafts alone drops to the capped window at its next
    // round. Example {7, 0}: drafts only while one lane is live (a backend whose windows are their own forward).
    std::vector<std::uint32_t> max_draft_by_live{};
    CycleCost prior{};                  // the engine's measured prior: fixed + row * rows + lane * lanes (ms)
    bool learn_cost = true;             // replace the prior by the fit of the observed steps
    std::uint32_t fit_min_steps = 8;    // steps with at least two distinct row counts before the fit is used
    double accept_prior = 0.8;
    // Weight (verified rows) of the drafter's claim against the lane's kept histogram; <= 0 = the claim alone.
    double accept_weight = 2.0;
    // Admission by bytes: every request reserves prompt + max_tokens + 1 positions. Greedy also reserves max_draft.
    std::uint64_t memory_budget = 0, position_bytes = 0, lane_bytes = 0;
    // Prompt service (StepPlanner time slicing).
    std::uint32_t prompt_slice = 512;   // prompt rows per slice
    // Prompt rows per slice while another lane decodes (0 = prompt_slice). The slice then cuts the backend chunk the
    // unbounded schedule runs (CohortBackend::prompt_piece_rule / prompt_call_chunk; prompt_slice is the most rows one
    // backend call runs as one chunk), so a decoding lane waits about one such slice and the prompt keeps its bits.
    std::uint32_t prompt_slice_decoding = 0;
    // Gap-bounded pieces (0 = every piece is prompt_slice_decoding rows): while another lane decodes, a piece takes the
    // most rows whose predicted pass (prompt_pass_fixed_ms + the previous pass's observed ms per row) plus the last
    // decode round fits prompt_gap_ms (gap_bounded_rows, step_planner.hpp), at least prompt_slice_decoding, at most
    // prompt_slice. The previous pass prices the next one, so a piece follows the cost of deeper positions and of more
    // decoding lanes. A grant prices ONE backend pass: beside a decoding lane a call never runs two backend chunks (a
    // region of a chunk plus a remainder the backend runs on its own is issued chunk by chunk, a decode round between).
    // A piece never takes more rows than its grant when a bit-safe cut within the grant exists (prompt_piece_within,
    // the actual chunk, an absorbed tail included); a piece whose predicted pass still exceeds the target (the floor,
    // or the smallest bit-safe piece when no cut fits) is recorded (CohortDecoder::gap_overshoots). The row cost is
    // learned only from calls of one backend pass of at least the floor.
    double prompt_gap_ms = 0.0;
    double prompt_pass_fixed_ms = 0.0;  // the backend's fixed cost of one prefill pass (data: weights re-read per pass)
    double prompt_share = 1.0;          // ms of decode owed per ms of prompt while lanes decode
    std::uint64_t short_first_rows = 2048;
    // Grouped prompts: whole short prompts ready together share one backend pass when the backend's capacity admits
    // it. A group never carries more rows than one prompt slice, so it never widens the decode gap of a slice.
    PrefillGroupPolicy prompt_group{};
    // One stderr line per step (CohortDecoder::step): the prompt rows and wall of the step's prefill call, the decode
    // lanes and wall of its decode round, the scheduler's idle time before it, the prompts waiting and the decode
    // time owed (prompt_share). Diagnostic only: no schedule or bits change.
    bool trace_steps = false;
    // Use the backend's shared verify pass and price equal-width windows.
    bool uniform_verify_windows = false;
};

// Binds a mixed-service policy (prompt_piece.hpp) to a cohort: pieces of the policy's rows beside decoding lanes where
// the backend's rule cuts a `chunk`-row call, and, where it binds, the policy's prompt share. STRIX_SERVE_PROMPT_SHARE
// (a number >= 0) replaces the share (the off switch of a policy's share), STRIX_SERVE_STEP_TRACE=1 sets trace_steps.
void bind_mixed_service(CohortConfig& config, MixedServicePieces pieces, PromptPieceRule rule, std::uint64_t chunk);

// Rows of the next prompt slice of a lane with `left` prompt rows unread. `want` is the planner's grant, `unit` the
// configured slice. A full grant on the final slice absorbs a remainder of up to `tail` rows (the backend's
// CohortBackend::prompt_tail_rows) so the prompt runs as ONE pass instead of a pass plus a short, weight-bound pass.
[[nodiscard]] constexpr std::uint64_t prompt_slice_rows(std::uint64_t want, std::uint64_t unit, std::uint64_t left,
                                                        std::uint64_t tail) {
    if (want >= left) return left;
    return tail != 0U && want == unit && left - want <= tail ? left : want;
}

// The configuration checks of bounded prompt slices and gap-bounded pieces, shared by the CohortDecoder constructor
// and by a launcher that validates its options before any device work. `rule` = the backend's prompt_piece_rule(),
// `slice_first_chunk` = its prompt_call_chunk(config.prompt_slice). Throws std::invalid_argument.
void validate_prompt_pieces(const CohortConfig& config, PromptPieceRule rule, std::uint64_t slice_first_chunk);

class CohortDecoder {
public:
    CohortDecoder(CohortBackend& backend, CohortConfig config);
    ~CohortDecoder();
    CohortDecoder(const CohortDecoder&) = delete;
    CohortDecoder& operator=(const CohortDecoder&) = delete;

    void submit(CohortRequest request);       // queued; joins at the next step boundary
    // Passes an engine control message to the backend now, between steps (CohortBackend::control).
    void control(std::span<const std::uint32_t> words);
    [[nodiscard]] bool idle() const noexcept;  // nothing queued or running
    void step();                              // one scheduler step
    void run();                               // steps until idle
    std::vector<CohortResult> take_finished();

    [[nodiscard]] CycleCost cost() const;      // the cost the next plan uses (fit or prior)
    // The cost of a step of `lanes` decoding lanes from a set of observed steps. At a steady cohort size the lane count
    // barely moves and is collinear with the fixed term; the three-term fit can then give a negative lane cost, and
    // clamping it to 0 would leave lanes * |lane_ms| in fixed_ms (an inflated step cost makes every saved step look
    // dear, so plan_rows over-buys draft rows). A negative lane cost is folded into the fixed cost at `lanes`, where
    // the fit is exact; a fit that is not positive there falls back to the prior.
    [[nodiscard]] static CycleCost fitted_cost(const CycleLedger::FitSums& sums, const CohortConfig& config,
                                               std::uint32_t lanes);
    [[nodiscard]] std::uint32_t live() const noexcept;
    [[nodiscard]] std::uint32_t peak_live() const noexcept { return peak_live_; }
    [[nodiscard]] std::uint64_t steps() const noexcept { return steps_; }
    [[nodiscard]] std::uint64_t physical_rows() const noexcept { return rows_; }
    [[nodiscard]] double decode_ms() const noexcept { return decode_ms_; }
    [[nodiscard]] double prompt_ms() const noexcept { return prompt_ms_; }
    // Grouped prompt passes run and the prompts they served.
    [[nodiscard]] std::uint64_t prompt_groups() const noexcept { return prompt_groups_; }
    [[nodiscard]] std::uint64_t grouped_prompts() const noexcept { return grouped_prompts_; }
    // Gap-bounded pieces whose predicted pass + decode round exceeded prompt_gap_ms (the floor or an indivisible piece
    // was the only choice), and the largest predicted excess (ms).
    [[nodiscard]] std::uint64_t gap_overshoots() const noexcept { return gap_overshoots_; }
    [[nodiscard]] double gap_overshoot_ms() const noexcept { return gap_overshoot_ms_; }
    // Decode time (ms) still owed to decoding lanes for prompt time (CohortConfig::prompt_share).
    [[nodiscard]] double prompt_debt_ms() const noexcept { return planner_.debt_ms(); }
    // Test hook: a fixed clock step (ms) per call instead of the wall clock; 0 = wall clock.
    void set_fake_clock(double ms_per_read) noexcept { fake_clock_ms_ = ms_per_read; }
    // Test hook: advances the fake clock by `ms` (a backend's modelled pass cost); no effect on the wall clock.
    void advance_fake_clock(double ms) noexcept { fake_now_ += ms; }

private:
    struct Lane;
    struct VerificationScratch {
        std::vector<CohortGroup> groups;
        std::vector<std::uint32_t> tokens, top, keep;
    };
    struct Waiting {
        CohortRequest request;
        double submit_ms = 0;
        unsigned skips = 0;
        std::uint64_t order = 0;
    };
    double now_ms();
    void admit();
    void finish(Lane& lane, std::string error = {});
    // Rows of the lane's next prefill call under bounded slices (prompt_slice_decoding != 0): the unbounded call at a
    // region start when the grant covers it (beside a decoding lane, `bounded`: only its first backend chunk), else
    // one piece of the backend chunk being read (prompt_piece_rows). Under a priced grant (`priced`, gap pieces) the
    // piece is the largest bit-safe cut of the actual chunk within the grant (prompt_piece_within); `over` = none fit.
    [[nodiscard]] std::uint64_t piece_rows(Lane& lane, std::uint64_t want, std::uint32_t full_slice, bool bounded,
                                           bool priced, bool& over);
    [[nodiscard]] std::uint64_t reserved_bytes(const CohortRequest& r) const;
    void observe_cost(std::uint32_t rows, std::uint32_t lanes, double ms);

    CohortBackend& backend_;
    CohortConfig config_;
    StepPlanner planner_;
    std::deque<Waiting> waiting_;
    std::vector<std::unique_ptr<Lane>> lanes_;  // index = backend lane; null = free
    std::vector<CohortResult> finished_;
    VerificationScratch verification_;
    std::uint64_t order_ = 0, steps_ = 0, rows_ = 0, used_bytes_ = 0;
    std::uint64_t prompt_groups_ = 0, grouped_prompts_ = 0, gap_overshoots_ = 0;
    double gap_overshoot_ms_ = 0;
    std::uint32_t peak_live_ = 0;
    double decode_ms_ = 0, prompt_ms_ = 0, fake_clock_ms_ = 0, fake_now_ = 0;
    // Gap-bounded pieces: ms per row of the last prefill pass (its fixed cost removed) and of the last decode round.
    double last_row_ms_ = 0, last_decode_ms_ = 0;
    double last_step_end_ms_ = 0;  // trace_steps: the end of the previous step (scheduler idle before the next)
    CycleLedger::FitSums sums_{};  // observed decode steps: y = ms, r = physical rows, l = lanes (CycleLedger::fit)
};

}  // namespace strix::serve
