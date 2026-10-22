#pragma once

#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_batch_schedule.hpp"
#include "strix/ds41_dspark_driver.hpp"
#include "strix/hip_ds41_dspark.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

// DSpark windows inside BatchForward, several requests per step (N > 1).
//
// Every decoding request of a step verifies its own window: its last token, then the drafts its own drafting context
// proposed, up to 6 rows (1 row = a plain decode row, no draft). The windows of all requests, and the prompt chunks
// of requests still reading their prompt, go in one BatchForward step (at most BatchForward::max_rows rows). Each
// request then keeps the rows whose drafts equalled the target's greedy token after the previous row, plus that
// correction: commit(slot, kept). So every request emits exactly its greedy tokens alone, whatever the windows were.
//
// The drafter runs on one rank (the planner, rank 0): one DeviceDrafter, one drafting context per slot (its rings,
// position and draft vocabulary; DeviceDrafter::select), fed by BatchForward's captured inputs of the last three
// layers for every kept row (prompt rows included). The plan (rows) and the commits are what the other rank needs:
// it steps the same rows and commits the same rows (the serving control channel already carries the rows).
//
// Which requests draft and how many rows each verifies: BatchWindowController (two stages, measured costs) or a fixed
// window (`fixed_rows`). Prompt rows come before drafts: decoding requests always get their own row, the prompt
// chunks the rows left (oldest first, as plan_step), the windows what remains.
namespace strix::deepseek41 {

struct BatchSpecOptions {
    std::string vocabulary = "full";  // ds41::dspark::choose_vocabulary spec, chosen per request from its prompt
    std::uint32_t fixed_rows = 0;     // 0: BatchWindowController; 2..6: every decoding request with room verifies that many
    std::uint32_t prompt_rows = 8;    // prompt rows per request per step (plan_step)
    std::uint32_t max_rows = BatchForward::max_rows;
    // a window never crosses a multiple of `align` positions (0: no limit): the prefix cache's session snapshots are
    // taken when a request's position lands on a block boundary
    std::uint32_t align = 0;
    // Drafts on both ranks (world 2): a request's drafting context lives on rank slot % 2, and every rank drafts its
    // own requests in parallel. Rank 0 plans; a draft round per step (512-byte exchanges through `exchange`, the same
    // size on both ranks) sends rank 1 the commits of the last step and the requests to draft, and brings their drafts
    // back. Rank 1 runs follow_round() before every step. world 1: rank 0 drafts every request.
    std::uint32_t rank = 0, world = 1;
    std::function<void(std::span<const std::byte>, std::span<std::byte>)> exchange;
    // world 2: a request start()ed from its prompt drafts on rank slot % 2 (both ranks call start with the prompt);
    // false: on rank 0 (the server's rank 1 does not see a prompt read in rows before it is read). imported()
    // requests always go to rank slot % 2.
    bool split_started = true;
};

struct BatchSpecStats {
    std::uint64_t steps = 0, drafted_steps = 0, windows = 0, plain_rows = 0, drafts_verified = 0, drafts_accepted = 0, rows = 0;
    std::uint64_t decode_steps = 0, decode_tokens = 0;  // steps without prompt rows and the tokens they emitted
    double draft_seconds = 0.0, step_seconds = 0.0, finish_seconds = 0.0, plan_seconds = 0.0;
    double decode_seconds = 0.0;  // wall of the decode-only steps (plan + drafts + step + finish)
    std::array<std::uint64_t, ds41::dspark::block_size + 2U> window_rows{};  // windows by rows (1 = plain row)
    std::array<std::uint64_t, ds41::dspark::block_size> reached{}, hit{};    // per depth: verified after j correct, correct
};

class BatchSpeculator {
public:
    // `target` must capture the main hidden rows (BatchForwardOptions::capture_main_hidden); `drafter` gets one
    // context per slot (set_sequences(slots)).
    BatchSpeculator(BatchForward& target, ds41::dspark::DeviceDrafter& drafter, std::uint32_t slots, BatchSpecOptions options);
    ~BatchSpeculator();
    BatchSpeculator(const BatchSpeculator&) = delete;
    BatchSpeculator& operator=(const BatchSpeculator&) = delete;

    // A new request in `slot` reading `prompt` from position 0: its drafting context and its controller history are
    // reset, its draft vocabulary chosen from the prompt.
    void start(std::uint32_t slot, std::span<const std::uint32_t> prompt);
    // A request whose prompt another path read (the device prefill, an imported state): its drafting context is reset
    // and starts at `position` (DeviceDrafter::skip_to; the drafter sees the rows from there on), the prompt and the
    // first token `first` in its draft vocabulary.
    void imported(std::uint32_t slot, std::span<const std::uint32_t> prompt, std::uint32_t position, std::uint32_t first);
    // The same with the seed rows the prefill kept (the input streams of the last three layers of the prompt's last
    // `rows` rows, [rows][hc][hidden] each, device memory ordered on `stream`): the context starts at position - rows
    // with them, as if the drafter had seen those rows (DevicePrefill::draft_seed).
    void imported(std::uint32_t slot, std::span<const std::uint32_t> prompt, std::uint32_t position, std::uint32_t first,
                  std::array<const float*, 3> seed, std::uint32_t rows, void* stream);

    // The next step of `seqs` (each with `remaining[i]` tokens it may still emit; a window never emits more): the
    // drafter runs here for the requests that draft. Returns the rows to step (owner: index into seqs).
    const PlannedStep& plan(std::span<BatchSequence* const> seqs, std::span<const std::uint32_t> remaining);
    // After target.step(plan().rows) returned `tokens`: verifies every window, commits, appends every kept row to its
    // drafting context and advances the sequences (position, fed, generated). Returns the commits that dropped rows
    // (slot, kept) -- what the other rank must apply before its next step.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> finish(std::span<BatchSequence* const> seqs, std::span<const std::uint32_t> tokens);

    // world 2. A draft round goes out only in steps where rank 1 drafts; it carries the commits made before it. The other
    // commits travel in the caller's own per-step message (take_commits on rank 0, follow_commit on rank 1).
    // rank 0: the commits not sent in a round yet (slot << 16 | kept); the caller sends them before the next step.
    [[nodiscard]] std::vector<std::uint32_t> take_commits();
    // rank 1: a commit of rank 0's last step, then follow_stage before the next step (or a round does it): the kept rows
    // of the last step (its rows and tokens, `step` its count; once per step) go to this rank's drafting contexts.
    void follow_commit(std::uint32_t slot, std::uint32_t kept);
    void follow_stage(std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens, std::uint64_t step);
    // rank 1: a round (receives it, applies its commits, stages, drafts the requests, sends the drafts back); false when
    // rank 0 closed the rounds. follow_packet: the same for a round already received (round_bytes, magic round_magic)
    // by a loop that also receives other 512-byte messages; the commits it applied are appended to `commits`.
    bool follow_round(std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens, std::uint64_t step);
    bool follow_packet(std::span<const std::byte> packet, std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens,
                       std::uint64_t step, std::vector<std::pair<std::uint32_t, std::uint32_t>>* commits = nullptr);
    // world 2, rank 1: `slot` was reset (a new request comes): its context is no longer this rank's.
    void release(std::uint32_t slot) { owners_.at(slot) = 0U; }
    static constexpr std::uint32_t round_magic = 0x44524631U;
    static constexpr std::size_t round_bytes = 512;
    [[nodiscard]] std::uint32_t owner(std::uint32_t slot) const noexcept { return owners_[slot]; }

    // Device bytes of the kept-row staging the constructor allocates for `slots` (the drafter's own: DeviceDrafter::footprint).
    [[nodiscard]] static std::uint64_t staging_bytes(std::uint32_t slots) noexcept;
    // false: the next plan()s draft nothing (every decoding request steps one plain row; the drafting contexts still
    // get every kept row, so drafting can resume). The tokens are those of plain steps either way.
    void allow_drafts(bool on) noexcept { allow_drafts_ = on; }

    [[nodiscard]] const BatchSpecStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ds41::dspark::BatchWindowController& controller() const noexcept { return controller_; }

private:
    struct Slots;
    float* staging(std::uint32_t slot) const;  // the slot's next free staging row
    void stage(std::uint32_t slot, std::uint32_t first, std::uint32_t count);
    void stage_from(std::uint32_t slot, std::array<const float*, 3> streams, std::uint32_t first, std::uint32_t count, void* stream);
    void flush(std::uint32_t slot);           // staged rows and pending context tokens -> the slot's drafting context

    BatchForward& target_;
    ds41::dspark::DeviceDrafter& drafter_;
    BatchSpecOptions options_;
    ds41::dspark::BatchWindowController controller_;
    std::unique_ptr<Slots> slots_;
    PlannedStep plan_;
    std::vector<ds41::dspark::Draft> drafts_;  // per owner (index into seqs): its draft
    std::vector<std::uint32_t> window_rows_;   // per owner: rows of its window (0: prompt rows or none)
    std::vector<bool> drafted_;                // per owner: the drafter ran
    std::uint32_t planned_drafting_ = 0, planned_decoders_ = 0, planned_total_ = 0;
    bool planned_prompt_rows_ = false;
    bool allow_drafts_ = true;
    double plan_seconds_ = 0.0;
    // measured costs (seconds, rank 0): a decode step of n requests without drafts (step + finish), a window row,
    // one request's draft (flush of its staged rows + the drafter call)
    std::array<double, ds41::dspark::BatchWindowController::max_requests + 1U> plain_mean_{};
    std::array<std::uint32_t, ds41::dspark::BatchWindowController::max_requests + 1U> plain_n_{};
    double row_mean_ = 0.0, draft_mean_ = 0.0, round_mean_ = 0.0;
    std::uint32_t row_n_ = 0, draft_n_ = 0, round_n_ = 0;
    std::vector<std::uint32_t> pending_commits_;  // world 2: slot << 16 | kept, for the next draft round
    // world 2, rank 1: what it knows of each slot (prompt, position) to stage the kept rows of its own requests
    std::vector<std::vector<std::uint32_t>> prompts_;
    std::vector<std::uint32_t> positions_;
    std::vector<std::uint32_t> owners_;  // the rank that drafts each slot's request
    std::array<std::uint32_t, ds41::dspark::BatchWindowController::max_requests> kept_{};  // rank 1: last step's commits (all ones: none)
    std::uint64_t staged_step_ = ~std::uint64_t{0};
    BatchSpecStats stats_;
};

}  // namespace strix::deepseek41
