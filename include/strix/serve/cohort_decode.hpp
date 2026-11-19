#pragma once

#include "strix/mtp_context_draft.hpp"
#include "strix/serve/cycle_receipt.hpp"
#include "strix/serve/row_budget.hpp"
#include "strix/serve/step_planner.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <string>
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
// Drafts come from the shared context drafter (strix/mtp_context_draft.hpp), host only. Scheduling never changes a
// request's tokens: every verified row is the target's greedy choice on rows that the backend computes exactly as for
// the request alone (backend contract below), so each request's ids equal its solo run whatever the plan.
//
// Pure host code: no device types. Engines bind CohortBackend; the arithmetic lives here.
namespace strix::serve {

// One request's rows in a forward: `rows` consecutive rows of one lane (its pending token, then its drafts).
struct CohortGroup {
    std::uint32_t lane = 0;
    std::uint32_t rows = 1;
};

// Engine side of the contract. Lanes are independent sessions held by the backend.
// Invariant the backend guarantees: the greedy id of every row depends only on that lane's committed state and the
// row's inputs, never on the other groups of the forward, their count or their order (batch invariance).
class CohortBackend {
public:
    virtual ~CohortBackend() = default;
    [[nodiscard]] virtual std::uint32_t lanes() const = 0;             // sessions the backend holds
    [[nodiscard]] virtual std::uint32_t rows_per_forward() const = 0;  // physical rows of one forward, all groups
    [[nodiscard]] virtual std::uint32_t max_positions() const = 0;     // positions one lane can hold
    // Starts or continues the lane's prompt: appends `tokens` to the lane's committed state (no logits needed).
    // first = true resets the lane before the rows.
    virtual void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) = 0;
    // One forward over `groups` (tokens in group order). top[i] = greedy id after row i. The rows stay a pending
    // window until commit().
    virtual void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                        std::span<std::uint32_t> top) = 0;
    // Keeps accept[g] (1..rows) leading rows of group g of the last verify, discards the rest.
    virtual void commit(std::span<const std::uint32_t> accept) = 0;
    virtual void release(std::uint32_t lane) = 0;  // the lane is free
};

struct CohortRequest {
    std::uint64_t id = 0;
    std::vector<std::uint32_t> prompt;  // >= 2 tokens: the last one is the first decode row
    std::uint32_t max_tokens = 128;
    std::vector<std::uint32_t> stop;    // ids that end the request (emitted, then the request leaves)
};

struct CohortResult {
    std::uint64_t id = 0;
    std::vector<std::uint32_t> ids;     // generated ids
    std::string error;                  // non-empty: refused (never ran)
    double submit_ms = 0, first_ms = 0, done_ms = 0;  // decoder clock
    std::uint32_t steps = 0;            // decode steps the request ran in
    std::uint64_t drafted = 0, accepted = 0;
};

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
    CycleCost prior{};                  // the engine's measured prior: fixed + row * rows + lane * lanes (ms)
    bool learn_cost = true;             // replace the prior by the fit of the observed steps
    std::uint32_t fit_min_steps = 8;    // steps with at least two distinct row counts before the fit is used
    double accept_prior = 0.8;
    // Admission by bytes: a request reserves (prompt + max_tokens + 1 + max_draft) positions; 0 budget = no cap.
    std::uint64_t memory_budget = 0, position_bytes = 0, lane_bytes = 0;
    // Prompt service (StepPlanner time slicing).
    std::uint32_t prompt_slice = 512;   // prompt rows per slice
    double prompt_share = 1.0;          // ms of decode owed per ms of prompt while lanes decode
    std::uint64_t short_first_rows = 2048;
};

class CohortDecoder {
public:
    CohortDecoder(CohortBackend& backend, CohortConfig config);
    ~CohortDecoder();
    CohortDecoder(const CohortDecoder&) = delete;
    CohortDecoder& operator=(const CohortDecoder&) = delete;

    void submit(CohortRequest request);       // queued; joins at the next step boundary
    [[nodiscard]] bool idle() const noexcept;  // nothing queued or running
    void step();                              // one scheduler step
    void run();                               // steps until idle
    std::vector<CohortResult> take_finished();

    [[nodiscard]] CycleCost cost() const;      // the cost the next plan uses (fit or prior)
    [[nodiscard]] std::uint32_t live() const noexcept;
    [[nodiscard]] std::uint32_t peak_live() const noexcept { return peak_live_; }
    [[nodiscard]] std::uint64_t steps() const noexcept { return steps_; }
    [[nodiscard]] std::uint64_t physical_rows() const noexcept { return rows_; }
    [[nodiscard]] double decode_ms() const noexcept { return decode_ms_; }
    [[nodiscard]] double prompt_ms() const noexcept { return prompt_ms_; }
    // Test hook: a fixed clock step (ms) per call instead of the wall clock; 0 = wall clock.
    void set_fake_clock(double ms_per_read) noexcept { fake_clock_ms_ = ms_per_read; }

private:
    struct Lane;
    struct Waiting {
        CohortRequest request;
        double submit_ms = 0;
        unsigned skips = 0;
        std::uint64_t order = 0;
    };
    double now_ms();
    void admit();
    void finish(Lane& lane, std::string error = {});
    [[nodiscard]] std::uint64_t reserved_bytes(const CohortRequest& r) const;
    void observe_cost(std::uint32_t rows, std::uint32_t lanes, double ms);

    CohortBackend& backend_;
    CohortConfig config_;
    StepPlanner planner_;
    std::deque<Waiting> waiting_;
    std::vector<std::unique_ptr<Lane>> lanes_;  // index = backend lane; null = free
    std::vector<CohortResult> finished_;
    std::uint64_t order_ = 0, steps_ = 0, rows_ = 0, used_bytes_ = 0;
    std::uint32_t peak_live_ = 0;
    double decode_ms_ = 0, prompt_ms_ = 0, fake_clock_ms_ = 0, fake_now_ = 0;
    CycleLedger::FitSums sums_{};  // observed decode steps: y = ms, r = physical rows, l = lanes (CycleLedger::fit)
};

}  // namespace strix::serve
