#pragma once

// Greedy speculative decoding with the DSpark drafter, independent of how the target runs.
//
// The target is seen one committed row at a time: step(token) feeds the token at the next position
// and returns its greedy next token, and main_hidden() gives the drafter's input row of that step
// (main_hidden_row of the inputs of the drafter's three target layers). A draft window is verified
// row by row: the target is stepped with t, then with each draft only while the previous draft
// equalled the target's greedy token, so a rejected draft is never fed and the target needs no
// rollback. With a window runtime (rows t, d_1..d_k in one pass) the accepted prefix is the same
// (verify_greedy); this driver is the reference of the protocol and the on/off token test.
//
// Flags (window_policy_from_env):
//   STRIX_DS41_DSPARK=1                 speculative decoding with the drafter (off: target-only)
//   STRIX_DS41_DSPARK_WINDOW=fixed:N    verify the first N drafts (0..5, default 5)
//   STRIX_DS41_DSPARK_WINDOW=auto[:c0,..,c5]  choose_drafts on the confidence head with the window
//                                       costs c_k (time of a k + 1 row window plus the draft step)

#include "strix/ds41_dspark.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix::ds41::dspark {

class SpeculativeTarget {
public:
    virtual ~SpeculativeTarget() = default;
    virtual std::uint32_t step(std::uint32_t token) = 0;
    // The drafter's input row of the last step (main_width values).
    virtual void main_hidden(std::span<float> out) const = 0;
};

class Drafter {
public:
    virtual ~Drafter() = default;
    // Appends committed rows (main_hidden [rows][main_width]) at the next positions.
    virtual void append(std::span<const float> main_hidden, std::uint32_t rows) = 0;
    virtual Draft draft(std::uint32_t token) = 0;
    // Context tokens (prompt, committed tokens) for a restricted draft vocabulary; ignored by default.
    virtual void add_context(std::span<const std::uint32_t> ids) { (void)ids; }
};

// The host reference drafter (ds41_dspark.hpp): bitwise the device drafter's tokens.
class HostDrafter final : public Drafter {
public:
    explicit HostDrafter(const Weights& weights, Vocab vocabulary = {}) : weights_(weights), vocabulary_(std::move(vocabulary)) {}
    void append(std::span<const float> main_hidden, std::uint32_t rows) override;
    Draft draft(std::uint32_t token) override;
    void add_context(std::span<const std::uint32_t> ids) override;

private:
    const Weights& weights_;
    Vocab vocabulary_;
    State state_;
};

// Survival controller: P(d_i correct | d_1..d_{i-1} correct) per depth i, calibrated online from the
// confidence head (8 confidence bins per depth, Beta(1, 1)-smoothed hit counts, learned only from
// positions whose earlier drafts were all correct); the window maximizes E[tokens] / cost[k] with
// E = 1 + sum_{j <= k} prod_{i <= j} P_i. Every 16th window verifies all five drafts (exploration).
// It only chooses how many drafts the target verifies; the tokens stay the target's greedy tokens.
class SurvivalController {
public:
    static constexpr std::uint32_t bins = 8;
    explicit SurvivalController(std::array<float, block_size + 1U> cost) : cost_(cost) {}
    [[nodiscard]] std::uint32_t choose(const Draft& d);
    void observe(const Draft& d, std::uint32_t verified, std::uint32_t accepted);
    [[nodiscard]] float probability(std::uint32_t depth, float confidence) const;
    [[nodiscard]] static std::uint32_t bin(float confidence);
    // Cost of verifying k drafts (a k + 1 row window plus the draft step), e.g. measured online.
    void set_cost(std::uint32_t k, float cost) { cost_.at(k) = cost; }

private:
    std::array<float, block_size + 1U> cost_;
    std::array<std::array<std::uint32_t, bins>, block_size> seen_{}, hits_{};
    std::uint64_t windows_ = 0;
};

// Window controller (STRIX_DS41_DSPARK_WINDOW=controller): whether to draft at all and how many drafts the target
// verifies, from MEASURED costs and calibrated survival. Deterministic in its inputs, so both ranks reach the same
// decisions when they share the cost table (rank 0 measures, a packet carries it); it never changes tokens.
//  - Costs (seconds): verify[R] = mean time of an R-row window (R = 1 is a single step), draft = mean draft step,
//    extra = mean append + control per drafted window. An unmeasured R is interpolated between measured sizes, or
//    extrapolated from the two largest measured ones (with only R1 measured: +15 % of R1 per row).
//  - Survival: P(d_j ok | d_<j ok) per depth j and confidence bin; each of the 40 cells is shrunk towards the bin's
//    curve pooled over all depths (4 pseudo-observations), so sparse cells borrow evidence.
//  - choose(): k maximising E_k / (draft + verify[k + 1] + extra), E_k = 1 + sum_j prod_{i<=j} p_i (k = 0 included).
//  - draft_next(): drafts when the best window by the pooled depth rates (accumulated hits / seen) beats a single
//    step, 1 / verify[1]. One exploration scheme: the first windows sweep k = 5, 4, 3, 2, 1 once (every size gets a
//    measured cost); afterwards every 16th decision explores: a drafted window when single steps lead, else the
//    window size with the fewest cost samples.
class WindowController {
public:
    static constexpr std::uint32_t bins = 8, max_rows = block_size + 1U, explore_every = 16;
    struct Costs {
        std::array<float, max_rows + 1U> verify{};  // [R], 0 = not measured
        std::array<std::uint32_t, max_rows + 1U> samples{};
        float draft = 0.0F, extra = 0.0F;
    };
    void set_costs(const Costs& costs) { costs_ = costs; }
    [[nodiscard]] const Costs& costs() const noexcept { return costs_; }
    [[nodiscard]] bool draft_next();
    [[nodiscard]] std::uint32_t choose(const Draft& d);
    void observe(const Draft& d, std::uint32_t verified, std::uint32_t accepted);
    [[nodiscard]] double verify_cost(std::uint32_t rows) const;
    [[nodiscard]] double probability(std::uint32_t depth, float confidence) const;
    [[nodiscard]] double pooled(std::uint32_t depth) const;
    [[nodiscard]] double window_rate(std::uint32_t drafts, const std::array<double, block_size>& p) const;

private:
    Costs costs_;
    std::array<std::array<std::uint32_t, bins>, block_size> seen_{}, hits_{};
    std::uint64_t decisions_ = 0;
    std::uint32_t sweep_ = 0;
    bool explore_ = false;
};

// Batch window controller (DSpark inside BatchForward, N > 1): one plan per step for every decoding request of the
// step, in two stages, from MEASURED costs. Deterministic in its inputs; it never changes tokens.
//  - Step cost model (seconds): plain[n] = a step of n requests with one row each (measured: the BatchForward step
//    of n segments), every extra window row adds `row`, every request that drafts adds `draft` (one drafter call per
//    request), and a step with drafts adds `extra` per request (append of its rows to its drafting context, commit).
//    Rows of prompts read in the same step are outside the plan (they do not depend on it).
//  - Stage 1, before drafting (no confidences yet): which requests draft, from each request's own survival per depth
//    (its verified windows, shrunk towards the rates pooled over every request with 8 pseudo-observations). The
//    choice maximises the step's expected tokens per second under the row budget; the all-plain plan is included.
//  - Stage 2, after drafting (the drafter calls are paid): each drafted request's rows (1..6) from its drafts'
//    confidences (P(d_j ok | d_<j ok) per depth x confidence bin pooled over requests, shrunk as in
//    WindowController), under the same row budget.
//  - Exploration: every 16th plan drafts one request that has room (round-robin; stage 2 still decides its rows), so
//    drafting costs and survival keep being measured while plain steps lead.
//  The search over row vectors is exhaustive up to 4 requests, else a local search (+1 / -1 / move one row) from
//  every uniform start; both include the all-plain plan.
class BatchWindowController {
public:
    static constexpr std::uint32_t max_requests = 8, max_window = block_size + 1U, explore_every = 16;
    struct Costs {
        std::array<float, max_requests + 1U> plain{};  // [n], 0 = not measured
        float row = 0.0F, draft = 0.0F, extra = 0.0F;  // row / draft 0 = not measured (defaults: see step_cost)
        // drafters running in parallel (the two ranks): n drafting requests cost ceil(n / drafters) drafts plus
        // `round` (the exchanges of a split draft round)
        std::uint32_t drafters = 1;
        float round = 0.0F;
    };
    explicit BatchWindowController(std::uint32_t slots);
    void set_costs(const Costs& costs) { costs_ = costs; }
    [[nodiscard]] const Costs& costs() const noexcept { return costs_; }
    // A new request in `slot`: its own survival history is forgotten.
    void reset(std::uint32_t slot);
    // Stage 1: for the decoding requests `slots` (at most limit[i] rows each, >= 1) and `budget` rows in total
    // (>= slots.size()), which ones draft.
    [[nodiscard]] std::vector<bool> plan_drafts(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> limit, std::uint32_t budget);
    // Stage 2: rows per request (1 = its token only) given which ones drafted and their drafts (others ignored).
    [[nodiscard]] std::vector<std::uint32_t> choose_rows(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> limit,
                                                         std::uint32_t budget, const std::vector<bool>& drafted, std::span<const Draft> drafts) const;
    // After the step: a drafted request verified `verified` drafts and accepted `accepted` of them.
    void observe(std::uint32_t slot, const Draft& d, std::uint32_t verified, std::uint32_t accepted);
    // Modelled step time of a plan (rows per request; `drafting` requests called the drafter).
    [[nodiscard]] double step_cost(std::span<const std::uint32_t> rows, std::uint32_t drafting) const;
    [[nodiscard]] double plain_cost(std::uint32_t n) const;
    [[nodiscard]] double probability(std::uint32_t depth, float confidence) const;
    [[nodiscard]] double slot_survival(std::uint32_t slot, std::uint32_t depth) const;

private:
    // the best rows per request for expected tokens expected[i][R - 1] under the budget; the step pays `drafting`
    // drafter calls (drafting < 0: one per request with more than one row)
    [[nodiscard]] std::vector<std::uint32_t> search(const std::vector<std::vector<double>>& expected, std::span<const std::uint32_t> limit,
                                                    std::uint32_t budget, int drafting) const;
    Costs costs_;
    std::array<std::array<std::uint32_t, WindowController::bins>, block_size> seen_{}, hits_{};  // pooled, by confidence
    std::vector<std::array<std::uint32_t, block_size>> slot_seen_, slot_hits_;
    std::uint64_t plans_ = 0;
};

struct WindowPolicy {
    bool automatic = false;
    std::uint32_t drafts = block_size;                  // fixed window
    std::array<float, block_size + 1U> cost{1.0F, 1.15F, 1.3F, 1.45F, 1.6F, 1.75F};
    std::shared_ptr<SurvivalController> survival;       // STRIX_DS41_DSPARK_WINDOW=survival[:c0,..,c5]
    std::shared_ptr<WindowController> controller;       // STRIX_DS41_DSPARK_WINDOW=controller
    [[nodiscard]] std::uint32_t choose(const Draft& d) const;
    void observe(const Draft& d, std::uint32_t verified, std::uint32_t accepted) const;
};
[[nodiscard]] bool dspark_enabled_from_env();
[[nodiscard]] WindowPolicy window_policy_from_env();

struct SpeculativeStats {
    std::uint32_t windows = 0, verified = 0, accepted = 0, target_steps = 0;
    std::array<std::uint32_t, block_size> accepted_at_depth{}; // windows whose drafts 1..j+1 were all accepted
    std::array<std::uint32_t, block_size> verified_at_depth{}; // windows that verified at least j+1 drafts
    std::array<std::uint32_t, block_size> reached_at_depth{};  // windows that verified draft j+1 after j correct drafts
    double draft_seconds = 0.0, target_seconds = 0.0;
};

// Target-only greedy decoding: feeds the prompt, then `count` generated tokens are returned.
[[nodiscard]] std::vector<std::uint32_t> generate_greedy(SpeculativeTarget& target, std::span<const std::uint32_t> prompt,
                                                         std::uint32_t count);
// Speculative greedy decoding; returns the same tokens as generate_greedy.
[[nodiscard]] std::vector<std::uint32_t> generate_speculative(SpeculativeTarget& target, Drafter& drafter,
                                                              std::span<const std::uint32_t> prompt, std::uint32_t count,
                                                              const WindowPolicy& policy, SpeculativeStats* stats = nullptr);

} // namespace strix::ds41::dspark
