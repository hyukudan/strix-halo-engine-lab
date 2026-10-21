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
