#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Shared step planner: what runs in one serving step.
//
// Decode progress is reserved, prompt work is cut to a bounded slice, and short prompts may overtake
// long ones under a starvation bound. StepPlanner is a pure host class: lane descriptors in, plan out.
// It decides scheduling only; a prompt's rows and bits do not depend on the order or the size of its
// units, and every decode lane's tokens come from its own rows, so any plan gives the ids of a solo run.
//
// One planner covers three policies that already exist in the families:
//   - time slicing with a debt ledger: a prompt unit runs before the decode round only while the
//     decoders are not owed time (co_forward = false);
//   - shortest-first admission with aging (short_first.hpp: kMaxQueueOvertakes);
//   - row fill of one forward: one row per decoding lane, the rows left to prompts (co_forward = true).
namespace strix::serve {

struct StepLane {
    std::uint32_t lane = 0;               // caller's identifier, returned in the plan
    bool prompt = false;                  // true: still reading its prompt; false: decoding
    std::uint64_t prompt_remaining = 0;   // prompt rows left (prompt lanes)
    std::uint64_t order = 0;              // arrival order, smaller = older
    unsigned overtakes = 0;               // times a younger prompt ran first (caller keeps the count)
    std::uint32_t max_prompt_rows = 0;    // per-lane cap for this step, 0 = none (a first token that must run alone = 1)
    bool eligible = true;                 // false: cannot run this step (an aged ineligible lane blocks the others)
};

enum class PromptOrder : std::uint8_t {
    short_class,  // prompts with at most short_first_rows left go first, oldest first inside each class
    shortest,     // strictly fewest rows left first, arrival order on ties (requires short_first_rows != 0)
};

struct StepLimits {
    bool co_forward = false;                        // prompt rows may share the decode forward
    std::uint32_t rows_per_forward = 0;             // co_forward: physical rows of one forward
    std::span<const std::uint32_t> prompt_units;    // family-certified slice sizes, ascending; empty = the family's own unit
    double prompt_share = 0.0;                      // ms of decode owed per ms of prompt; 0 = prompts never wait for decode
    double max_decode_gap_ms = 0.0;                 // 0 = no bound; else the prompt unit is capped to fit while lanes decode
    // max_decode_gap_ms prices a unit as prompt_pass_fixed_ms + row cost x rows + decode_round_ms (gap_bounded_rows):
    // the fixed cost of any prefill pass (excluded from the observed row cost) and the decode round that follows it.
    double prompt_pass_fixed_ms = 0.0;
    double decode_round_ms = 0.0;
    std::uint64_t short_first_rows = 2048;          // 0 = oldest prompt first, no overtaking (an ineligible oldest lane blocks)
    PromptOrder prompt_order = PromptOrder::short_class;
    double prompt_row_ms = 0.0;                     // prior cost of one prompt row; the observed cost replaces it
};

// Rows of one prompt unit under a decode gap bound, the ONE gap calculation of the serving schedulers (StepPlanner's
// max_decode_gap_ms selector, CohortConfig::prompt_gap_ms): the most rows r with fixed_ms + row_ms * r + decode_ms <=
// gap_ms, where fixed_ms is the cost of any prefill pass (e.g. all expert weights re-streamed), row_ms the cost of one
// row and decode_ms the decode round that follows the unit, i.e. the gap a decoding lane sees across one unit.
// Clamped to [floor, cap]; no row cost or no target = the floor (floor >= cap = cap). The caller keeps chunk and cut
// admission (bits never depend on the rows).
// gap_pass_ms is the forward price of a unit, the ONE expression: the grant and every overshoot check use it. The
// grant is the largest r whose forward price fits, found by bisection over the forward predicate (the price is
// non-decreasing in r for a positive row cost): a truncated inverse quotient (119.99999999999999 for a target priced
// at exactly 120 rows) or one whose forward price rounds above the target (72 rows at 0.001 ms against 0.072 ms)
// never decides the grant, and the search costs at most 66 forward prices whatever the cap (a row cost so small that
// the price never moves cannot walk the cap row by row).
[[nodiscard]] constexpr double gap_pass_ms(double fixed_ms, double row_ms, double decode_ms, std::uint64_t rows) noexcept {
    return fixed_ms + row_ms * static_cast<double>(rows) + decode_ms;
}
[[nodiscard]] constexpr std::uint64_t gap_bounded_rows(double gap_ms, double fixed_ms, double row_ms, double decode_ms,
                                                       std::uint64_t floor, std::uint64_t cap) noexcept {
    if (!(row_ms > 0.0) || !(gap_ms > 0.0) || cap <= floor) return floor < cap ? floor : cap;
    const auto fits = [&](std::uint64_t r) { return gap_pass_ms(fixed_ms, row_ms, decode_ms, r) <= gap_ms; };
    if (fits(cap)) return cap;
    if (!fits(floor)) return floor;
    std::uint64_t lo = floor, hi = cap;  // fits(lo), !fits(hi)
    while (hi - lo > 1U) {
        const std::uint64_t mid = lo + (hi - lo) / 2U;
        if (fits(mid))
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

struct StepPlan {
    std::vector<std::uint32_t> decode_lanes;        // lanes that decode this step, in the order given
    struct Prompt {
        std::uint32_t lane;
        std::uint32_t rows;                         // 0 = the family's own unit (time slicing with no certified sizes)
    };
    std::vector<Prompt> prompt;                     // time slicing: at most one entry (the head of the unit or group)
    std::vector<std::uint32_t> overtaken;           // older prompt lanes passed over by the head: their overtakes grow by one
    bool prompt_before_decode = false;              // time slicing: run the prompt unit, then the decode round
};

class StepPlanner {
public:
    // Plans one step. Clears the debt when no prompt lane is present.
    StepPlan plan(std::span<const StepLane> lanes, const StepLimits& limits);
    // Overwrites the plan and reuses its vector capacity.
    void plan(std::span<const StepLane> lanes, const StepLimits& limits, StepPlan& out);

    // After the step: prompt time (less the time it was suspended for other service) is charged with the
    // share of the last plan, decode time pays it back. prompt_rows = rows the prompt unit really ran
    // (0 = the rows of the last plan); it feeds the per-row cost used by max_decode_gap_ms.
    void account(double prompt_ms, double decode_ms, double suspended_ms, std::uint64_t prompt_rows = 0);

    void reset();
    [[nodiscard]] double debt_ms() const noexcept { return debt_ms_; }
    [[nodiscard]] double prompt_row_ms() const noexcept { return observed_rows_ ? observed_ms_ / static_cast<double>(observed_rows_) : 0.0; }

private:
    double debt_ms_ = 0.0;
    double share_ = 0.0;
    double fixed_ms_ = 0.0;  // the last plan's prompt_pass_fixed_ms: not part of the observed row cost
    std::uint64_t planned_rows_ = 0;
    double observed_ms_ = 0.0;
    std::uint64_t observed_rows_ = 0;
    std::vector<std::size_t> prompt_arrival_;
};

}  // namespace strix::serve
