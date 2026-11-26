#include "strix/serve/row_budget.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace strix::serve {

std::uint32_t fixed_kv_slots(std::uint64_t budget, std::uint64_t slot_bytes, std::uint32_t rows, std::uint32_t cap) {
    if (slot_bytes == 0) return 0;
    const auto affordable = std::min<std::uint64_t>(budget / slot_bytes, rows);
    return static_cast<std::uint32_t>(cap == 0 ? affordable : std::min<std::uint64_t>(affordable, cap));
}

std::uint32_t resident_kv_slots(std::uint64_t free, std::uint64_t total, std::uint64_t device_limit,
                               std::uint64_t kv_limit, std::uint64_t slot_bytes, std::uint32_t rows) {
    if (free > total || slot_bytes > total - free) return 0;
    if (device_limit != 0U && device_limit < total) {
        const auto used = total - free;
        if (used > device_limit) return 0;
        free = device_limit - used;
    }
    // The first slot was already charged in 'used', so add it back for the
    // total slot count. The explicit KV cap includes that first slot.
    const auto slot_budget = free - free / 8U + slot_bytes;
    return fixed_kv_slots(kv_limit == 0U ? slot_budget : std::min(kv_limit, slot_budget), slot_bytes, rows);
}

PlaneKvSlots resident_kv_slots_with_plane(std::uint64_t free, std::uint64_t total, std::uint64_t device_limit,
                                          std::uint64_t kv_limit, std::uint64_t slot_bytes, std::uint32_t rows,
                                          std::uint64_t plane_bytes) {
    const auto with_plane = resident_kv_slots(free, total, device_limit, kv_limit, slot_bytes, rows);
    if (plane_bytes == 0U) return {with_plane, false};
    const auto used = total - std::min(free, total);
    const auto without = resident_kv_slots(free + std::min(plane_bytes, used), total, device_limit, kv_limit,
                                           slot_bytes, rows);
    return without > with_plane ? PlaneKvSlots{without, false} : PlaneKvSlots{with_plane, true};
}

namespace {

double clean(double value) { return std::isfinite(value) && value > 0 ? value : 0.0; }

double clamp01(double value, double fallback) {
    if (!std::isfinite(value)) return fallback;
    return std::clamp(value, 0.0, 1.0);
}

// Conditional acceptance of row `row` (0-based) of one lane.
double accept_at(const LaneDemand& demand, std::uint32_t row, double prior) {
    if (row < demand.accept.size()) return clamp01(static_cast<double>(demand.accept[row]), prior);
    return row == 0 ? 1.0 : prior;
}

bool qualified(std::uint32_t total, const RowLimits& limits) {
    if (limits.rows_per_forward != 0 && total > limits.rows_per_forward) return false;
    if (limits.totals.empty()) return true;
    return std::find(limits.totals.begin(), limits.totals.end(), total) != limits.totals.end();
}

struct Solver {
    std::span<const LaneDemand> demands;
    CycleCost cost;
    const RowLimits& limits;
    double prior;

    double cycle_ms(std::size_t lanes, std::uint32_t rows) const {
        // Every running lane holds at least one row, so rows >= lanes.
        return cost.fixed_ms + cost.row_ms * rows + cost.lane_ms * static_cast<double>(lanes) +
               cost.draft_row_ms * static_cast<double>(rows - lanes);
    }

    // Plans the admitted lanes. `relaxed` ignores the latency guard and accepts an unqualified
    // minimum; it is used only for a single lane that cannot meet the limits.
    template<class Lane, class Growth>
    bool solve(const std::vector<Lane>& lanes, bool relaxed, RowPlan& plan, std::vector<Growth>& growth) const {
        std::uint32_t min_total = 0;
        double min_tokens = 0;
        growth.clear();
        const auto common_min = lanes.front().min_rows;
        const bool uniform = limits.uniform_windows && std::all_of(lanes.begin(), lanes.end(), [&](const Lane& lane) {
            return lane.min_rows == common_min;
        });
        std::uint32_t common_max = ~0U;
        if (uniform) {
            for (const auto& lane : lanes)
                common_max = std::min(common_max, std::min(lane.max_rows, std::max(common_min, lane.useful_cap)));
        }
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            const Lane& lane = lanes[i];
            const LaneDemand& demand = demands[lane.index];
            min_total += lane.min_rows;
            const std::uint32_t top = std::min(common_max, std::min(lane.max_rows, std::max(lane.min_rows, lane.useful_cap)));
            double value = 1.0;
            for (std::uint32_t row = 0; row < top; ++row) {
                value *= accept_at(demand, row, prior);
                if (row < lane.min_rows) {
                    if (row < lane.useful_cap) min_tokens += value;
                } else {
                    growth.push_back({value, row, i, lane.id});
                }
            }
        }
        // Uniform windows buy a complete depth at once. Otherwise values
        // never rise along a lane, so ranking by value preserves row order.
        std::sort(growth.begin(), growth.end(), [uniform](const Growth& a, const Growth& b) {
            if (uniform && a.row != b.row) return a.row < b.row;
            if (a.value != b.value) return a.value > b.value;
            if (a.row != b.row) return a.row < b.row;
            if (a.id != b.id) return a.id < b.id;
            return a.lane < b.lane;
        });

        const double floor_ms = 1e-9;
        bool found = false;
        std::size_t best = 0;
        double best_tokens = 0, best_ms = 0, tokens = min_tokens;
        for (std::size_t extra = 0; extra <= growth.size(); ++extra) {
            if (extra != 0) tokens += growth[extra - 1].value;
            if (uniform && extra % lanes.size() != 0) continue;
            const std::uint32_t total = min_total + static_cast<std::uint32_t>(extra);
            if (limits.rows_per_forward != 0 && total > limits.rows_per_forward) break;
            const double ms = std::max(cycle_ms(lanes.size(), total), floor_ms);
            if (relaxed) {
                // Smallest qualified total wins; the bare minimum stands when none is reachable.
                if (qualified(total, limits)) { found = true; best = extra; best_tokens = tokens; best_ms = ms; break; }
                if (extra == 0) { found = true; best = 0; best_tokens = tokens; best_ms = ms; }
                continue;
            }
            if (!qualified(total, limits)) continue;
            if (limits.max_cycle_ms > 0 && ms > limits.max_cycle_ms) continue;
            if (!found || tokens * best_ms > best_tokens * ms) {
                found = true; best = extra; best_tokens = tokens; best_ms = ms;
            }
        }
        if (!found) return false;

        std::fill(plan.rows.begin(), plan.rows.end(), 0u);
        for (const Lane& lane : lanes) plan.rows[lane.index] = lane.min_rows;
        for (std::size_t extra = 0; extra < best; ++extra) ++plan.rows[lanes[growth[extra].lane].index];
        plan.expected_tokens = best_tokens;
        plan.expected_ms = cycle_ms(lanes.size(), min_total + static_cast<std::uint32_t>(best));
        return true;
    }
};

// Expected tokens of one cycle in which the lane holds `rows` rows (only rows below useful_cap can yield).
template<class Lane>
double expected_tokens(const LaneDemand& demand, const Lane& lane, std::uint32_t rows, double prior) {
    double value = 1.0, tokens = 0.0;
    for (std::uint32_t row = 0; row < std::min(rows, lane.useful_cap); ++row) {
        value *= accept_at(demand, row, prior);
        tokens += value;
    }
    return tokens;
}

// Step 4 of plan_rows (row_budget.hpp): the drain bound is the largest expected number of cycles to the end with
// every offered row; a lane that ends within it at its minimum rows keeps only its minimum.
template<class Lane>
void pace_drain(std::span<const LaneDemand> demands, std::vector<Lane>& lanes, double prior) {
    double bound = 0.0;
    for (const Lane& lane : lanes) {
        const LaneDemand& demand = demands[lane.index];
        const double full = std::isfinite(demand.rate) && demand.rate > 0.0
                                ? demand.rate
                                : expected_tokens(demand, lane, lane.max_rows, prior);
        if (full > 0.0) bound = std::max(bound, static_cast<double>(lane.useful_cap) / full);
    }
    for (Lane& lane : lanes) {
        if (lane.max_rows == lane.min_rows) continue;
        const double base = expected_tokens(demands[lane.index], lane, lane.min_rows, prior);
        if (base > 0.0 && static_cast<double>(lane.useful_cap) < bound * base) lane.max_rows = lane.min_rows;
    }
}

}  // namespace

RowPlan plan_rows(std::span<const LaneDemand> demands, const CycleCost& cost, const RowLimits& limits,
                  double accept_prior) {
    RowPlanner planner;
    planner.plan(demands, cost, limits, accept_prior);
    return std::move(planner.result_);
}

const RowPlan& RowPlanner::plan(std::span<const LaneDemand> demands, const CycleCost& cost, const RowLimits& limits,
                               double accept_prior) {
    result_.rows.assign(demands.size(), 0u);
    result_.expected_tokens = result_.expected_ms = 0.0;
    waiting_.clear();
    waiting_.reserve(demands.size());
    for (std::size_t i = 0; i < demands.size(); ++i) {
        const LaneDemand& demand = demands[i];
        if (demand.remaining_tokens == 0) continue;
        Lane lane;
        lane.index = i;
        lane.id = demand.lane;
        lane.min_rows = std::max<std::uint32_t>(demand.min_rows, 1);
        lane.max_rows = std::max(demand.max_rows, lane.min_rows);
        lane.useful_cap = demand.remaining_tokens;
        lane.waited_ms = clean(demand.waited_ms);
        // A lane whose minimum can never fit one forward is left parked.
        if (limits.rows_per_forward != 0 && lane.min_rows > limits.rows_per_forward) continue;
        waiting_.push_back(lane);
    }
    const double prior = clamp01(accept_prior, 0.8);
    if (limits.drain) pace_drain(demands, waiting_, prior);
    std::sort(waiting_.begin(), waiting_.end(), [](const Lane& a, const Lane& b) {
        if (a.waited_ms != b.waited_ms) return a.waited_ms > b.waited_ms;
        if (a.id != b.id) return a.id < b.id;
        return a.index < b.index;
    });

    admitted_.clear();
    std::uint32_t min_total = 0;
    for (const Lane& lane : waiting_) {
        if (limits.max_lanes != 0 && admitted_.size() >= limits.max_lanes) break;
        if (limits.rows_per_forward != 0 && lane.min_rows > limits.rows_per_forward - min_total) continue;
        min_total += lane.min_rows;
        admitted_.push_back(lane);
    }
    if (admitted_.empty()) return result_;

    const Solver solver{demands,
                        {clean(cost.fixed_ms), clean(cost.row_ms), clean(cost.lane_ms), clean(cost.draft_row_ms)},
                        limits, prior};
    while (!solver.solve(admitted_, false, result_, growth_)) {
        if (admitted_.size() == 1) {
            solver.solve(admitted_, true, result_, growth_);
            break;
        }
        admitted_.pop_back();  // shortest-waiting lane; it ages and returns first
    }
    return result_;
}

}  // namespace strix::serve
