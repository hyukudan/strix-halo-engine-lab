#pragma once

#include <cstdint>
#include <span>
#include <vector>

// Row budget across lanes: how many verify rows each live lane gets in one batched cycle.
//
// A cycle pays a fixed cost once and a cost per physical row, so a row is worth buying while its
// chance of yielding a token, divided by its cost, beats the cycle's tokens per millisecond.
// plan_rows() is a pure host function: descriptors in, plan out. It decides scheduling only; every
// lane's tokens still come from its own rows, so any plan gives the ids of a solo run.
namespace strix::serve {

// Capacity of equal-sized physical KV slots. The forward must fit every live
// slot with at least one row. A zero cap means use all affordable slots.
[[nodiscard]] std::uint32_t fixed_kv_slots(std::uint64_t budget, std::uint64_t slot_bytes,
                                          std::uint32_t rows, std::uint32_t cap = 0);

// Admit fixed resident slots after the decoder has allocated one slot, its
// weights and workspace. Cap the free capacity by declared total device memory
// and leave one eighth of the remaining bytes for runtime headroom.
[[nodiscard]] std::uint32_t resident_kv_slots(std::uint64_t free, std::uint64_t total,
    std::uint64_t device_limit, std::uint64_t kv_limit, std::uint64_t slot_bytes, std::uint32_t rows);

// resident_kv_slots beside an optional resident plane (`plane_bytes`, already allocated when `free` was read) that only
// speeds a path whose result is the same without it (e.g. an EXACT screened greedy head plane): the plane stays only
// when it costs no slot. Releasing it would return its bytes to `free` (never past `total`); when that admits more
// slots, keep_plane is false and `slots` is the count without the plane, so the caller releases the plane before it
// commits the slots. plane_bytes 0: resident_kv_slots, keep_plane false.
struct PlaneKvSlots {
    std::uint32_t slots = 0;
    bool keep_plane = false;
};
[[nodiscard]] PlaneKvSlots resident_kv_slots_with_plane(std::uint64_t free, std::uint64_t total,
    std::uint64_t device_limit, std::uint64_t kv_limit, std::uint64_t slot_bytes, std::uint32_t rows,
    std::uint64_t plane_bytes);

// Cycle time model: fixed_ms + row_ms * rows + lane_ms * running lanes + draft_row_ms * drafted rows,
// where a lane with r rows drafts r - 1 of them. The defaults are priors; callers pass measured costs.
struct CycleCost {
    double fixed_ms = 48;
    double row_ms = 7.5;
    double lane_ms = 0;
    double draft_row_ms = 0;
};

struct LaneDemand {
    std::uint32_t lane = 0;              // caller's lane id; ties break by it, then by position
    std::uint32_t remaining_tokens = 0;  // tokens the request may still emit; 0 = nothing to run
    std::uint32_t min_rows = 1;          // physical rows a running lane must hold (0 is read as 1)
    std::uint32_t max_rows = 1;          // window rows, draft depth + 1, engine segment limit
    // accept[k] = P(row k+1 yields a token | rows 1..k did). accept[0] belongs to the first row,
    // which is 1 for a lane whose first row is its committed target row. Missing entries: the
    // first row counts as certain, later rows use accept_prior. Values are clamped to [0, 1].
    std::span<const float> accept;
    double waited_ms = 0;                // time since this lane last ran; parked lanes age
    // Drain pacing only (RowLimits::drain): expected tokens per cycle the lane sustains when it gets every row it
    // offers, the caller's running estimate (one cycle's offer comes and goes); <= 0 = this cycle's offer.
    double rate = 0;
};

struct RowLimits {
    std::uint32_t rows_per_forward = 0;     // physical rows one forward can hold; 0 = no limit
    std::span<const std::uint32_t> totals;  // qualified physical totals, e.g. {4, 8, 16, 32}; empty = any
    std::uint32_t max_lanes = 0;            // lanes one cycle can run; 0 = no limit
    double max_cycle_ms = 0;                // per-token latency guard on the planned cycle; <= 0 = none
    // No request waits for a lane: the lanes drain, and the cohort ends when its slowest lane ends. An extra row
    // then shortens the run only on a lane that would otherwise end last; on a lane that ends earlier at its
    // minimum rows a yielded token only removes that lane's own later row (step 4 below).
    bool drain = false;
    // Price equal-width windows over the admitted lanes together. If their
    // mandatory rows differ, use the ordinary ragged planner.
    bool uniform_windows = false;
};

struct RowPlan {
    std::vector<std::uint32_t> rows;  // one entry per demand, in input order; 0 = parked this cycle
    double expected_tokens = 0;
    double expected_ms = 0;           // 0 when no lane runs
};

// Deterministic.
// 1. Admission: lanes with tokens left are taken longest-waiting first (then lane id) while the lane
//    limit and the minimum rows fit; the rest are parked. A parked lane's waited_ms keeps growing in
//    the caller, so it moves to the front and cannot starve.
// 2. Growth: rows beyond the minimum are ranked by the chance that they yield a token. The plan takes
//    the prefix of that ranking with the most expected tokens per millisecond (fewest rows on a tie),
//    never more rows than max_rows or remaining_tokens.
// 3. Qualified totals and the latency guard restrict which prefix lengths are allowed, so the rows
//    dropped to reach a total are always the lowest-value ones. If no length is allowed, the
//    shortest-waiting lane is parked and the plan is redone. A single lane that still cannot meet the
//    limits runs with its minimum rows (padded up to the smallest reachable qualified total).
// 4. Drain pacing (limits.drain): before growth, each lane with tokens left gets its expected cycles to the end
//    with every row it offers, remaining_tokens / rate (rate <= 0: expected tokens at max_rows); the largest is the
//    drain bound. A lane whose remaining tokens at its minimum rows alone end strictly within that bound gets no rows
//    beyond the minimum (its lead stays bounded; it drafts again once the slowest lane catches up). The slowest
//    lane(s) keep the full economics, so a single lane plans as without drain.
// min_rows is physical: a lane with fewer tokens left than min_rows still gets min_rows, and only
// the rows that can yield a token count towards expected_tokens.
RowPlan plan_rows(std::span<const LaneDemand> demands, const CycleCost& cost, const RowLimits& limits,
                  double accept_prior = 0.8);

}  // namespace strix::serve
