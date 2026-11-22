#pragma once

// Key-split plan of one decode row (backend neutral, host and device).
//
// A decode row at position p attends to the keys [first, p] with first = window_lo(p, window) (window 0 = full prefix).
// The keys are cut into chunks counted from `first`: chunk c covers [first + c * chunk, min(first + (c + 1) * chunk, p + 1)).
// chunk = max(kUnit, kUnit * ceil(ceil(keys / kMaxSplits) / kUnit)), so splits = ceil(keys / chunk) <= kMaxSplits and
// every key belongs to exactly one chunk. The plan depends only on (p, window): never on the device, the SM count or the
// launch shape, so a kernel that merges its chunk partials in ascending chunk order gives the same bits on every device.

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_SPLIT_PLAN_HD __host__ __device__
#else
#define STRIX_SPLIT_PLAN_HD
#endif

namespace strix::attn_split {

inline constexpr unsigned kUnit = 256;       // positions per CTA pass (one key per thread)
inline constexpr unsigned kMaxSplits = 64;   // upper bound on chunks of one row

struct DecodeSplitPlan {
    unsigned first = 0;   // first visible key position
    unsigned keys = 0;    // visible keys, first .. position inclusive
    unsigned chunk = 0;   // keys per chunk (multiple of kUnit)
    unsigned splits = 0;  // chunks with at least one key
};

STRIX_SPLIT_PLAN_HD constexpr unsigned split_window_lo(unsigned position, unsigned window) {
    return window != 0U && position >= window ? position - window + 1U : 0U;
}

STRIX_SPLIT_PLAN_HD constexpr DecodeSplitPlan decode_split_plan(unsigned position, unsigned window) {
    DecodeSplitPlan p;
    p.first = split_window_lo(position, window);
    p.keys = position - p.first + 1U;
    const unsigned per_split = (p.keys + kMaxSplits - 1U) / kMaxSplits;
    const unsigned units = (per_split + kUnit - 1U) / kUnit;
    p.chunk = (units == 0U ? 1U : units) * kUnit;
    p.splits = (p.keys + p.chunk - 1U) / p.chunk;
    return p;
}

// Chunks a launch must provide for every position below `positions` (grid size fixed at capture time).
STRIX_SPLIT_PLAN_HD constexpr unsigned decode_split_capacity(unsigned positions, unsigned window) {
    if (positions == 0U) return 0U;
    // splits = ceil(keys / chunk) <= ceil(keys / kUnit) and <= kMaxSplits for every keys <= span (not monotone in keys).
    const unsigned span = window != 0U && window < positions ? window : positions;
    const unsigned units = (span + kUnit - 1U) / kUnit;
    return units < kMaxSplits ? units : kMaxSplits;
}

// Lanes form (strix/attn/prepared_lanes.hpp): lane i of one launch block is one query row with its own position and
// cache. Its plan is decode_split_plan(its position, window): the same chunks as the one-row launch of that row, so a
// lane's bits do not depend on the other lanes or on the block width. Workspace of a block: every lane slot owns
// `kv_heads` ticket words, then every lane slot owns the one-row record area [kv_head][capacity][group][record].
struct SplitLaneLayout {
    unsigned kv_heads = 0;
    unsigned capacity = 0;    // decode_split_capacity(positions, window): chunks the grid provides per row
    unsigned group = 0;       // query heads per KV head
    unsigned record = 0;      // floats per (kv head, chunk, group row)
    unsigned lane_slots = 0;  // lanes one launch block can serve

    STRIX_SPLIT_PLAN_HD constexpr unsigned ticket_words() const { return (kv_heads * lane_slots + 3U) / 4U * 4U; }
    STRIX_SPLIT_PLAN_HD constexpr unsigned lane_ticket(unsigned lane) const { return lane * kv_heads; }
    STRIX_SPLIT_PLAN_HD constexpr unsigned long long lane_floats() const {
        return 1ULL * kv_heads * capacity * group * record;
    }
    STRIX_SPLIT_PLAN_HD constexpr unsigned long long lane_partials(unsigned lane) const { return lane * lane_floats(); }
    STRIX_SPLIT_PLAN_HD constexpr unsigned long long bytes() const {
        return 4ULL * ticket_words() + 4ULL * lane_slots * lane_floats();
    }
};

// A lane may run in a block whose grid provides `capacity` chunks per row when its own plan needs no more.
STRIX_SPLIT_PLAN_HD constexpr bool lane_fits(unsigned position, unsigned window, unsigned capacity) {
    return decode_split_plan(position, window).splits <= capacity;
}

}  // namespace strix::attn_split

#undef STRIX_SPLIT_PLAN_HD
