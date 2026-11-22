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

}  // namespace strix::attn_split

#undef STRIX_SPLIT_PLAN_HD
