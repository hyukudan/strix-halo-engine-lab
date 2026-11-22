#pragma once

// The one key-split plan of a decode attention row (backend neutral, host and device): position canonical.
//
// A row at position p attends to the keys [first, p], first = window_lo(p, window) (window 0 = full prefix). Keys are
// grouped in absolute units of `unit` positions, unit u = [u * unit, (u + 1) * unit). A chunk is k consecutive units
// aligned to absolute multiples of k: chunk index a covers [a * k * unit, (a + 1) * k * unit), clipped to [first, p].
//   span = p / unit - first / unit + 1 (units the row touches)
//   k    = 1 when max_splits == 0 (uncapped) or span <= max_splits, else ceil(span / (max_splits - 1))
//   splits = p / (k unit) - first / (k unit) + 1 <= max_splits  (span <= (max_splits - 1) k bounds the difference by
//            floor((span - 1) / k) + 1 <= max_splits - 1)
// Short-row rule (shape.short_one_pass): a row of at most `unit` keys is one chunk, one pass starting at `first` (no
// record, no merge), so a short sliding window never pays a two-chunk merge for straddling a unit boundary.
// Otherwise a kernel thread lane is the key's absolute offset in its unit (key mod unit). The plan depends only on (p, window,
// shape): never on the device, the SM count, the other rows of a launch or the launch width, so M = 1 decode, verify rows,
// tree rows and cohort lanes give one key the same chunk and lane, and an ascending-chunk merge gives the same bits.
// The uncapped shape {256, 0} is the absolute 256-key split of the global-max decode family (split = key / 256).

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_SPLIT_PLAN_HD __host__ __device__
#else
#define STRIX_SPLIT_PLAN_HD
#endif

namespace strix::attn_split {

inline constexpr unsigned kUnit = 256;       // positions per CTA pass (one key per thread)
inline constexpr unsigned kMaxSplits = 64;   // upper bound on chunks of one row (capped decode shape)

// Shape of the plan: data of a kernel family, not of a model. max_splits 0 = uncapped (k = 1), else >= 2.
struct SplitShape {
    unsigned unit = kUnit;
    unsigned max_splits = kMaxSplits;
    bool short_one_pass = true;
};

inline constexpr SplitShape kDecodeShape{kUnit, kMaxSplits, true};  // online-P prepared decode split (one-pass merge)
inline constexpr SplitShape kUnitShape{kUnit, 0U, false};           // absolute unit split (global-max decode family)

struct DecodeSplitPlan {
    unsigned first = 0;        // first visible key position
    unsigned last = 0;         // the row position (last visible key)
    unsigned keys = 0;         // visible keys, first .. last inclusive
    unsigned chunk = 0;        // keys per aligned chunk (k x unit)
    unsigned first_chunk = 0;  // absolute index of the chunk holding `first`
    unsigned splits = 0;       // chunks with at least one key
    unsigned origin = 0;       // pass-loop start of chunk 0: the unit holding `first`, or `first` for a short row

    // Pass-loop start of local chunk c: an absolute multiple of `chunk` (key - origin = thread lane); chunk 0 starts at
    // the unit holding `first` (still unit aligned), or at `first` itself for a short row.
    STRIX_SPLIT_PLAN_HD constexpr unsigned chunk_origin(unsigned c) const {
        return c == 0U ? origin : (first_chunk + c) * chunk;
    }
    // Keys of local chunk c: [chunk_begin, chunk_end), the aligned interval clipped to [first, last].
    STRIX_SPLIT_PLAN_HD constexpr unsigned chunk_begin(unsigned c) const {
        return c == 0U ? first : (first_chunk + c) * chunk;
    }
    STRIX_SPLIT_PLAN_HD constexpr unsigned chunk_end(unsigned c) const {
        return c + 1U >= splits ? last + 1U : (first_chunk + c + 1U) * chunk;
    }
};

STRIX_SPLIT_PLAN_HD constexpr unsigned split_window_lo(unsigned position, unsigned window) {
    return window != 0U && position >= window ? position - window + 1U : 0U;
}

STRIX_SPLIT_PLAN_HD constexpr DecodeSplitPlan decode_split_plan(unsigned position, unsigned window,
                                                                SplitShape shape = kDecodeShape) {
    DecodeSplitPlan p;
    p.first = split_window_lo(position, window);
    p.last = position;
    p.keys = position - p.first + 1U;
    const unsigned span = position / shape.unit - p.first / shape.unit + 1U;
    const unsigned k = shape.max_splits == 0U || span <= shape.max_splits
                           ? 1U
                           : (span + shape.max_splits - 2U) / (shape.max_splits - 1U);
    p.chunk = k * shape.unit;
    p.first_chunk = p.first / p.chunk;
    const bool short_row = shape.short_one_pass && p.keys <= shape.unit;
    p.splits = short_row ? 1U : position / p.chunk - p.first_chunk + 1U;
    p.origin = short_row ? p.first : p.first / shape.unit * shape.unit;  // no fully masked leading pass
    return p;
}

// Crossover: from which row the split is bound at all (backend neutral; derived from the shape, no device constant).
// A row whose plan is ONE chunk gets no key parallelism from the split: it is a single pass like the one-pass flash it
// replaces, in a different launch shape (measured: -3.5 % tok/s at a 7-token context). A caller that owns a one-pass
// flash therefore keeps it for such rows and binds the split from `crossover_keys` visible keys on: a row with that many
// keys has at least `min_splits` chunks (for min_splits <= max_splits / 2), and with the default min_splits = 2 the rule
// is exactly "the plan has two chunks or more". Visible keys never shrink as the position grows, for every window, so
// the rule is monotone: a sequence crosses it once per window class (a captured program switches at most once), and a
// window smaller than the crossover never engages.
struct SplitCrossover {
    unsigned min_splits = 2;  // chunks a row needs before the split is bound; <= 1 = every row
};

STRIX_SPLIT_PLAN_HD constexpr unsigned split_visible_keys(unsigned position, unsigned window) {
    return position - split_window_lo(position, window) + 1U;
}

STRIX_SPLIT_PLAN_HD constexpr unsigned crossover_keys(SplitCrossover crossover = {}, SplitShape shape = kDecodeShape) {
    return crossover.min_splits <= 1U ? 1U : (crossover.min_splits - 1U) * shape.unit + 1U;
}

STRIX_SPLIT_PLAN_HD constexpr bool split_engages(unsigned position, unsigned window, SplitCrossover crossover = {},
                                                SplitShape shape = kDecodeShape) {
    return split_visible_keys(position, window) >= crossover_keys(crossover, shape);
}

// First position of a window class that engages the split (every later position engages too); kNeverEngages when the
// window holds fewer keys than the crossover.
inline constexpr unsigned kNeverEngages = ~0U;
STRIX_SPLIT_PLAN_HD constexpr unsigned crossover_position(unsigned window, SplitCrossover crossover = {},
                                                          SplitShape shape = kDecodeShape) {
    const unsigned keys = crossover_keys(crossover, shape);
    return window != 0U && window < keys ? kNeverEngages : keys - 1U;
}

// Chunks a launch must provide for every position below `positions` (grid size fixed at capture time): splits <= span,
// span <= ceil(positions / unit), a window of w keys touches at most (w - 1) / unit + 2 units, and splits <= max_splits.
STRIX_SPLIT_PLAN_HD constexpr unsigned decode_split_capacity(unsigned positions, unsigned window,
                                                             SplitShape shape = kDecodeShape) {
    if (positions == 0U) return 0U;
    unsigned units = (positions + shape.unit - 1U) / shape.unit;
    if (window != 0U && window < positions) {
        const unsigned windowed = (window - 1U) / shape.unit + 2U;
        units = windowed < units ? windowed : units;
    }
    return shape.max_splits != 0U && shape.max_splits < units ? shape.max_splits : units;
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
