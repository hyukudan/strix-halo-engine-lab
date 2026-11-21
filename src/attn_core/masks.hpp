#pragma once
// attn_core masks and block skip: cores call these, never re-derive them. Policies keep live(kt) (idx, MSA selected
// blocks) as their own block-skip hook on top.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_ATTN_MASK_HD __host__ __device__
#else
#define STRIX_ATTN_MASK_HD
#endif

namespace strix::attn_core::mask {

// Sliding window: keys p - window + 1 .. p; window 0 = full causal.
STRIX_ATTN_MASK_HD inline unsigned window_lo(unsigned p, unsigned window) { return window && p >= window ? p - window + 1U : 0U; }
STRIX_ATTN_MASK_HD inline bool causal_ok(unsigned key, unsigned pos, unsigned lo) { return key <= pos && key >= lo; }
// Keys a run of consecutive query positions [first, last] can see: [begin, end) (block skip outside it).
struct Range {
    unsigned begin, end;
};
STRIX_ATTN_MASK_HD inline Range tile_range(unsigned first, unsigned last, unsigned window) { return {window_lo(first, window), last + 1U}; }
// Per-row key lists: entry j of a list of n is live when j < n; dead entries load entry n - 1 (in bounds, finite)
// and are masked to -inf (n >= 1).
STRIX_ATTN_MASK_HD inline bool list_ok(unsigned j, unsigned n) { return j < n; }
STRIX_ATTN_MASK_HD inline unsigned list_clamp(unsigned j, unsigned n) { return j < n ? j : n - 1U; }

// Non-causal (bidirectional) attention over n keys on the causal cores, cores untouched. The policy returns
// position() = bidir_position(n) and window() = bidir_window(n): row r then sits at logical position n - 1 + r and
// its window is the n logical keys r .. n - 1 + r. The policy backs logical key p with physical key bidir_key(p, n)
// (K rows and V^T columns stored for bidir_capacity(n) logical positions), so every row sees each of the n keys
// exactly once, in an order rotated by r. all_keys(n) is the physical range.
STRIX_ATTN_MASK_HD constexpr Range all_keys(unsigned n) { return {0U, n}; }
STRIX_ATTN_MASK_HD constexpr unsigned bidir_position(unsigned n) { return n - 1U; }
STRIX_ATTN_MASK_HD constexpr unsigned bidir_window(unsigned n) { return n; }
STRIX_ATTN_MASK_HD constexpr unsigned bidir_key(unsigned p, unsigned n) { return p % n; }
// Logical positions the policy must back: the 16-key tiles up to the last row's position 2 n - 2.
STRIX_ATTN_MASK_HD constexpr unsigned bidir_capacity(unsigned n) { return (2U * n + 15U) & ~15U; }

}  // namespace strix::attn_core::mask

#undef STRIX_ATTN_MASK_HD
