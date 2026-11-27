#pragma once
// Key list of a sliding-window layer with pooled (compressed) history rows: the contract shared by the host reference
// (compressed_history_reference.hpp: window rows first, then every completed group) and the device attend
// (dense_decode.hpp `Keys` trait). Backend-neutral; no model names.
//
// For the query at position t: the logical keys are k = 0 .. count(t) - 1, with
//   window rows   k <  wc = mask::window_count(t, window): row window_lo(t, window) + k of the position-indexed rows,
//   pooled rows   k >= wc: pooled row (k - wc), stored from row `pool_base` on in the same row array.
// count(t) = wc + (t + 1) / ratio: a group closes at position ratio * (g + 1) - 1, and the row that completes it sees it.
// The online-softmax wave of a key is k % waves (dense_decode), the host declared sink_head walks the same indices.
#include "attn_core/masks.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_WINDOW_POOL_HD __host__ __device__
#else
#define STRIX_WINDOW_POOL_HD
#endif

namespace strix::attn_core::window_pool {

struct Keys {
    unsigned window, ratio, pool_base;

    STRIX_WINDOW_POOL_HD unsigned window_rows(unsigned t) const { return mask::window_count(t, window); }
    STRIX_WINDOW_POOL_HD unsigned pooled_rows(unsigned t) const { return (t + 1U) / ratio; }
    // Inclusive last logical key of position t (always >= 0: the window holds at least the row itself).
    STRIX_WINDOW_POOL_HD unsigned last(unsigned t) const { return window_rows(t) + pooled_rows(t) - 1U; }
    // Row of the row array that logical key k of position t reads.
    STRIX_WINDOW_POOL_HD unsigned slot(unsigned t, unsigned k) const {
        const unsigned wc = window_rows(t);
        return k < wc ? mask::window_lo(t, window) + k : pool_base + (k - wc);
    }
};

}  // namespace strix::attn_core::window_pool

#undef STRIX_WINDOW_POOL_HD
