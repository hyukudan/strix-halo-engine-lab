// build_routes_fast_kernel's body (rows * top_k <= 64), shared with the fused router selection + routes kernel of
// STRIX_DS41_R1FUSE_ROUTES (hip_ds41_formats.hip). Threads past the 64 entries only take part in the barriers.
#pragma once

#include "strix/hip_ds41_formats.hpp"

#include <hip/hip_runtime.h>

namespace strix::ds41 {
__device__ __forceinline__ void build_routes_fast_body(const BuildRoutesView& view) {
    __shared__ unsigned id[64];
    __shared__ unsigned leader[64];
    __shared__ unsigned totals[3];  // union experts, owned entries, most entries of one expert
    const unsigned t = threadIdx.x, n = view.rows * view.top_k;
    if (t < 3U) totals[t] = 0U;
    if (t < n) id[t] = view.ids[t];
    __syncthreads();
    const bool mine = t < n && id[t] >= view.first && id[t] < view.end;
    unsigned before = 0, below = 0, same = 0;
    if (t < n) {
        for (unsigned s = 0; s < n; ++s) {
            const bool owned = id[s] >= view.first && id[s] < view.end;
            if (id[s] == id[t]) { before += s < t ? 1U : 0U; ++same; }
            if (owned && id[s] < id[t]) below += 1U;
        }
    }
    const bool lead = mine && before == 0U;
    if (t < 64U) leader[t] = lead ? 1U : 0U;
    if (lead) { atomicAdd(&totals[0], 1U); atomicMax(&totals[2], same); }
    if (mine) atomicAdd(&totals[1], 1U);
    __syncthreads();
    if (mine) {
        unsigned position = 0;
        for (unsigned s = 0; s < n; ++s) position += leader[s] && id[s] < id[t] ? 1U : 0U;
        const unsigned entry = below + before;
        view.rows_out[entry] = t / view.top_k;
        view.slots[entry] = t;
        if (before == 0U) { view.experts[position] = id[t] - view.base; view.begins[position] = below; }
    }
    if (t < n) {
        const unsigned row = t / view.top_k;
        unsigned rank = 0, low = 0;
        for (unsigned k = 0; k < view.top_k; ++k) {
            const unsigned other = id[row * view.top_k + k];
            rank += other < id[t] ? 1U : 0U;
            low += other < view.split ? 1U : 0U;
        }
        view.order[row * view.top_k + rank] = t;
        if (t % view.top_k == 0U) view.low_counts[row] = low;
    }
    if (t == 0U) {
        view.begins[totals[0]] = totals[1];
        view.counts[0] = totals[0];
        view.counts[1] = totals[2];
    }
}

} // namespace strix::ds41
