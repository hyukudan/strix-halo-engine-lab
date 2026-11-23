#pragma once
// Exact ordered top-k, "Threshold" schedule of the shared select family (src/attn_core/select.hpp has the Chunked /
// Segmented / FourPerThread schedules of the same rule). Rule, every schedule: the k largest uint32 order keys, equal keys
// by the smaller position, emitted in ascending position. This schedule: per-wave histograms merged per bin, a lane scan
// of the bins from the top, then one ordered compaction per block-size chunk with a per-element callback; plus the
// float order key, the candidate-block pass (block maxima, newest block always kept), the two-level slice / merge form
// for long rows and the count <= k shortcut. Integer only: the schedule never changes the selected positions.
// Lifted unchanged from the DeepSeek-family sparse indexer (one block of Threads threads per row / slice).
//
// Template parameters:
//   Threads  block size (multiple of 32, < 65536; the row forms need Threads >= 256 and Threads >= kSlice / 8).
//   Legacy   false: parallel per-bin merge, whole-bin early end and the packed (gt | eq << 16) compaction scan;
//            true:  wave 0 merges the waves, four full passes, two compaction scans. Same positions either way.
//   View     the caller's row view (passed through, never copied), fields:
//              rows, ratio, max_count, topk, cand_blocks, block_size; pool; pos (u64, count = (pos + 1) / ratio);
//              scores (rows x max_count f32); candidates (rows x ceil(max_count / block_size) u8); keys (u32 scratch);
//              idx (rows x topk), idx_count (rows); compact (bool).
//            decltype(v.pool) names the enumerators candidate_source and candidate_user (any other value = plain).
// Host-neutral statement of the same rule and passes: select_threshold_rule.hpp (scalar, used by the CPU test).
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strix::attn_core::select::threshold {

inline constexpr unsigned kSlice = 4096;   // positions per level-one slice of the two-level form

// Monotone float -> uint32 (larger float, larger key; -inf below every finite value, NaN by its bits).
__device__ __forceinline__ std::uint32_t order_key(float x) {
    const std::uint32_t b = __builtin_bit_cast(std::uint32_t, x);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}

template <unsigned Threads>
__device__ unsigned block_exclusive_scan(unsigned x, unsigned* warp_tot, unsigned& total) {
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    unsigned incl = x;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const unsigned y = __shfl_up(incl, static_cast<unsigned>(d), 32);
        if (lane >= static_cast<unsigned>(d)) incl += y;
    }
    if (lane == 31U) warp_tot[w] = incl;
    __syncthreads();
    if (w == 0U) {
        unsigned t = warp_tot[lane], ti = t;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const unsigned y = __shfl_up(ti, static_cast<unsigned>(d), 32);
            if (lane >= static_cast<unsigned>(d)) ti += y;
        }
        warp_tot[lane] = ti - t;
        if (lane == 31U) warp_tot[32] = ti;
    }
    __syncthreads();
    const unsigned r = warp_tot[w] + incl - x;
    total = warp_tot[32];
    __syncthreads();
    return r;
}
// Threshold T (the k-th largest key) and how many keys equal to T are taken (need); requires n > k. Per-wave
// histograms (integer counts, so the result does not depend on the order of the adds), merged into one total per bin by
// 256 threads in parallel (each sums its bin over the waves); wave 0 scans the 256 totals from the top, 8 per lane,
// with a lane prefix scan. A bin taken whole ends the search early: every key of the bin and above is taken and every
// key below it is not, so T = (the bin's lowest key) - 1 with need 0 selects the same keys (the k-th largest key's
// equals are all in the bin, so the rule "key > T, then the first need keys equal to T" picks them all either way).
template <unsigned Threads>
__device__ void radix_threshold(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, unsigned (*hist)[256],
    std::uint32_t& thr, std::uint32_t& need) {
    constexpr unsigned kWaves = Threads / 32;
    __shared__ std::uint32_t prefix_s, rem_s, done_s, total[256];
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    std::uint32_t prefix = 0, mask = 0, rem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned b = lane; b < 256U; b += 32U) hist[w][b] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n; i += Threads) {
            const std::uint32_t key = keys[i];
            if ((key & mask) == prefix) atomicAdd(&hist[w][(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (threadIdx.x < 256U) {
            unsigned c = 0;
            for (unsigned ww = 0; ww < kWaves; ++ww) c += hist[ww][threadIdx.x];
            total[threadIdx.x] = c;
        }
        __syncthreads();
        if (w == 0U) {
            unsigned cnt[8], local = 0;   // lane owns bins 255 - 8 lane - i, i = 0..7 (descending)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                cnt[i] = total[255U - 8U * lane - i];
                local += cnt[i];
            }
            unsigned incl = local;
#pragma unroll
            for (int d = 1; d < 32; d *= 2) {
                const unsigned y = __shfl_up(incl, static_cast<unsigned>(d), 32);
                if (lane >= static_cast<unsigned>(d)) incl += y;
            }
            const unsigned before = incl - local;
            if (before < rem && incl >= rem) {   // exactly one lane
                unsigned cum = before, i = 0;
                while (cum + cnt[i] < rem) { cum += cnt[i]; ++i; }
                const std::uint32_t low = prefix | ((255U - 8U * lane - i) << shift);
                const bool whole = cnt[i] == rem - cum && low != 0U;
                prefix_s = whole ? low - 1U : low;
                rem_s = whole ? 0U : rem - cum;
                done_s = whole ? 1U : 0U;
            }
        }
        __syncthreads();
        prefix = prefix_s; rem = rem_s;
        const bool done = done_s != 0U;
        mask |= 255U << shift;
        __syncthreads();
        if (done) break;
    }
    thr = prefix; need = rem;
}

// Visits keys in index order; f(i, selected, out_index) for every i < n. One block scan per chunk counts the keys above
// T (low 16 bits) and equal to T (high 16 bits; a chunk has at most Threads of each): the selected keys before i are
// those above T plus the equal ones whose rank among the equals is below need.
template <unsigned Threads, class F>
__device__ void ordered_select(const std::uint32_t* keys, std::uint32_t n, std::uint32_t thr, std::uint32_t need,
    unsigned* warp_tot, F f) {
    static_assert(Threads < 65536U, "packed counts");
    std::uint32_t eq_carry = 0, out_carry = 0;
    for (std::uint32_t base = 0; base < n; base += Threads) {
        const std::uint32_t i = base + threadIdx.x;
        const std::uint32_t key = i < n ? keys[i] : 0U;
        const unsigned gt = i < n && key > thr ? 1U : 0U, eq = i < n && key == thr ? 1U : 0U;
        unsigned total;
        const unsigned ex = block_exclusive_scan<Threads>(gt | eq << 16U, warp_tot, total);
        const std::uint32_t eq_rank = eq_carry + (ex >> 16U), taken = min(eq_carry, need);
        const unsigned sel = gt != 0U || (eq != 0U && eq_rank < need) ? 1U : 0U;
        const unsigned out = out_carry + (ex & 0xFFFFU) + (min(eq_rank, need) - taken);
        if (i < n) f(i, sel != 0U, out);
        const std::uint32_t eq_next = eq_carry + (total >> 16U);
        out_carry += (total & 0xFFFFU) + (min(eq_next, need) - taken);
        eq_carry = eq_next;
    }
}

// Legacy schedule (the <true> instantiations): the selection before the parallel histogram merge, the packed scan and
// the whole-bin end (same thresholds, same selected positions in the same order).
template <unsigned Threads>
__device__ void radix_threshold_v1(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, unsigned (*hist)[256],
    std::uint32_t& thr, std::uint32_t& need) {
    constexpr unsigned kWaves = Threads / 32;
    __shared__ std::uint32_t prefix_s, rem_s;
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    std::uint32_t prefix = 0, mask = 0, rem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned b = lane; b < 256U; b += 32U) hist[w][b] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n; i += Threads) {
            const std::uint32_t key = keys[i];
            if ((key & mask) == prefix) atomicAdd(&hist[w][(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (w == 0U) {
            unsigned cnt[8], local = 0;   // lane owns bins 255 - 8 lane - i, i = 0..7 (descending)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned b = 255U - 8U * lane - i;
                unsigned c = 0;
                for (unsigned ww = 0; ww < kWaves; ++ww) c += hist[ww][b];
                cnt[i] = c; local += c;
            }
            unsigned incl = local;
#pragma unroll
            for (int d = 1; d < 32; d *= 2) {
                const unsigned y = __shfl_up(incl, static_cast<unsigned>(d), 32);
                if (lane >= static_cast<unsigned>(d)) incl += y;
            }
            const unsigned before = incl - local;
            if (before < rem && incl >= rem) {   // exactly one lane
                unsigned cum = before, i = 0;
                while (cum + cnt[i] < rem) { cum += cnt[i]; ++i; }
                prefix_s = prefix | ((255U - 8U * lane - i) << shift);
                rem_s = rem - cum;
            }
        }
        __syncthreads();
        prefix = prefix_s; rem = rem_s;
        mask |= 255U << shift;
        __syncthreads();
    }
    thr = prefix; need = rem;
}
// Visits keys in index order; f(i, selected, out_index) for every i < n.
template <unsigned Threads, class F>
__device__ void ordered_select_v1(const std::uint32_t* keys, std::uint32_t n, std::uint32_t thr, std::uint32_t need,
    unsigned* warp_tot, F f) {
    std::uint32_t eq_carry = 0, out_carry = 0;
    for (std::uint32_t base = 0; base < n; base += Threads) {
        const std::uint32_t i = base + threadIdx.x;
        const std::uint32_t key = i < n ? keys[i] : 0U;
        const unsigned eq = i < n && key == thr ? 1U : 0U;
        unsigned eq_total;
        const unsigned eq_rank = block_exclusive_scan<Threads>(eq, warp_tot, eq_total) + eq_carry;
        const unsigned sel = i < n && (key > thr || (eq != 0U && eq_rank < need)) ? 1U : 0U;
        unsigned sel_total;
        const unsigned out = block_exclusive_scan<Threads>(sel, warp_tot, sel_total) + out_carry;
        if (i < n) f(i, sel != 0U, out);
        eq_carry += eq_total; out_carry += sel_total;
    }
}

template <unsigned Threads, bool Legacy>
__device__ __forceinline__ void select_threshold(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, unsigned (*hist)[256],
    std::uint32_t& thr, std::uint32_t& need) {
    if constexpr (Legacy) radix_threshold_v1<Threads>(keys, n, k, hist, thr, need);
    else radix_threshold<Threads>(keys, n, k, hist, thr, need);
}
template <unsigned Threads, bool Legacy, class F>
__device__ __forceinline__ void select_ordered(const std::uint32_t* keys, std::uint32_t n, std::uint32_t thr, std::uint32_t need,
    unsigned* warp_tot, F f) {
    if constexpr (Legacy) ordered_select_v1<Threads>(keys, n, thr, need, warp_tot, f);
    else ordered_select<Threads>(keys, n, thr, need, warp_tot, f);
}

// One block per row. Candidate source: the block maxima (the newest block's key is +inf) select cand_blocks candidate
// blocks (every block above -inf when there are at most cand_blocks). Candidate user: positions outside the candidate
// blocks score -inf. Then the row's top-k positions in ascending order (every position when count <= topk).
template <unsigned Threads, bool Legacy, class View>
__device__ __forceinline__ void select_core(const View& v, const uint3 bid, unsigned (*hist)[256], unsigned* warp_tot) {
    using Pool = decltype(v.pool);
    const unsigned r = bid.x;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio);
    const float* scores = v.scores + static_cast<std::size_t>(r) * v.max_count;
    std::uint32_t* keys = v.keys + static_cast<std::size_t>(r) * v.max_count;
    std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
    const float ninf = -__builtin_inff();
    if (v.pool == Pool::candidate_source && n != 0U) {
        const std::uint32_t nb = (n + v.block_size - 1) / v.block_size, last = (n - 1) / v.block_size;
        for (std::uint32_t b = threadIdx.x; b < nb; b += Threads) {
            float m = ninf;
            for (std::uint32_t t = b * v.block_size; t < (b + 1) * v.block_size && t < n; ++t) m = fmaxf(m, scores[t]);
            keys[b] = order_key(b == last ? __builtin_inff() : m);
        }
        __syncthreads();
        const std::uint32_t ninf_key = order_key(ninf);
        if (nb <= v.cand_blocks) {
            for (std::uint32_t b = threadIdx.x; b < nb; b += Threads) cand[b] = keys[b] > ninf_key ? 1U : 0U;
        } else {
            std::uint32_t thr, need;
            select_threshold<Threads, Legacy>(keys, nb, v.cand_blocks, hist, thr, need);
            select_ordered<Threads, Legacy>(keys, nb, thr, need, warp_tot, [&](std::uint32_t b, bool s, unsigned) {
                cand[b] = s && keys[b] > ninf_key ? 1U : 0U;
            });
        }
        __syncthreads();
    }
    for (std::uint32_t t = threadIdx.x; t < n; t += Threads)
        keys[t] = order_key(v.pool == Pool::candidate_user && cand[t / v.block_size] == 0U ? ninf : scores[t]);
    __syncthreads();
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(r) * v.topk;
    if (n <= v.topk) {
        for (std::uint32_t t = threadIdx.x; t < n; t += Threads) idx[t] = t;
        if (threadIdx.x == 0U) v.idx_count[r] = n;
        return;
    }
    std::uint32_t thr, need;
    select_threshold<Threads, Legacy>(keys, n, v.topk, hist, thr, need);
    select_ordered<Threads, Legacy>(keys, n, thr, need, warp_tot, [&](std::uint32_t t, bool s, unsigned out) { if (s) idx[out] = t; });
    if (threadIdx.x == 0U) v.idx_count[r] = v.topk;
}
template <unsigned Threads, bool Legacy, class View>
__device__ __forceinline__ void select_body(const View& v, const uint3 bid) {
    __shared__ unsigned hist[Threads / 32][256];
    __shared__ unsigned warp_tot[33];
    select_core<Threads, Legacy>(v, bid, hist, warp_tot);
}
// Two-level selection for long rows (max_count > kSlice). Level one, one block per slice of kSlice positions: the
// slice's keys in LDS, its own top-k (same rule: key, then the smaller position) written in position order and
// padded to k with key 0 (below every float's key); for the candidate source also the slice's block maxima. Every
// element of the row's top-k is in its slice's top-k (fewer than k elements of the slice beat it), so level two, the
// same selection over the slices' survivors in slice order, gives exactly the row's top-k, in ascending positions.
// Scratch (keys, per row): survivor keys [S x k], survivor positions [S x k], block-maximum keys [nb].
struct SelScratch { std::uint32_t *keys, *pos, *bmax; unsigned slices; };
template <class View>
__device__ __forceinline__ SelScratch sel_scratch(const View& v, unsigned r) {
    const unsigned slices = (v.max_count + kSlice - 1) / kSlice;
    std::uint32_t* base = v.keys + static_cast<std::size_t>(r) * v.max_count;
    return {base, base + static_cast<std::size_t>(slices) * v.topk, base + 2 * static_cast<std::size_t>(slices) * v.topk, slices};
}
// View::compact (candidate user): the slice keeps only its candidate positions' keys, in position order, and selects
// among them (padded with key 0 to k, as a short slice). The full slice holds the key of -inf at every other position.
// If every candidate key of the row is above -inf's and the row has more than k candidate positions, the row's top-k
// are candidates only, every one of them is in its slice's candidate top-k, and the merge over the survivors (keys,
// then positions in order) gives the same result. The slice records its candidate count and whether a candidate key is
// at or below -inf's (in the block-maximum scratch, unused by users); the merge checks both and otherwise runs the
// one-block selection over the whole row. Needs a block_size >= 8 dividing kSlice.
template <unsigned Threads, bool Legacy, class View>
__device__ void select_slice_compact(const View& v, unsigned s, std::uint32_t n, std::uint32_t t0, std::uint32_t m,
    const float* scores, const std::uint8_t* cand, const SelScratch& sc, std::uint32_t* skeys, unsigned (*hist)[256], unsigned* warp_tot) {
    __shared__ std::uint32_t boff[kSlice / 8U];
    const std::uint32_t bs = v.block_size, b0 = t0 / bs, nsb = (m + bs - 1) / bs, ninf_key = order_key(-__builtin_inff());
    const std::uint32_t b = b0 + threadIdx.x;
    const unsigned len = threadIdx.x < nsb && cand[b] != 0U ? (n - b * bs < bs ? n - b * bs : bs) : 0U;
    unsigned ms;
    const unsigned off = block_exclusive_scan<Threads>(len, warp_tot, ms);
    if (threadIdx.x < nsb) boff[threadIdx.x] = off;
    bool bad = false;
    for (unsigned j = 0; j < len; ++j) {
        const std::uint32_t key = order_key(scores[b * bs + j]);
        skeys[off + j] = key;
        bad = bad || key <= ninf_key;
    }
    bad = __syncthreads_or(bad ? 1 : 0) != 0;
    if (threadIdx.x == 0U) { sc.bmax[2U * s] = ms; sc.bmax[2U * s + 1U] = bad ? 1U : 0U; }
    if (n <= v.topk) return;
    const auto position = [&](std::uint32_t i) {   // the slice block holding compacted index i: the last with boff <= i
        std::uint32_t lo = 0, hi = nsb;
        while (hi - lo > 1U) { const std::uint32_t mid = (lo + hi) / 2U; if (boff[mid] <= i) lo = mid; else hi = mid; }
        return (b0 + lo) * bs + (i - boff[lo]);
    };
    std::uint32_t* ok = sc.keys + static_cast<std::size_t>(s) * v.topk;
    std::uint32_t* op = sc.pos + static_cast<std::size_t>(s) * v.topk;
    if (ms <= v.topk) {
        for (std::uint32_t i = threadIdx.x; i < v.topk; i += Threads) { ok[i] = i < ms ? skeys[i] : 0U; op[i] = i < ms ? position(i) : t0; }
        return;
    }
    std::uint32_t thr, need;
    select_threshold<Threads, Legacy>(skeys, ms, v.topk, hist, thr, need);
    select_ordered<Threads, Legacy>(skeys, ms, thr, need, warp_tot, [&](std::uint32_t i, bool sel, unsigned out) {
        if (sel) { ok[out] = skeys[i]; op[out] = position(i); }
    });
}
// Level one: block (slice s = bid.x, row r = bid.y).
template <unsigned Threads, bool Legacy, class View>
__device__ __forceinline__ void select_slice_body(const View& v, const uint3 bid) {
    using Pool = decltype(v.pool);
    __shared__ unsigned hist[Threads / 32][256];
    __shared__ unsigned warp_tot[33];
    __shared__ std::uint32_t skeys[kSlice];
    const unsigned s = bid.x, r = bid.y;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio), t0 = s * kSlice;
    if (t0 >= n) return;
    const std::uint32_t m = n - t0 < kSlice ? n - t0 : kSlice;
    const float* scores = v.scores + static_cast<std::size_t>(r) * v.max_count;
    const std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
    const SelScratch sc = sel_scratch(v, r);
    const float ninf = -__builtin_inff();
    if (v.compact && v.pool == Pool::candidate_user) {
        select_slice_compact<Threads, Legacy>(v, s, n, t0, m, scores, cand, sc, skeys, hist, warp_tot);
        return;
    }
    for (std::uint32_t i = threadIdx.x; i < m; i += Threads) {
        const std::uint32_t t = t0 + i;
        skeys[i] = order_key(v.pool == Pool::candidate_user && cand[t / v.block_size] == 0U ? ninf : scores[t]);
    }
    if (v.pool == Pool::candidate_source) {
        const std::uint32_t b0 = t0 / v.block_size, b1 = (t0 + m + v.block_size - 1) / v.block_size, last = (n - 1) / v.block_size;
        for (std::uint32_t b = b0 + threadIdx.x; b < b1; b += Threads) {
            float mx = ninf;
            for (std::uint32_t t = b * v.block_size; t < (b + 1) * v.block_size && t < n; ++t) mx = fmaxf(mx, scores[t]);
            sc.bmax[b] = order_key(b == last ? __builtin_inff() : mx);
        }
    }
    __syncthreads();
    if (n <= v.topk) return;   // level two takes every position
    std::uint32_t* ok = sc.keys + static_cast<std::size_t>(s) * v.topk;
    std::uint32_t* op = sc.pos + static_cast<std::size_t>(s) * v.topk;
    if (m <= v.topk) {
        for (std::uint32_t i = threadIdx.x; i < v.topk; i += Threads) { ok[i] = i < m ? skeys[i] : 0U; op[i] = t0 + i; }
        return;
    }
    std::uint32_t thr, need;
    select_threshold<Threads, Legacy>(skeys, m, v.topk, hist, thr, need);
    select_ordered<Threads, Legacy>(skeys, m, thr, need, warp_tot, [&](std::uint32_t i, bool sel, unsigned out) {
        if (sel) { ok[out] = skeys[i]; op[out] = t0 + i; }
    });
}
// Level two: block 0 of a row selects the top-k among the survivors; block 1 (candidate source only) the candidate
// blocks among the block maxima.
template <unsigned Threads, bool Legacy, class View>
__device__ __forceinline__ void select_merge_body(const View& v, const uint3 bid) {
    using Pool = decltype(v.pool);
    __shared__ unsigned hist[Threads / 32][256];
    __shared__ unsigned warp_tot[33];
    const unsigned r = bid.y;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio);
    const SelScratch sc = sel_scratch(v, r);
    if (bid.x == 1U) {
        if (n == 0U) return;
        std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
        const std::uint32_t nb = (n + v.block_size - 1) / v.block_size, ninf_key = order_key(-__builtin_inff());
        if (nb <= v.cand_blocks) {
            for (std::uint32_t b = threadIdx.x; b < nb; b += Threads) cand[b] = sc.bmax[b] > ninf_key ? 1U : 0U;
            return;
        }
        std::uint32_t thr, need;
        select_threshold<Threads, Legacy>(sc.bmax, nb, v.cand_blocks, hist, thr, need);
        select_ordered<Threads, Legacy>(sc.bmax, nb, thr, need, warp_tot, [&](std::uint32_t b, bool s, unsigned) {
            cand[b] = s && sc.bmax[b] > ninf_key ? 1U : 0U;
        });
        return;
    }
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(r) * v.topk;
    if (n <= v.topk) {
        for (std::uint32_t t = threadIdx.x; t < n; t += Threads) idx[t] = t;
        if (threadIdx.x == 0U) v.idx_count[r] = n;
        return;
    }
    if (v.compact && v.pool == Pool::candidate_user) {   // the compact slices' records (select_slice_compact)
        std::uint32_t cnt = 0, bad = 0;
        for (std::uint32_t k = threadIdx.x; k < (n + kSlice - 1) / kSlice; k += Threads) { cnt += sc.bmax[2U * k]; bad |= sc.bmax[2U * k + 1U]; }
        unsigned all_cnt;
        (void)block_exclusive_scan<Threads>(cnt, warp_tot, all_cnt);
        if (__syncthreads_or(static_cast<int>(bad)) != 0 || all_cnt <= v.topk) {
            select_core<Threads, Legacy>(v, make_uint3(r, 0U, 0U), hist, warp_tot);
            return;
        }
    }
    const std::uint32_t total = ((n + kSlice - 1) / kSlice) * v.topk;
    std::uint32_t thr, need;
    select_threshold<Threads, Legacy>(sc.keys, total, v.topk, hist, thr, need);
    select_ordered<Threads, Legacy>(sc.keys, total, thr, need, warp_tot, [&](std::uint32_t i, bool s, unsigned out) { if (s) idx[out] = sc.pos[i]; });
    if (threadIdx.x == 0U) v.idx_count[r] = v.topk;
}

// Every row's count <= topk: the selection of select_body without its scores (every position, in order) and, for the
// candidate source, its pool: with nb <= cand_blocks select_body takes every block whose maximum is above -inf, and the
// last block always (its key is +inf); a block's maximum of finite scores is finite, so every block of the row.
template <class View>
__device__ __forceinline__ void select_direct_body(const View& v, const uint3 bid) {
    using Pool = decltype(v.pool);
    const unsigned r = bid.x;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio);
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(r) * v.topk;
    for (std::uint32_t t = threadIdx.x; t < n; t += blockDim.x) idx[t] = t;
    if (threadIdx.x == 0U) v.idx_count[r] = n;
    if (v.pool != Pool::candidate_source) return;
    std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
    const std::uint32_t nb = (n + v.block_size - 1) / v.block_size;
    for (std::uint32_t b = threadIdx.x; b < nb; b += blockDim.x) cand[b] = 1U;
}

} // namespace strix::attn_core::select::threshold
