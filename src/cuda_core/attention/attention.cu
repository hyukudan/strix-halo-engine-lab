// GQA decode attention over a BF16 KV cache, rows 1..32 with causal masking inside the window. See attention.hpp.
// Three kernels (scores, values, combine) parallel over position chunks; the reduction order is defined per 256-position chunk.
#include "strix/cuda_core/attention.hpp"

#include <atomic>
#include <limits>
#include <mutex>
#include <string>

#include "../common.cuh"
#include "strix/cuda_core/attention_row_walk.hpp"
#include "strix/cuda_core/attention_score_reduce.hpp"

namespace strix::cuda_core {
namespace {


constexpr std::uint32_t CH = kAttnChunk;  // 256 positions per chunk
constexpr int D = 256;
// Longest row walk per kernel. Scores keep a running max per (row of the walk, warp, head) in shared memory (12 x 8 x 8
// FP32 = 3 KiB) and re-read q of a row (L1) when a warp switches rows. Values keep the weights of every (row, head) pair of
// the walk for the whole chunk in shared memory, transposed [position][pair] (256 x 36 FP32 = 36 KiB), and one accumulator
// per pair per thread: a values walk takes value_walk_rows(GT) rows (attention_row_walk.hpp). A kernel runs GT =
// attn_group_width(heads / kv_heads) query heads of a kv head as independent chains; heads past G compute on zero q /
// unused weights and are never stored.
constexpr std::uint32_t kScoreWalkRows = 12;
constexpr std::uint32_t kValuePairs = kValueWalkPairs;
// Loads in flight per warp (scores: K rows of positions p, p+8, ...) and per thread (values: V elements, double-buffered).
constexpr int kScoreBatch = 8;
constexpr int kValueBatch = 16;
template <int GT>
constexpr std::uint32_t kValueWalk = value_walk_rows(static_cast<std::uint32_t>(GT));

__device__ __forceinline__ void store_kv_head(const float* k, const float* v, __nv_bfloat16* kc,
                                             __nv_bfloat16* vc, std::uint32_t width,
                                             std::size_t src, std::size_t dst) {
    for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        kc[dst + i] = __float2bfloat16_rn(k[src + i]);
        vc[dst + i] = __float2bfloat16_rn(v[src + i]);
    }
}

template <bool SELF>
__global__ void kv_store_kernel(AttentionDims d, const float* k, const float* v, __nv_bfloat16* kc, __nv_bfloat16* vc,
                                const std::uint32_t* pos0p, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t r = blockIdx.y, h = blockIdx.x;
    const std::uint32_t pos = rm.pos ? rm.pos[r] : *pos0p + r;
    const std::size_t so = rm.slot ? rm.slot[r] * rm.slot_stride : 0;
    const std::size_t src = (static_cast<std::size_t>(r) * d.kv_heads + h) * d.head_dim;
    const std::uint32_t at = SELF ? rm.self[r] : pos;  // token tree: a leaf row's K/V index (strix/spec/tree_window.hpp)
    const std::size_t dst = so + (static_cast<std::size_t>(h) * d.max_positions + at) * d.head_dim;
    store_kv_head(k, v, kc, vc, d.head_dim, src, dst);
}

__global__ void kv_store_prepared_kernel(AttentionDims d, const float* k, const float* v, __nv_bfloat16* kc,
                                         __nv_bfloat16* vc, std::uint32_t pos0, std::uint32_t rows, bool wrapped) {
    const std::uint32_t head = blockIdx.x;
    for (std::uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const std::uint32_t pos = pos0 + row;
        const std::uint32_t slot = wrapped ? pos % d.max_positions : pos;
        const std::size_t src = (std::size_t{row} * d.kv_heads + head) * d.head_dim;
        const std::size_t dst = (std::size_t{slot} * d.kv_heads + head) * d.head_dim;
        store_kv_head(k, v, kc, vc, d.head_dim, src, dst);
    }
}

struct Scratch {
    float* scores;  // [rows][heads][max_positions]
    float* cmax;    // [rows][heads][maxchunks]
    float* dpart;   // [rows][heads][maxchunks]
    float* part;    // [rows][heads][maxchunks][D]
};
__host__ __device__ inline Scratch carve(float* base, const AttentionDims& d, std::uint32_t rows) {
    const std::size_t rh = static_cast<std::size_t>(rows) * d.heads, mc = attn_max_chunks(d);
    Scratch s;
    s.scores = base;
    s.cmax = s.scores + rh * d.max_positions;
    s.dpart = s.cmax + rh * mc;
    s.part = s.dpart + rh * mc;
    return s;
}


// Rows of the launch, read once per block. s_len[r] = positions attended by row r. A walk = rows r0 .. r0 + n - 1 of one
// cache slot (rows_per_walk of the active chunk count, never across a slot: attention_row_walk.hpp); s_walk[k] = first
// row of walk k in ascending order, s_walk[nwalks] = rows. Block-synchronised.
// SELF (token tree, RowMap::self): a leaf row (own K/V index off its position) starts a walk and the row after it starts
// the next one, so a leaf always walks alone (n == 1) and a shared walk never holds a leaf: the K/V row a walk loads at a
// position is then the right key for every row of the walk, and the kernels read a lone leaf's own key from rm.self.
template <bool SELF>
__device__ __forceinline__ std::uint32_t leaf_walk_starts(const RowMap& rm, std::uint32_t lane, std::uint32_t rows, std::uint32_t len) {
    if constexpr (SELF) {
        const std::uint32_t lv = __ballot_sync(0xffffffffu, lane < rows && rm.self[lane] != len - 1);
        return lv | (lv << 1);
    } else {
        return 0u;
    }
}
struct RowSet {
    std::uint32_t nwalks = 0, active = 0;  // active = chunks of the longest row
};
template <bool SELF = false>
__device__ __forceinline__ RowSet row_set(const std::uint32_t* pos0p, std::uint32_t rows, const RowMap& rm, const RowWalk& rw,
                                          std::uint32_t* s_len, std::uint32_t* s_walk) {
    const std::uint32_t lane = threadIdx.x & 31;
    std::uint32_t len = 0, slot = 0;
    if (lane < rows) {
        len = (rm.pos ? rm.pos[lane] : *pos0p + lane) + 1;
        slot = rm.slot ? rm.slot[lane] : 0u;
    }
    const std::uint32_t longest = __reduce_max_sync(0xffffffffu, len);
    const std::uint32_t active = (longest + CH - 1) / CH;
    const std::uint32_t per = rows_per_walk(rw, active);
    const std::uint32_t prev = __shfl_up_sync(0xffffffffu, slot, 1);
    const std::uint32_t starts = __ballot_sync(0xffffffffu, lane == 0 || slot != prev)  // first row of each cache slot
                                 | leaf_walk_starts<SELF>(rm, lane, rows, len);
    const std::uint32_t slot_start = 31u - static_cast<std::uint32_t>(__clz(static_cast<int>(starts & ((2u << lane) - 1u))));
    const std::uint32_t leads = __ballot_sync(0xffffffffu, lane < rows && walk_start(lane, slot_start, per));
    const std::uint32_t nwalks = static_cast<std::uint32_t>(__popc(static_cast<int>(leads)));
    if (threadIdx.x < 32) {
        if (lane < rows) s_len[lane] = len;
        if ((leads >> lane) & 1u) s_walk[__popc(static_cast<int>(leads & ((1u << lane) - 1u)))] = lane;
        if (lane == 0) s_walk[nwalks] = rows;
    }
    __syncthreads();
    RowSet rs;
    rs.nwalks = nwalks;
    rs.active = active;
    return rs;
}

// Work item = (chunk, kv head, walk). Shared walks (rw.adjacent = 1): items in a persistent loop over a grid of at most
// two resident waves, walk fastest, then kv head, then chunk, so the walks of one chunk run next to each other (a
// repeated K/V line is an L2 hit) and no block is launched without work. One row per block (rw.adjacent = 0): the grid
// (max chunks, kv heads, rows) of the one-row form, one item per block.
struct Walk {
    std::uint32_t chunk = 0, kvh = 0, r0 = 0, n = 0, lmax = 0;  // lmax = longest row of the walk
};
__device__ __forceinline__ std::uint32_t item_count(const RowWalk& rw, const RowSet& rs, std::uint32_t kv_heads) {
    return rw.adjacent ? rs.active * kv_heads * rs.nwalks : 1u;
}
__device__ __forceinline__ Walk item_walk(std::uint32_t item, const RowWalk& rw, const RowSet& rs, std::uint32_t kv_heads,
                                          const std::uint32_t* s_len, const std::uint32_t* s_walk) {
    Walk wk;
    std::uint32_t k = blockIdx.z;
    if (rw.adjacent) {
        const RowWalkItem ri = row_walk_item(item, rs.nwalks, kv_heads);
        k = ri.walk;
        wk.kvh = ri.kv_head;
        wk.chunk = ri.chunk;
    } else {
        wk.chunk = blockIdx.x;
        wk.kvh = blockIdx.y;
    }
    wk.r0 = s_walk[k];
    const std::uint32_t next = s_walk[k + 1];
    wk.n = next - wk.r0;
    for (std::uint32_t r = wk.r0; r < next; ++r) wk.lmax = max(wk.lmax, s_len[r]);
    return wk;
}

// Phase A: item = (chunk, kv head, rows r0 .. r0 + n - 1); warp w handles positions base + w + 8 i for the G query heads
// of the group and every row of the walk (a K row is loaded once per walk). dot: lane owns 8 consecutive dims
// (sequential), xor butterfly 16..1; score = fl(dot) * scale. The GT heads run as independent chains (same per-head
// operations and order). Per (row, head) a warp takes its positions in the same ascending order with the same running max
// as the one-row form; between load batches the running max of (row, warp) waits in shared memory and q of the next row is
// re-read (L1). The butterfly runs as the transposed halving over P positions (attention_score_reduce.hpp, same bits);
// the lane that ends with a dot stores it (one store per (head, position)).
template <int GT>
using ScorePair = score_reduce::ScoreMap<GT, score_reduce::score_positions(GT)>;
// mx[g] = fmaxf(mx[g], max of head g's slot over the step positions), read from the slot's fold lane (compile-time lane / slot).
template <int GT, int G0 = 0>
__device__ __forceinline__ void fold_heads(float (&mx)[GT], const float (&smx)[ScorePair<GT>::C]) {
    if constexpr (G0 < GT) {
        constexpr int slot = ScorePair<GT>::fold_slot(G0);
        constexpr std::uint32_t src = ScorePair<GT>::fold_lane(G0);
        static_assert(slot >= 0 && src < 32u, "every head has a fold source");
        mx[G0] = fmaxf(mx[G0], __shfl_sync(0xffffffffu, smx[slot], static_cast<int>(src)));
        fold_heads<GT, G0 + 1>(mx, smx);
    }
}
__device__ __forceinline__ float score_xor(float x, std::uint32_t off) { return __shfl_xor_sync(0xffffffffu, x, static_cast<int>(off)); }
__device__ __forceinline__ float score_add(float a, float b) { return __fadd_rn(a, b); }
template <int GT, bool SELF = false>
__global__ void __launch_bounds__(256, 2) attn_scores_kernel(AttentionDims d, const float* q, GroupedView qv, const __nv_bfloat16* kc,
                                                          const std::uint32_t* pos0p, std::uint32_t rows, Scratch sc, RowMap rm,
                                                          RowWalk rw, std::uint32_t qvec) {
    pdl_trigger();
    pdl_wait();
    __shared__ std::uint32_t s_len[kMaxRows], s_walk[kMaxRows + 1];
    __shared__ float wm[kScoreWalkRows][8][GT];  // running max per (row of the walk, warp, head)
    // Loads of kScoreBatch positions (p, p+8, ...) are issued together and the next batch is issued before the current one
    // is computed, so a warp waits for memory about once per batch. The batch is staged in shared memory (each lane reads
    // back only its own slots) so the position loop stays rolled.
    __shared__ uint4 kst[kScoreBatch][8][32];
    const RowSet rs = row_set<SELF>(pos0p, rows, rm, rw, s_len, s_walk);
    const std::uint32_t G = d.heads / d.kv_heads, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const std::uint32_t mc = attn_max_chunks(d), items = item_count(rw, rs, d.kv_heads);
    const float scale = d.scale;
    const std::size_t qrs = qv.rs(), qgs = qv.gs(), mp = d.max_positions;
    using PM = ScorePair<GT>;
    constexpr int P = score_reduce::score_positions(GT);  // positions per reduction step
    static_assert(kScoreBatch % P == 0, "a batch holds whole steps");
    // this lane's dots after the halving: position pq of the pair, head hs[i] of slot i; store_lane = replica representative
    const std::uint32_t pq = PM::position(lane);
    const bool store_lane = (lane & score_reduce::replica_mask(PM::N)) == 0u;
    std::uint32_t hs[PM::C];
#pragma unroll
    for (int i = 0; i < PM::C; ++i) hs[i] = PM::head(lane, i);
    for (std::uint32_t it = rw.adjacent ? blockIdx.x : 0u; it < items; it += rw.adjacent ? gridDim.x : 1u) {
        const Walk wk = item_walk(it, rw, rs, d.kv_heads, s_len, s_walk);
        const std::uint32_t base = wk.chunk * CH, kvh = wk.kvh, chunk = wk.chunk;
        if (base >= wk.lmax) continue;  // block-uniform
        const __nv_bfloat16* kb = kc + (rm.slot ? rm.slot[wk.r0] * rm.slot_stride : 0) + static_cast<std::size_t>(kvh) * mp * D + lane * 8;
        // SELF: a one-row walk reads its own key (position len - 1) at rm.self[r0] (a token-tree leaf's K/V index; a chain
        // row's index is its position); kp() is the identity otherwise.
        const std::uint32_t own_p = SELF && wk.n == 1u ? s_len[wk.r0] - 1u : 0xffffffffu, own = SELF ? rm.self[wk.r0] : 0u;
        const auto kp = [&](std::uint32_t p) { return SELF && p == own_p ? own : p; };
        for (std::uint32_t i = lane; i < wk.n * GT; i += 32u) wm[i / GT][w][i % GT] = __int_as_float(0xff800000);
        __syncwarp();
        float qr[GT][8], mx[GT];
        std::uint32_t cur = wk.n;  // row of the walk whose q and running max are in registers (none yet)
        float* srow = nullptr;     // scores of row cur, head kvh * G
        const std::uint32_t endw = min(wk.lmax, base + CH);
        uint4 u[kScoreBatch];
#pragma unroll
        for (int b = 0; b < kScoreBatch; ++b) {
            const std::uint32_t p = base + w + 8u * static_cast<std::uint32_t>(b);
            u[b] = p < endw ? __ldg(reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(kp(p)) * D)) : make_uint4(0, 0, 0, 0);
        }
        for (std::uint32_t p0 = base + w; p0 < endw; p0 += 8u * kScoreBatch) {
#pragma unroll
            for (int b = 0; b < kScoreBatch; ++b) kst[b][w][lane] = u[b];
            const std::uint32_t p1 = p0 + 8u * kScoreBatch;
#pragma unroll
            for (int b = 0; b < kScoreBatch; ++b) {
                const std::uint32_t p = p1 + 8u * static_cast<std::uint32_t>(b);
                u[b] = p < endw ? __ldg(reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(kp(p)) * D)) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll 1
            for (std::uint32_t j = 0; j < wk.n; ++j) {
                const std::uint32_t r = wk.r0 + j, end = min(s_len[r], base + CH);
                if (p0 >= end) continue;  // warp-uniform
                if (j != cur) {
                    if (cur < wk.n) {
#pragma unroll
                        for (int g = 0; g < GT; ++g) wm[cur][w][g] = mx[g];
                    }
                    const float* qp = q + static_cast<std::size_t>(r) * qrs + static_cast<std::size_t>(kvh) * G * qgs + lane * 8;
#pragma unroll
                    for (int g = 0; g < GT; ++g) {
                        if (static_cast<std::uint32_t>(g) < G) {
                            const float* qg = qp + static_cast<std::size_t>(g) * qgs;
                            if (qvec) {
                                const float4 a = *reinterpret_cast<const float4*>(qg), c = *reinterpret_cast<const float4*>(qg + 4);
                                qr[g][0] = a.x; qr[g][1] = a.y; qr[g][2] = a.z; qr[g][3] = a.w;
                                qr[g][4] = c.x; qr[g][5] = c.y; qr[g][6] = c.z; qr[g][7] = c.w;
                            } else {
#pragma unroll
                                for (int k = 0; k < 8; ++k) qr[g][k] = qg[k];
                            }
                        } else {
#pragma unroll
                            for (int k = 0; k < 8; ++k) qr[g][k] = 0.0f;
                        }
                    }
#pragma unroll
                    for (int g = 0; g < GT; ++g) mx[g] = wm[j][w][g];
                    cur = j;
                    srow = sc.scores + (static_cast<std::size_t>(r) * d.heads + static_cast<std::size_t>(kvh) * G) * mp;
                }
                // P positions per step (score_positions: 2 for GT <= 6, 1 for GT 8): the P GT dots (position q, head g;
                // index q * GT + g) go through the transposed butterfly halving (attention_score_reduce.hpp): each dot keeps
                // the contract's butterfly tree, so the same bits, with 1.1-1.5 shuffles per dot instead of 5. A lane then holds
                // PM::C complete dots (position pq, heads hs[i]); one lane per replica group stores them and keeps a running
                // max per slot, folded into the per-head max once per row and batch (fmaxf over the same scores).
                float smx[PM::C];
#pragma unroll
                for (int i = 0; i < PM::C; ++i) smx[i] = __int_as_float(0xff800000);
#pragma unroll 1
                for (std::uint32_t b = 0; b < static_cast<std::uint32_t>(kScoreBatch); b += static_cast<std::uint32_t>(P)) {
                    if (p0 + 8u * b >= end) break;  // warp-uniform
                    float dot[PM::N];
#pragma unroll
                    for (int q2 = 0; q2 < P; ++q2) {
                        const uint4 k4 = kst[b + static_cast<std::uint32_t>(q2)][w][lane];
                        const float kf[8] = {bf16_lo(k4.x), bf16_hi(k4.x), bf16_lo(k4.y), bf16_hi(k4.y),
                                             bf16_lo(k4.z), bf16_hi(k4.z), bf16_lo(k4.w), bf16_hi(k4.w)};
#pragma unroll
                        for (int g = 0; g < GT; ++g) dot[q2 * GT + g] = 0.0f;
#pragma unroll
                        for (int k = 0; k < 8; ++k)
#pragma unroll
                            for (int g = 0; g < GT; ++g)
                                dot[q2 * GT + g] = __fadd_rn(dot[q2 * GT + g], __fmul_rn(qr[g][k], kf[k]));
                    }
                    score_reduce::reduce<PM::N>(dot, lane, score_xor, score_add);
                    const std::uint32_t p = p0 + 8u * (b + pq);
#pragma unroll
                    for (int i = 0; i < PM::C; ++i) {
                        const float s = __fmul_rn(dot[i], scale);
                        if (p < end && hs[i] < G) {
                            smx[i] = fmaxf(smx[i], s);
                            if (store_lane) srow[static_cast<std::size_t>(hs[i]) * mp + p] = s;
                        }
                    }
                }
                if constexpr (P == 2) {  // the two positions of a slot sit in lanes l and l ^ 16
#pragma unroll
                    for (int i = 0; i < PM::C; ++i) smx[i] = fmaxf(smx[i], __shfl_xor_sync(0xffffffffu, smx[i], 16));
                }
                fold_heads<GT>(mx, smx);
            }
        }
        if (cur < wk.n) {
#pragma unroll
            for (int g = 0; g < GT; ++g) wm[cur][w][g] = mx[g];
        }
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < wk.n * G; i += blockDim.x) {
            const std::uint32_t j = i / G, g = i - j * G, r = wk.r0 + j;
            if (base >= s_len[r]) continue;  // the row ends before this chunk: nothing written (as the one-row form)
            float m = wm[j][0][g];
            for (int k = 1; k < 8; ++k) m = fmaxf(m, wm[j][k][g]);
            sc.cmax[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * mc + chunk] = m;
        }
        __syncthreads();  // wm of the next item
    }
}

// Phase C accumulation of a walk of n <= NR rows: thread t = output dim; acc[j][g] = sum over i ascending below lim[j] (the
// row's positions in this chunk) of fl(e[j][g][i] * v[i][t]), rounded mul then add, as the one-row form. The weights of
// position i for every pair of the walk are one row es[i][.] read as 128-bit broadcasts. V in batches of kValueBatch
// positions: the next batch is issued before the current one is accumulated; the batch is staged in shared memory (each
// thread reads back only its own column). Each V element is loaded once for the whole walk. Rows of a walk differ only in
// the causal tail of the last chunk: batches below every row's count skip the per-row check. Rows j >= n (a shorter walk
// on a wider instantiation) run on stale weights and are never stored.
// SELF (one-row walks only, NR == 1): chunk offset own_i (the row's own position) is read from vown (rm.self).
template <int GT, int NR, bool SELF = false>
__device__ __forceinline__ void values_accumulate(const AttentionDims& d, const __nv_bfloat16* vp, const float (*es)[kValuePairs],
                                                  const std::uint32_t* lim, std::uint32_t n, std::uint32_t G, std::uint32_t r0,
                                                  std::uint32_t kvh, std::uint32_t chunk, const Scratch& sc, __nv_bfloat16 (*vst)[CH],
                                                  std::uint32_t own_i = 0xffffffffu, const __nv_bfloat16* vown = nullptr) {
    static_assert(!SELF || NR == 1, "a token-tree leaf walks alone");
    const auto vat = [&](std::uint32_t i) { return SELF && i == own_i ? *vown : vp[static_cast<std::size_t>(i) * D]; };
    constexpr int M = GT * NR, MV = (M + 3) / 4;
    static_assert(MV * 4 <= static_cast<int>(kValuePairs), "walk weights exceed the shared row");
    const std::uint32_t t = threadIdx.x, mc = attn_max_chunks(d);
    std::uint32_t lr[NR], n_hi = 0, n_lo = CH;
#pragma unroll
    for (int j = 0; j < NR; ++j) {
        lr[j] = static_cast<std::uint32_t>(j) < n ? lim[j] : 0u;
        if (static_cast<std::uint32_t>(j) < n) {
            n_hi = max(n_hi, lr[j]);
            n_lo = min(n_lo, lr[j]);
        }
    }
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    const __nv_bfloat16 vzero = __ushort_as_bfloat16(static_cast<unsigned short>(0));
    __nv_bfloat16 u[kValueBatch];
#pragma unroll
    for (int b = 0; b < kValueBatch; ++b) u[b] = static_cast<std::uint32_t>(b) < n_hi ? vat(static_cast<std::uint32_t>(b)) : vzero;
    for (std::uint32_t i0 = 0; i0 < n_hi; i0 += kValueBatch) {
#pragma unroll
        for (int b = 0; b < kValueBatch; ++b) vst[b][t] = u[b];
        const std::uint32_t i1 = i0 + kValueBatch;
#pragma unroll
        for (int b = 0; b < kValueBatch; ++b) {
            const std::uint32_t i = i1 + static_cast<std::uint32_t>(b);
            u[b] = i < n_hi ? vat(i) : vzero;
        }
        const std::uint32_t nb = min(n_hi - i0, static_cast<std::uint32_t>(kValueBatch));
        if (i0 + nb <= n_lo) {
#pragma unroll 2
            for (std::uint32_t b = 0; b < nb; ++b) {
                const float vv = __bfloat162float(vst[b][t]);
                const float4* wr = reinterpret_cast<const float4*>(es[i0 + b]);
#pragma unroll
                for (int v = 0; v < MV; ++v) {
                    const float4 x = wr[v];
                    const float xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        if (4 * v + e < M) acc[4 * v + e] = __fadd_rn(acc[4 * v + e], __fmul_rn(xs[e], vv));
                }
            }
        } else {
#pragma unroll 1
            for (std::uint32_t b = 0; b < nb; ++b) {
                const float vv = __bfloat162float(vst[b][t]);
                const std::uint32_t i = i0 + b;
#pragma unroll
                for (int j = 0; j < NR; ++j) {
                    if (i >= lr[j]) continue;
#pragma unroll
                    for (int g = 0; g < GT; ++g) acc[j * GT + g] = __fadd_rn(acc[j * GT + g], __fmul_rn(es[i][j * GT + g], vv));
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NR; ++j) {
        if (lr[j] == 0) continue;  // the row ends before this chunk (or j >= n): nothing written
#pragma unroll
        for (int g = 0; g < GT; ++g)
            if (static_cast<std::uint32_t>(g) < G)
                sc.part[((static_cast<std::size_t>(r0 + static_cast<std::uint32_t>(j)) * d.heads + kvh * G + static_cast<std::uint32_t>(g)) * mc + chunk) * D + t] =
                    acc[j * GT + g];
    }
}

// Phase C: item = (chunk, kv head, rows r0 .. r0 + n - 1, n <= value_walk_rows(GT)); thread t = output dim and (for the
// weights) position base + t. Per (row, head): e_p = expf(s_p - m) with the exact global max m; chunk denominator = the
// block tree of block_sum256 (warp xor butterfly, then the 8 warp sums added in order 0..7; thread t holds position
// base + t), here run for every pair of the walk with one barrier; chunk value partial = sum over the chunk positions
// ascending of e_p * v[p][t] (separate rounded mul, add).
template <int GT, bool SELF = false>
__global__ void __launch_bounds__(256, 2) attn_values_kernel(AttentionDims d, const __nv_bfloat16* vc, const std::uint32_t* pos0p,
                                                          std::uint32_t rows, Scratch sc, RowMap rm, RowWalk rw) {
    pdl_trigger();
    pdl_wait();
    constexpr std::uint32_t WR = kValueWalk<GT>;
    __shared__ std::uint32_t s_len[kMaxRows], s_walk[kMaxRows + 1];
    __shared__ float mg[kValuePairs];
    __shared__ float red[kValuePairs][8];
    __shared__ std::uint32_t lim[WR];
    __shared__ __align__(16) float es[CH][kValuePairs];
    __shared__ __nv_bfloat16 vst[kValueBatch][CH];
    const RowSet rs = row_set<SELF>(pos0p, rows, rm, rw, s_len, s_walk);
    const std::uint32_t G = d.heads / d.kv_heads, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const std::uint32_t mc = attn_max_chunks(d), items = item_count(rw, rs, d.kv_heads);
    for (std::uint32_t it = rw.adjacent ? blockIdx.x : 0u; it < items; it += rw.adjacent ? gridDim.x : 1u) {
        const Walk wk = item_walk(it, rw, rs, d.kv_heads, s_len, s_walk);
        const std::uint32_t base = wk.chunk * CH, kvh = wk.kvh, chunk = wk.chunk, n = wk.n;
        if (base >= wk.lmax) continue;  // block-uniform
        const __nv_bfloat16* vcs = vc + (rm.slot ? rm.slot[wk.r0] * rm.slot_stride : 0);
        // exact global max of each (row, head) over the row's chunk maxima: one warp per (row, head); lane-strided maxima
        // in ascending order, then the xor butterfly (8 loads in flight; -inf padding leaves the max unchanged)
        for (std::uint32_t s = w; s < n * G; s += 8u) {
            const std::uint32_t j = s / G, g = s - j * G, r = wk.r0 + j, len = s_len[r];
            if (base >= len) continue;  // warp-uniform
            const std::uint32_t nch = (len + CH - 1) / CH;
            const float* cm = sc.cmax + (static_cast<std::size_t>(r) * d.heads + kvh * G + g) * mc;
            float m = __int_as_float(0xff800000);
            for (std::uint32_t c0 = lane; c0 < nch; c0 += 256u) {
                float v[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const std::uint32_t c = c0 + 32u * static_cast<std::uint32_t>(k);
                    v[k] = c < nch ? cm[c] : __int_as_float(0xff800000);
                }
#pragma unroll
                for (int k = 0; k < 8; ++k) m = fmaxf(m, v[k]);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
            if (lane == 0) mg[j * GT + g] = m;
        }
        if (t < n) {
            const std::uint32_t len = s_len[wk.r0 + t];
            lim[t] = base < len ? min(len - base, CH) : 0u;
        }
        __syncthreads();
        // weights of position p = base + t for every pair, and the warp stage of the chunk denominators
        const std::uint32_t p = base + t;
#pragma unroll 1
        for (std::uint32_t j = 0; j < n; ++j) {
            const std::uint32_t r = wk.r0 + j, len = s_len[r];
            if (base >= len) continue;  // block-uniform
            float e[GT];
#pragma unroll
            for (int g = 0; g < GT; ++g)
                e[g] = (static_cast<std::uint32_t>(g) < G && p < len)
                           ? sc.scores[(static_cast<std::size_t>(r) * d.heads + kvh * G + static_cast<std::uint32_t>(g)) * d.max_positions + p]
                           : 0.0f;
#pragma unroll
            for (int g = 0; g < GT; ++g) {
                if (static_cast<std::uint32_t>(g) < G && p < len) e[g] = cc_expf(__fsub_rn(e[g], mg[j * GT + static_cast<std::uint32_t>(g)]));
                es[t][j * GT + static_cast<std::uint32_t>(g)] = e[g];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
#pragma unroll
                for (int g = 0; g < GT; ++g) e[g] = __fadd_rn(e[g], __shfl_xor_sync(0xffffffffu, e[g], off));
            if (lane == 0) {
#pragma unroll
                for (int g = 0; g < GT; ++g) red[j * GT + static_cast<std::uint32_t>(g)][w] = e[g];
            }
        }
        __syncthreads();
        // block stage of the chunk denominators: thread = pair (j, g), warp sums 0..7 in order (as block_sum256)
        if (t < n * GT) {
            const std::uint32_t j = t / GT, g = t - j * GT, r = wk.r0 + j;
            if (g < G && base < s_len[r]) {
                float ds = red[t][0];
#pragma unroll
                for (int i = 1; i < 8; ++i) ds = __fadd_rn(ds, red[t][i]);
                sc.dpart[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * mc + chunk] = ds;
            }
        }
        const __nv_bfloat16* vp = vcs + (static_cast<std::size_t>(kvh) * d.max_positions + base) * D + t;
        if constexpr (SELF) {
            if (n <= 1) {  // token tree: the row's own value (chunk offset len - 1 - base) at rm.self[r0]
                const std::uint32_t own_p = s_len[wk.r0] - 1u;
                const __nv_bfloat16* vown = vcs + (static_cast<std::size_t>(kvh) * d.max_positions + rm.self[wk.r0]) * D + t;
                values_accumulate<GT, 1, true>(d, vp, es, lim, n, G, wk.r0, kvh, chunk, sc, vst, own_p - base, vown);
                __syncthreads();  // es / red / lim / vst of the next item
                continue;
            }
        }
        if (n <= 1) values_accumulate<GT, 1>(d, vp, es, lim, n, G, wk.r0, kvh, chunk, sc, vst);
        else if (n <= 2) values_accumulate<GT, 2>(d, vp, es, lim, n, G, wk.r0, kvh, chunk, sc, vst);
        else if (n <= 3 || WR < 6) values_accumulate<GT, 3>(d, vp, es, lim, n, G, wk.r0, kvh, chunk, sc, vst);
        else if constexpr (WR >= 6) values_accumulate<GT, 6>(d, vp, es, lim, n, G, wk.r0, kvh, chunk, sc, vst);
        __syncthreads();  // es / red / lim / vst of the next item
    }
}

// Phase D: block (head, row), thread = dim. out = (sum over chunks ascending of the partials) / (sum of chunk denominators
// ascending), times sigmoid(gate).
__global__ void __launch_bounds__(256) attn_combine_kernel(AttentionDims d, const float* gate, GroupedView gv, const std::uint32_t* pos0p,
                                                           float* out, Scratch sc, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const std::uint32_t len = (rm.pos ? rm.pos[r] : *pos0p + r) + 1, nch = (len + CH - 1) / CH, mc = attn_max_chunks(d);
    const std::size_t rh = static_cast<std::size_t>(r) * d.heads + h;
    float denom = 0.0f, acc = 0.0f;
    for (std::uint32_t c = 0; c < nch; ++c) {
        denom = __fadd_rn(denom, sc.dpart[rh * mc + c]);
        acc = __fadd_rn(acc, sc.part[(rh * mc + c) * D + t]);
    }
    float o = __fdiv_rn(acc, denom);
    if (gate != nullptr) o = __fmul_rn(o, cc_sigmoid(gate[static_cast<std::size_t>(r) * gv.rs() + static_cast<std::size_t>(h) * gv.gs() + t]));
    out[rh * D + t] = o;
}

// Occupancy floor of the shared plan and size of the shared-walk grid: two waves of resident blocks of `kernel` on the
// current device (SM count and the kernel's real occupancy, read once per device and kernel). 0 = unknown: the caller keeps
// one row per block.
constexpr int kWaveKernels = 12;  // scores / values x group widths 4 / 6 / 8 x SELF (token tree)
std::uint32_t two_waves(const void* kernel, int which) {
    constexpr int kDevices = 64;
    static std::atomic<std::uint32_t> cache[kDevices][kWaveKernels];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kDevices) {
        (void)cudaGetLastError();
        return 0;
    }
    std::uint32_t v = cache[dev][which].load(std::memory_order_relaxed);
    if (v != 0) return v;
    int sms = 0, per_sm = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 256, 0) != cudaSuccess || sms <= 0 || per_sm <= 0) {
        (void)cudaGetLastError();
        return 0;
    }
    v = 2u * static_cast<std::uint32_t>(sms) * static_cast<std::uint32_t>(per_sm);
    cache[dev][which].store(v, std::memory_order_relaxed);
    return v;
}

// The shared plan's measured table of the current device and this head shape (attn_row_share::dense_decode_tables),
// nullptr when either was not measured or the device is unreadable. Device key read once per device.
const attn_row_share::DeviceTable* row_share_table(const AttentionDims& d) {
    constexpr int kDevices = 64;
    static std::once_flag once[kDevices];
    static std::string arch[kDevices];
    static unsigned sms[kDevices];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kDevices) {
        (void)cudaGetLastError();
        return nullptr;
    }
    std::call_once(once[dev], [dev] {
        int major = 0, minor = 0, n = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || major <= 0 || n <= 0) {
            (void)cudaGetLastError();
            return;  // no key: no table
        }
        arch[dev] = cuda_arch_key(major, minor);
        sms[dev] = static_cast<unsigned>(n);
    });
    if (d.kv_heads == 0 || d.heads % d.kv_heads != 0) return nullptr;
    return attn_row_share::find_table(arch[dev], sms[dev], d.heads / d.kv_heads, d.kv_heads, d.head_dim);
}

constexpr std::uint8_t kRowShareUnset = 0xffu;  // not set yet: the first decode_row_share() reads the switch
std::atomic<std::uint8_t> g_row_share{kRowShareUnset};

template <int GT, bool SELF>
cudaError_t decode_attention(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                             const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, const std::uint32_t* pos0, std::uint32_t rows,
                             float* out, const Scratch& sc, cudaStream_t stream, const RowMap& rm, int which) {
    const std::uint32_t mc = attn_max_chunks(d);
    // q rows of 8 floats per lane as two 128-bit loads when every (row, head) start is 16-byte aligned
    const std::uint32_t qvec = (reinterpret_cast<std::uintptr_t>(q) % 16u == 0 && qv.rs() % 4u == 0 && qv.gs() % 4u == 0) ? 1u : 0u;
    // Row walks (same bits, attention_row_walk.hpp): the step tables of the shared plan for this launch shape, capped by
    // the device's measured table (none: one row per block); shared walks run as a persistent item loop on a grid of two
    // resident waves (no block without work).
    RowWalk rs, rv;
    dim3 gs(mc, d.kv_heads, rows), gvd(mc, d.kv_heads, rows);
    const DecodeRowShare mode = rows > 1 ? decode_row_share() : DecodeRowShare::off;
    const attn_row_share::DeviceTable* table = mode == DecodeRowShare::automatic ? row_share_table(d) : nullptr;
    const std::uint32_t cs = walk_cap(mode, table, rows, kScoreWalkRows), cv = walk_cap(mode, table, rows, value_walk_rows(GT));
    if (cs > 1 && cv > 1) {
        const int ws = which + (SELF ? 6 : 0);  // the SELF instantiations have their own occupancy
        const std::uint32_t fs = two_waves(reinterpret_cast<const void*>(attn_scores_kernel<GT, SELF>), ws);
        const std::uint32_t fv = two_waves(reinterpret_cast<const void*>(attn_values_kernel<GT, SELF>), ws + 3);
        if (fs != 0 && fv != 0) {
            rs = row_walk(mode, rows, d.kv_heads, mc, cs, fs);
            rv = row_walk(mode, rows, d.kv_heads, mc, cv, fv);
            const std::uint32_t cap = mc * d.kv_heads * rows;
            if (rs.adjacent) gs = dim3(min(cap, fs));
            if (rv.adjacent) gvd = dim3(min(cap, fv));
        }
    }
    if (auto e = launch_pdl(attn_scores_kernel<GT, SELF>, gs, dim3(256), stream, d, q, qv, k_cache, pos0, rows, sc, rm, rs, qvec); e != cudaSuccess)
        return e;
    if (auto e = launch_pdl(attn_values_kernel<GT, SELF>, gvd, dim3(256), stream, d, v_cache, pos0, rows, sc, rm, rv); e != cudaSuccess)
        return e;
    return launch_pdl(attn_combine_kernel, dim3(d.heads, rows), dim3(256), stream, d, gate, gv, pos0, out, sc, rm);
}

}  // namespace

void set_decode_row_share(DecodeRowShare mode) { g_row_share.store(static_cast<std::uint8_t>(mode), std::memory_order_relaxed); }
DecodeRowShare decode_row_share() {
    std::uint8_t v = g_row_share.load(std::memory_order_relaxed);
    if (v == kRowShareUnset) {
        const auto want = static_cast<std::uint8_t>(row_share_from_env());
        // a set_decode_row_share() that raced ahead keeps its mode
        v = g_row_share.compare_exchange_strong(v, want, std::memory_order_relaxed) ? want : v;
    }
    return static_cast<DecodeRowShare>(v);
}

cudaError_t kv_store(const AttentionDims& d, const float* k, const float* v, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                     const std::uint32_t* pos0, std::uint32_t rows, cudaStream_t stream, const RowMap& rm) {
    if (rows == 0) return cudaErrorInvalidValue;
    return launch_pdl(rm.self ? kv_store_kernel<true> : kv_store_kernel<false>, dim3(d.kv_heads, rows), dim3(256), stream, d, k, v, k_cache, v_cache, pos0, rm);
}

cudaError_t kv_store_prepared_bf16(const AttentionDims& d, const float* k, const float* v,
                                   __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, std::uint32_t pos0,
                                   std::uint32_t rows, bool wrapped, cudaStream_t stream) {
    constexpr std::uint32_t max_grid_rows = 65535U;
    constexpr auto limit = std::numeric_limits<std::uint32_t>::max() - max_grid_rows;
    if (!k || !v || !k_cache || !v_cache || k_cache == v_cache || d.kv_heads == 0 || d.head_dim == 0 ||
        rows == 0 || pos0 > limit || rows > limit - pos0 || (wrapped ? rows > d.max_positions
                                                                  : pos0 + rows > d.max_positions)) {
        return cudaErrorInvalidValue;
    }
    const auto grid_rows = rows < max_grid_rows ? rows : max_grid_rows;
    const auto threads = d.head_dim < 256U ? d.head_dim : 256U;
    kv_store_prepared_kernel<<<dim3(d.kv_heads, grid_rows), threads, 0, stream>>>(
        d, k, v, k_cache, v_cache, pos0, rows, wrapped);
    return cudaGetLastError();
}

cudaError_t attention_decode(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                             const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                             const std::uint32_t* pos0, std::uint32_t rows, float* out, float* scratch, cudaStream_t stream, const RowMap& rm) {
    if (d.head_dim != 256 || rows == 0 || rows > kMaxRows || d.kv_heads == 0 || d.heads % d.kv_heads != 0 || d.heads / d.kv_heads > 8)
        return cudaErrorInvalidValue;
    const Scratch sc = carve(scratch, d, rows);
    if (rm.self != nullptr) {  // token tree (RowMap::self): leaf rows walk alone and read their own K/V there
        switch (attn_group_width(d.heads / d.kv_heads)) {
            case 4: return decode_attention<4, true>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, sc, stream, rm, 0);
            case 6: return decode_attention<6, true>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, sc, stream, rm, 1);
            default: return decode_attention<8, true>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, sc, stream, rm, 2);
        }
    }
    switch (attn_group_width(d.heads / d.kv_heads)) {
        case 4: return decode_attention<4, false>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, sc, stream, rm, 0);
        case 6: return decode_attention<6, false>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, sc, stream, rm, 1);
        default: return decode_attention<8, false>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, sc, stream, rm, 2);
    }
}

namespace {
__global__ void kv_move_row_kernel(AttentionDims d, KvLayerCaches c, std::uint32_t from, std::uint32_t to) {
    const std::uint32_t h = blockIdx.x, l = blockIdx.y;
    const std::size_t src = (static_cast<std::size_t>(h) * d.max_positions + from) * d.head_dim;
    const std::size_t dst = (static_cast<std::size_t>(h) * d.max_positions + to) * d.head_dim;
    for (std::uint32_t i = threadIdx.x; i < d.head_dim; i += blockDim.x) {
        c.k[l][dst + i] = c.k[l][src + i];
        c.v[l][dst + i] = c.v[l][src + i];
    }
}
}  // namespace

cudaError_t kv_move_row(const AttentionDims& d, const KvLayerCaches& caches, std::uint32_t layers, std::uint32_t from, std::uint32_t to,
                        cudaStream_t stream) {
    if (layers > kMaxKvMoveLayers || from >= d.max_positions || to >= d.max_positions) return cudaErrorInvalidValue;
    if (layers == 0 || from == to) return cudaSuccess;
    kv_move_row_kernel<<<dim3(d.kv_heads, layers), dim3(256), 0, stream>>>(d, caches, from, to);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
