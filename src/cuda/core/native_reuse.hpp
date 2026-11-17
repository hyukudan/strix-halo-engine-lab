// Shared contract of the native SIMT expert route (arithmetic id cuda-native-iq-fp32@1), host and device.
//
// The per-output arithmetic DAG of src/cuda/core/moe_a8.cuh native_project_kernel / native_down_kernel, written once as
// executable code so that (a) a CPU oracle can reproduce the device bits, and (b) a kernel that reuses decoded weights
// (decode once per row tile instead of once per 8-row chunk) or blocks several (row, output) items per thread can prove
// it keeps the DAG. Nothing here changes arithmetic, storage formats, packs or defaults.
//
// Per (assignment a, output o) of one 32-value-grouped matrix row (width = 32 * groups):
//   leaf(g)  = fma chain from +0 over j = 0..31 ascending: p = fma(w[32 g + j], x[32 g + j], p)        (group_chain)
//   lane(l)  = LaneSeed::direct      : leaf(l) for l < groups, +0 otherwise (groups <= 32; down, ffn 640)
//              LaneSeed::zero_seeded : ((+0 + leaf(l)) + leaf(l + 32)) + ...  for l < groups      (project, hidden 2560)
//   root     = xor butterfly over the 32 lanes at distances 16, 8, 4, 2, 1, each lane forming v[l] + v[l ^ d]
// The two seeds differ only for a -0 leaf (+0 + -0 = +0); a leaf is -0 only through underflow of every product, but a
// kernel must still keep the seed of the route it replaces.
//
// Decoded-weight reuse: the decoded group w[32 g .. 32 g + 31] of (expert, o) is a pure function of the row bytes and g
// (quant::decode_row_group), independent of the assignment. Decoding it once per tile and applying it to every
// assignment of the tile leaves every leaf unchanged. Blocking several items per thread is exact as long as each
// item's leaf chain, lane order and butterfly tree are those above; transpose_fold_* below is the register form of the
// butterfly (the mechanism of src/cuda/models/qwen/qwen_hc.cu fold_level) and gives every item the root bits at its
// owner lane (IEEE addition is commutative; NaN-ness is preserved, NaN payloads are canonical on the device).
#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define STRIX_NATIVE_REUSE_HD __host__ __device__ __forceinline__
#else
#define STRIX_NATIVE_REUSE_HD inline
#endif

namespace strix::cuda::core::native_reuse {

STRIX_NATIVE_REUSE_HD float fma_rn(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
STRIX_NATIVE_REUSE_HD float add_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    return a + b;  // binary32 on SSE2 / NEON hosts; the oracle is built without contraction or fast-math
#endif
}

// Leaf of one 32-value group: w = the decoded group, x(j) = FP32 value of input element j of the group.
template <class X> STRIX_NATIVE_REUSE_HD float group_chain(const float* w, X&& x) {
    float p = 0.0F;
    for (unsigned j = 0; j < 32U; ++j) p = fma_rn(w[j], x(j), p);
    return p;
}

enum class LaneSeed : std::uint8_t { direct, zero_seeded };

// Lane value of lane l: leaf(g) is called for the lane's groups in ascending order.
template <class Leaf> STRIX_NATIVE_REUSE_HD float lane_value(LaneSeed seed, unsigned lane, unsigned groups, Leaf&& leaf) {
    if (lane >= groups) return 0.0F;
    if (seed == LaneSeed::direct) return leaf(lane);  // requires groups <= 32
    float v = 0.0F;
    for (unsigned g = lane; g < groups; g += 32U) v = add_rn(v, leaf(g));
    return v;
}

// Root of the xor butterfly as lane 0 forms it (operand order of every lane kept, so host NaN payloads match too).
STRIX_NATIVE_REUSE_HD float butterfly_root(const float (&v)[32]) {
    float a[32], b[32];
    for (unsigned l = 0; l < 32U; ++l) a[l] = v[l];
    for (unsigned d = 16U; d; d >>= 1U) {
        for (unsigned l = 0; l < 32U; ++l) b[l] = add_rn(a[l], a[l ^ d]);
        for (unsigned l = 0; l < 32U; ++l) a[l] = b[l];
    }
    return a[0];
}

// Tree order: the lane-0 butterfly root is the balanced binary tree whose bottom level pairs lane positions l and l ^ 16,
// the next l ^ 8, ... and the top l ^ 1, each pair with the lower position first. Visiting the positions in 5-bit
// bit-reversed order m -> tree_position(m) = 0, 16, 8, 24, 4, ... and merging every completed pair of equal-level partials
// (older first) reproduces exactly those additions with at most five live partials (TreeCounter). So one thread can own a
// whole (assignment, output) item over the full K range, K staged one lane position at a time, with no shuffles.
STRIX_NATIVE_REUSE_HD constexpr unsigned tree_position(unsigned m) {
    return ((m & 1U) << 4) | ((m & 2U) << 2) | (m & 4U) | ((m & 8U) >> 2) | ((m & 16U) >> 4);
}
// Groups of lane position l, applied in ascending order g = l + 32 t (0: the lane's value is the literal +0).
STRIX_NATIVE_REUSE_HD constexpr unsigned position_groups(LaneSeed seed, unsigned l, unsigned groups) {
    return l >= groups ? 0U : (seed == LaneSeed::direct ? 1U : (groups - l + 31U) / 32U);
}
// The data stage after stage (m, t) of the tree-ordered K schedule (one 32-value group per stage, every group once):
// returns false after the last; otherwise sets next_m / next_group.
STRIX_NATIVE_REUSE_HD bool next_stage(LaneSeed seed, unsigned groups, unsigned m, unsigned t, unsigned& next_m, unsigned& next_group) {
    const unsigned l = tree_position(m);
    if (t + 1U < position_groups(seed, l, groups)) { next_m = m; next_group = l + 32U * (t + 1U); return true; }
    for (unsigned n = m + 1U; n < 32U; ++n)
        if (position_groups(seed, tree_position(n), groups) != 0U) { next_m = n; next_group = tree_position(n); return true; }
    return false;
}
// Thread -> item map of moe_a8.cuh native_project_kernel (K = 2 outputs per matrix) and native_down_kernel (K = 4): 256
// threads over a 64-row tile, warp w owns rows 8 w .. 8 w + 7 (warp-uniform skip of rows past the tile), lane = 2 og + rh,
// a thread owns rows row(t, i) = 8 w + 4 rh + i (i < 4) and outputs output(t, k) = og + 16 k (k < K). Per 4 k-values a
// thread reads 4 X rows and K (x matrices) W rows with one LDS.128 each and issues 16 K (x matrices) FMAs: every shared
// operand feeds 2 K or 4 FMAs instead of 2 (the 1024-thread map of 2 rows x 2 items). Inside a quarter warp (8 lanes) the
// X reads touch 2 rows and the W reads 4 consecutive outputs: distinct 16-byte bank quads at pitch 36 (host test).
template <unsigned K> struct MicroTile {
    static constexpr unsigned threads = 256U, rows = 4U, outputs = K, tile_rows = 64U, tile_outputs = 16U * K;
    STRIX_NATIVE_REUSE_HD static constexpr unsigned row(unsigned t, unsigned i) { return (t >> 5) * 8U + (t & 1U) * 4U + i; }
    STRIX_NATIVE_REUSE_HD static constexpr unsigned output(unsigned t, unsigned k) { return ((t & 31U) >> 1) + 16U * k; }
};

// Binary counter over N items: push(m, v) with m = 0..31 in order; after push(31, v), v holds the root of every item,
// bitwise butterfly_root of the 32 lane values (host test). Levels 0..2 change every 1, 2, 4 pushes and use constant slot
// indices (registers in a kernel); levels 3 and 4 are touched only at m = 7, 15, 23, 31, so a kernel instantiates Deep =
// volatile float to keep them in thread-local memory (24 accesses per item set) instead of the register budget. The branches
// depend only on m (warp-uniform). Mid = volatile float also moves level 2 (touched at m % 4 == 3, 8 of 32 pushes) out of
// the register budget, for a kernel whose items fill it (native_project_kernel: 16 items x chain + leaf + levels 0..1).
// The additions and their order do not depend on Deep or Mid.
template <unsigned N, class Deep = float, class Mid = float> struct TreeCounter {
    float s[2][N];
    Mid s2[N];
    Deep d[2][N];
    STRIX_NATIVE_REUSE_HD static void fold(const float (&older)[N], float (&v)[N]) {
        for (unsigned i = 0; i < N; ++i) v[i] = add_rn(older[i], v[i]);
    }
    STRIX_NATIVE_REUSE_HD static void keep(float (&slot)[N], const float (&v)[N]) {
        for (unsigned i = 0; i < N; ++i) slot[i] = v[i];
    }
    STRIX_NATIVE_REUSE_HD void push(unsigned m, float (&v)[N]) {
        if (!(m & 1U)) { keep(s[0], v); return; }
        fold(s[0], v);
        if (!(m & 2U)) { keep(s[1], v); return; }
        fold(s[1], v);
        if (!(m & 4U)) { for (unsigned i = 0; i < N; ++i) s2[i] = v[i]; return; }
        for (unsigned i = 0; i < N; ++i) v[i] = add_rn(float(s2[i]), v[i]);
        if (!(m & 8U)) { for (unsigned i = 0; i < N; ++i) d[0][i] = v[i]; return; }
        for (unsigned i = 0; i < N; ++i) v[i] = add_rn(float(d[0][i]), v[i]);
        if (!(m & 16U)) { for (unsigned i = 0; i < N; ++i) d[1][i] = v[i]; return; }
        for (unsigned i = 0; i < N; ++i) v[i] = add_rn(float(d[1][i]), v[i]);
    }
};

// Register transpose-fold of N items per lane (N a power of two). Level k (distance 16 >> k) halves the live items while
// N >> k >= 2: a lane keeps the half selected by its lane bit and adds its partner's value of the same items (own value
// first, as the butterfly). Once one item is live the remaining levels are plain butterfly levels. After five levels
// item i is finished at owner_lane(i) in slot owner_slot(i).
template <unsigned N> struct TransposeFold {
    static_assert(N && !(N & (N - 1U)), "items per lane must be a power of two");
    STRIX_NATIVE_REUSE_HD static constexpr unsigned folding_levels() { unsigned k = 0; while (k < 5U && (N >> k) >= 2U) ++k; return k; }
    STRIX_NATIVE_REUSE_HD static constexpr unsigned live_items() { return N >> folding_levels(); }
    // Lane bit 4 - k selects item bit log2(N) - 1 - k at folding level k; lane bits below the folding levels are free.
    STRIX_NATIVE_REUSE_HD static constexpr unsigned owner_lane(unsigned item) {
        unsigned lane = 0;
        for (unsigned k = 0; k < folding_levels(); ++k) {
            const unsigned item_bit = (item >> (log2n() - 1U - k)) & 1U;
            lane |= item_bit << (4U - k);
        }
        return lane;  // any lane with these high bits holds the item; the lowest is returned
    }
    STRIX_NATIVE_REUSE_HD static constexpr unsigned owner_slot(unsigned item) { return item & (live_items() - 1U); }
    // Shuffle-adds per lane: sum of the halves, then one per plain level.
    STRIX_NATIVE_REUSE_HD static constexpr unsigned shuffles() {
        unsigned s = 0;
        for (unsigned k = 0; k < folding_levels(); ++k) s += N >> (k + 1U);
        return s + (5U - folding_levels()) * live_items();
    }
    STRIX_NATIVE_REUSE_HD static constexpr unsigned log2n() { unsigned k = 0; while ((1U << k) < N) ++k; return k; }
};

// Host simulation of the fold over 32 lanes (the oracle of a device fold; acc[l][i] = item i of lane l).
template <unsigned N> inline void transpose_fold_simulate(float (&acc)[32][N]) {
    using F = TransposeFold<N>;
    unsigned live = N;
    for (unsigned k = 0; k < 5U; ++k) {
        const unsigned d = 16U >> k;
        float next[32][N];
        if (k < F::folding_levels()) {
            const unsigned half = live / 2U;
            for (unsigned l = 0; l < 32U; ++l) {
                const bool upper = (l & d) != 0U;
                const unsigned p = l ^ d;
                for (unsigned i = 0; i < half; ++i) {
                    const float keep = upper ? acc[l][i + half] : acc[l][i];
                    const float send = (p & d) ? acc[p][i] : acc[p][i + half];  // partner sends the half the lane keeps
                    next[l][i] = add_rn(keep, send);
                }
            }
            live = half;
        } else {
            for (unsigned l = 0; l < 32U; ++l)
                for (unsigned i = 0; i < live; ++i) next[l][i] = add_rn(acc[l][i], acc[l ^ d][i]);
        }
        for (unsigned l = 0; l < 32U; ++l)
            for (unsigned i = 0; i < live; ++i) acc[l][i] = next[l][i];
    }
}

// Resource bound of one decoded-tile mapping (planning numbers, checked by static_assert in a kernel that adopts it).
// A block covers `outputs` outputs of `matrices` matrices (project 2, down 1); a thread blocks `rows_per_thread` x
// `items_per_row` (row, output-matrix) items. Weights resident in shared memory per (output, matrix, group) as FP32
// (128 B) or as a code view (32 int8 codes + 2 FP32 sub-scales = 40 B; its exactness for the six native GSQ storage
// types is checked by tests/cuda/native_reuse_host_test.cpp). Inputs staged as x_groups pitch-36 groups per row.
struct TileBudget {
    unsigned outputs, matrices, w_groups, w_group_bytes, rows_per_pass, x_groups, x_bytes, rows_per_thread, items_per_row;
    bool zero_seeded;
    STRIX_NATIVE_REUSE_HD constexpr unsigned items() const { return rows_per_thread * items_per_row; }
    // chain + lane partial per item (zero_seeded), chain only (direct); plus the live x and w operands of one j step.
    STRIX_NATIVE_REUSE_HD constexpr unsigned accumulator_regs() const { return items() * (zero_seeded ? 2U : 1U) + rows_per_thread + items_per_row; }
    STRIX_NATIVE_REUSE_HD constexpr unsigned weight_smem() const { return outputs * matrices * w_groups * w_group_bytes; }
    STRIX_NATIVE_REUSE_HD constexpr unsigned x_smem() const { return rows_per_pass * x_groups * 36U * x_bytes; }
    STRIX_NATIVE_REUSE_HD constexpr unsigned smem() const { return weight_smem() + x_smem(); }
    // Shared-memory bytes read per FMA: rows_per_thread x values and items_per_row weights feed items() FMAs.
    STRIX_NATIVE_REUSE_HD constexpr double smem_bytes_per_fma(unsigned w_value_bytes) const {
        return double(rows_per_thread * x_bytes + items_per_row * w_value_bytes) / double(items());
    }
};

}  // namespace strix::cuda::core::native_reuse
