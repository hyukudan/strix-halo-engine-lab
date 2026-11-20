#pragma once
// Transposed halving of the warp xor butterfly (CUDA decode attention scores, attention.cu attn_scores_kernel).
//
// The exact decode contract (attention.hpp attention_decode) defines every score dot as: lane l sums its 8 dims, then the
// xor butterfly over offsets 16, 8, 4, 2, 1 (lane l adds lane l ^ off's partial at each step). A warp that reduces N
// independent dots at once does not need 5 shuffles per dot: at offset 16 a lane keeps half of its N partials and sends
// the other half, its partner (bit 16 opposite) keeps the opposite half, and each adds the partner's partial of the SAME
// dot: keep + partner == own + partner, the butterfly's pair sum (IEEE addition is commutative, so the bits are equal).
// Repeat at 8, 4, ... while the live count is even, then run the plain butterfly on the c = remaining(N) values left.
// Every dot keeps the butterfly's summation tree, so every score has the bits of the per-dot butterfly; the dot ends in
// the lanes whose halving bits select it (slot_index), replicated over the lane bits of the plain steps.
// Cost per dot (N = 12): 1.5 shuffles + 1.5 adds + 1.5 selects instead of 5 shuffles + 5 adds.
#include <cstdint>

#if defined(__CUDACC__)
#define STRIX_SCORE_REDUCE_HD __host__ __device__ __forceinline__
#define STRIX_SCORE_REDUCE_C __host__ __device__  // constexpr helpers, also called from device code
// reduce() calls the caller's shuffle functor: a device lambda in the kernel, a host functor in the host test.
#define STRIX_SCORE_REDUCE_EXEC_CHECK _Pragma("nv_exec_check_disable")
#define STRIX_SCORE_REDUCE_UNROLL _Pragma("unroll")
#else
#define STRIX_SCORE_REDUCE_HD inline
#define STRIX_SCORE_REDUCE_C
#define STRIX_SCORE_REDUCE_EXEC_CHECK
#define STRIX_SCORE_REDUCE_UNROLL
#endif

namespace strix::cuda_core::score_reduce {

constexpr int kSteps = 5;  // warp of 32 lanes: offsets 16, 8, 4, 2, 1
[[nodiscard]] STRIX_SCORE_REDUCE_C constexpr std::uint32_t offset(int step) { return 16u >> step; }

// Halving steps for n partials per lane: halve while the live count is even (at most kSteps).
[[nodiscard]] STRIX_SCORE_REDUCE_C constexpr int halvings(int n) {
    int s = 0;
    while (s < kSteps && n % 2 == 0) {
        n /= 2;
        ++s;
    }
    return s;
}
// Values per lane after the halvings (each then reduced by the plain butterfly over the remaining offsets).
[[nodiscard]] STRIX_SCORE_REDUCE_C constexpr int remaining(int n) { return n >> halvings(n); }
// Lanes that hold the same values at the end (they differ only in the bits of the plain steps): (lane & replica_mask) == 0
// picks one representative per group.
[[nodiscard]] STRIX_SCORE_REDUCE_C constexpr std::uint32_t replica_mask(int n) { return (32u >> halvings(n)) - 1u; }

// Dot index (0 .. n-1) held in slot i (i < remaining(n)) by `lane` after reduce<n>.
[[nodiscard]] STRIX_SCORE_REDUCE_C constexpr std::uint32_t slot_index(int n, std::uint32_t lane, int i) {
    std::uint32_t idx = 0;
    int h = n;
    for (int s = 0; s < halvings(n); ++s) {
        h /= 2;
        if ((lane & offset(s)) != 0u) idx += static_cast<std::uint32_t>(h);
    }
    return idx + static_cast<std::uint32_t>(i);
}

// Reduces v[0 .. N) (one partial of N independent dots per lane) in place; afterwards v[i], i < remaining(N), holds the
// complete dot slot_index(N, lane, i). xr(x, off) must return lane (lane ^ off)'s x of the same call (warp shuffle xor),
// add(a, b) the rounded sum (the contract's __fadd_rn). Every lane of the warp must call it together.
STRIX_SCORE_REDUCE_EXEC_CHECK
template <int N, int S = 0, class Xor, class Add>
STRIX_SCORE_REDUCE_HD void reduce(float (&v)[N], std::uint32_t lane, Xor& xr, Add& add) {
    if constexpr (S < kSteps) {
        constexpr std::uint32_t off = offset(S);
        if constexpr (S < halvings(N)) {
            constexpr int h = (N >> S) / 2;  // live values before this step: N >> S
            const bool up = (lane & off) != 0u;
STRIX_SCORE_REDUCE_UNROLL
            for (int i = 0; i < h; ++i) {
                const float send = up ? v[i] : v[i + h];
                const float keep = up ? v[i + h] : v[i];
                v[i] = add(keep, xr(send, off));
            }
        } else {
STRIX_SCORE_REDUCE_UNROLL
            for (int i = 0; i < remaining(N); ++i) v[i] = add(v[i], xr(v[i], off));
        }
        reduce<N, S + 1>(v, lane, xr, add);
    }
}

// Scores kernel form: P positions per step (1 or 2), N = P * GT dots = (position q, head g) at index q * GT + g. With
// P = 2 the first halving (offset 16, half = GT) splits the positions, so lane bit 16 is the position of every slot of a
// lane. P = 1 keeps today's live registers (the GT = 8 kernel has no room for 16 dots: __launch_bounds__(256, 2)).
template <int GT, int P>
struct ScoreMap {
    static constexpr int N = P * GT;
    static constexpr int C = remaining(N);
    static_assert(P == 1 || P == 2, "one or two positions per step");
    static_assert(P == 1 || (halvings(N) >= 1 && (N >> 1) == GT), "the offset-16 halving must split the two positions");
    [[nodiscard]] STRIX_SCORE_REDUCE_C static constexpr std::uint32_t position(std::uint32_t lane) {
        return P == 2 ? (lane >> 4) & 1u : 0u;
    }
    [[nodiscard]] STRIX_SCORE_REDUCE_C static constexpr std::uint32_t head(std::uint32_t lane, int i) {
        return slot_index(N, lane, i) - position(lane) * static_cast<std::uint32_t>(GT);
    }
    // A lane and its slot that hold head g: the source of g's per-head fold (after the position fold when P = 2).
    [[nodiscard]] STRIX_SCORE_REDUCE_C static constexpr std::uint32_t fold_lane(int g) {
        for (std::uint32_t l = 0; l < 32u; ++l)
            for (int i = 0; i < C; ++i)
                if (head(l, i) == static_cast<std::uint32_t>(g)) return l;
        return 32u;
    }
    [[nodiscard]] STRIX_SCORE_REDUCE_C static constexpr int fold_slot(int g) {
        for (std::uint32_t l = 0; l < 32u; ++l)
            for (int i = 0; i < C; ++i)
                if (head(l, i) == static_cast<std::uint32_t>(g)) return i;
        return -1;
    }
};
// Positions per step of the scores kernel: 2 while the 2 GT live dots fit its register budget (GT 4 / 6: 119 / 128
// registers, no spill, sm_120a), else 1 (GT 8 spills 104 B with 2).
[[nodiscard]] STRIX_SCORE_REDUCE_C constexpr int score_positions(int gt) { return gt <= 6 ? 2 : 1; }

}  // namespace strix::cuda_core::score_reduce

#undef STRIX_SCORE_REDUCE_HD
#undef STRIX_SCORE_REDUCE_C
#undef STRIX_SCORE_REDUCE_EXEC_CHECK
#undef STRIX_SCORE_REDUCE_UNROLL
