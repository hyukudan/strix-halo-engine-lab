#pragma once

// Lane-class tree: the reduction order of a grouped integer product declared over 32 lane classes.
// Home: shared (backend-neutral moe_core contract; header only, no SDK include, host + device under HIPCC / CUDACC).
//
// A row of K = 32 G inputs is cut into G groups of 32. Group g belongs to lane class c = g mod 32. Per (row, output):
//   S_c = +0; for j ascending: S_c = S_c + term(c + 32 j)          (one FP32 chain per class, from +0)
//   x[c] = S_c; for d = 16, 8, 4, 2, 1: x[c] = x[c] + x[c + d] for c < d          (lower class is the left operand)
//   y = x[0]
// i.e. the wave butterfly v[l] += v[l ^ d] read at lane 0. Every placement of the tree (one wave holding all classes,
// classes spread over waves and joined through LDS, two passes over class bit 0) produces the same bits as long as it
// adds the same pairs in the same operand order; reference() is the definition the placements are tested against.
//
// fold() is the in-register placement: the classes are visited in bit-reversed order i -> class_of(i) = rev5(i), so
// positions 2m and 2m + 1 hold classes c and c + 16 (level 1), and each finished subtree of 2^k positions is carried in
// stack[k] (a binary counter over i). The term formula (group dot, scales, contraction) is not part of this contract:
// it belongs to the operand class that produces S_c.
//
// NaN payloads: one IEEE add per pair in the stated operand order; callers that need payload equality must keep that
// order (the compiler may commute a += b).

#if !defined(__HIPCC__) && !defined(__CUDACC__) && !defined(__GNUC__) && !defined(__clang__)
#include <bit>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_CLASS_TREE_HD __attribute__((host, device)) inline __attribute__((always_inline))
#else
#define STRIX_CLASS_TREE_HD inline
#endif
#if defined(__clang__)
#define STRIX_CLASS_TREE_UNROLL _Pragma("unroll")
#else
#define STRIX_CLASS_TREE_UNROLL
#endif

namespace strix::moe_core::class_tree {

inline constexpr unsigned kClasses = 32U;
inline constexpr unsigned kLevels = 5U;

// Class visited at position i (0..31) of the bit-reversed walk.
STRIX_CLASS_TREE_HD unsigned class_of(unsigned i) {
#if defined(__clang__)
    return __builtin_bitreverse32(i) >> 27U;
#else
    unsigned r = 0U;
    for (unsigned b = 0; b < 32U; ++b) r |= ((i >> b) & 1U) << (31U - b);
    return r >> 27U;
#endif
}

// p = left + p for every element (left = the earlier, lower-class subtree).
template <unsigned MT, unsigned E>
STRIX_CLASS_TREE_HD void add_level(const float (&left)[MT][E], float (&p)[MT][E]) {
STRIX_CLASS_TREE_UNROLL
    for (unsigned t = 0; t < MT; ++t)
STRIX_CLASS_TREE_UNROLL
        for (unsigned e = 0; e < E; ++e) p[t][e] = left[t][e] + p[t][e];
}

template <unsigned MT, unsigned E>
STRIX_CLASS_TREE_HD void copy_level(const float (&from)[MT][E], float (&to)[MT][E]) {
STRIX_CLASS_TREE_UNROLL
    for (unsigned t = 0; t < MT; ++t)
STRIX_CLASS_TREE_UNROLL
        for (unsigned e = 0; e < E; ++e) to[t][e] = from[t][e];
}

// Folds the class sum p of visiting position i into the tree: binary carry over the trailing one bits of i (uniform
// branches on constant stack levels, so the stack stays in registers). After position 31, v holds the tree result.
// stack must start at any value: every level is written before it is read.
template <unsigned MT, unsigned E>
STRIX_CLASS_TREE_HD void fold(unsigned i, float (&p)[MT][E], float (&stack)[kLevels][MT][E], float (&v)[MT][E]) {
#if defined(__HIPCC__) || defined(__CUDACC__) || defined(__GNUC__) || defined(__clang__)
    const unsigned carries = static_cast<unsigned>(__builtin_ctz(~i));
#else
    const unsigned carries = static_cast<unsigned>(std::countr_one(i));  // same value: trailing ones of i
#endif
    if (carries >= 1U) add_level(stack[0], p);
    if (carries >= 2U) add_level(stack[1], p);
    if (carries >= 3U) add_level(stack[2], p);
    if (carries >= 4U) add_level(stack[3], p);
    if (carries >= 5U) add_level(stack[4], p);
    if (carries == 0U) copy_level(p, stack[0]);
    else if (carries == 1U) copy_level(p, stack[1]);
    else if (carries == 2U) copy_level(p, stack[2]);
    else if (carries == 3U) copy_level(p, stack[3]);
    else if (carries == 4U) copy_level(p, stack[4]);
    else copy_level(p, v);
}

// The definition: the tree over the 32 class sums x (indexed by class).
STRIX_CLASS_TREE_HD float reference(const float (&x)[kClasses]) {
    float s[kClasses];
    for (unsigned c = 0; c < kClasses; ++c) s[c] = x[c];
    for (unsigned d = kClasses / 2U; d != 0U; d /= 2U)
        for (unsigned c = 0; c < d; ++c) s[c] = s[c] + s[c + d];
    return s[0];
}

} // namespace strix::moe_core::class_tree

#undef STRIX_CLASS_TREE_UNROLL
#undef STRIX_CLASS_TREE_HD
