#pragma once
// Pre-paired FP16 activation layout of the W4A16 small-M core (core/w4a16.hpp wave_rows<..., UniformX = true>).
// The core multiplies the nibbles at bits s and 16 + s of a weight word (k j and k j + 4 of an 8-k block) with the
// activation pair (x[k j], x[k j + 4]) in one v_dot2. With the natural layout every wave re-pairs the activation words of
// every row and k-tile (w4a16::pair_activations, 8 bit operations per row and k-tile); with this layout the prep kernel
// stores each 8-k block as the words (x0,x4) (x1,x5) (x2,x6) (x3,x7), so the loaded words ARE the dot2 operands.
// Pure permutation inside each 8-k block: the operands, and so the dot2 results, are bit-identical.
// Host and device (no HIP dependency): the host test checks the permutation against the core's pairing rule.
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define STRIX_IU4_PAIRS_HD __host__ __device__
#else
#define STRIX_IU4_PAIRS_HD
#endif

namespace strix::core::iu4_pairs {

// Element position of activation i (row-major [rows][inputs], inputs % 8 == 0) in the paired layout.
STRIX_IU4_PAIRS_HD constexpr std::size_t paired_index(std::size_t i) {
    return (i & ~std::size_t{7}) | ((i & 3U) << 1U) | ((i >> 2U) & 1U);
}

// Host statement of w4a16::pair_activations (same bit operations): a = natural words (x 2i, x 2i + 1) of 8 k,
// p[j] = (x j, x j + 4) as one 32-bit word, low half first.
constexpr void pair_words(const std::uint32_t (&a)[4], std::uint32_t (&p)[4]) {
    p[0] = (a[0] & 0xFFFFU) | (a[2] << 16U);
    p[1] = (a[0] >> 16U) | (a[2] & 0xFFFF0000U);
    p[2] = (a[1] & 0xFFFFU) | (a[3] << 16U);
    p[3] = (a[1] >> 16U) | (a[3] & 0xFFFF0000U);
}

}  // namespace strix::core::iu4_pairs

#undef STRIX_IU4_PAIRS_HD
