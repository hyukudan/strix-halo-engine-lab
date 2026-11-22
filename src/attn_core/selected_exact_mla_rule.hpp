#pragma once
// Host-neutral statement of the EXACT materialised rows-selected latent attention (selected_exact_mla.hpp): the lane
// chains, the 256-lane halving tree, the binary-counter evaluation of that tree used by obar_lanes_kernel and the
// lane-major probability layout written by softmax_kernel<S, true>. Scalar code in the device code's order. Used by
// the CPU test to show the counter form and the tree form give the same bits, and that the layout is a bijection.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::attn_core::selected_exact::rule {

inline constexpr unsigned kLanes = 256;

constexpr unsigned bitrev8(unsigned x) {
    unsigned r = 0;
    for (unsigned i = 0; i < 8; ++i) r |= ((x >> i) & 1U) << (7U - i);
    return r;
}
constexpr unsigned rounds(unsigned list_cap) { return (list_cap + kLanes - 1) / kLanes; }
// Lane-major slot of list entry t: lane t % 256 owns kRounds contiguous slots, lanes in k = bitrev8(lane) order.
constexpr unsigned lane_major_slot(unsigned t, unsigned list_cap) { return rounds(list_cap) * bitrev8(t % kLanes) + t / kLanes; }

// Halving tree over the 256 lane values: s[l] = s[l] + s[l + h], h = 128 .. 1 (softmax sum, o-bar).
inline float halving_tree(std::array<float, kLanes> s) {
    for (unsigned h = kLanes / 2; h > 0; h >>= 1U)
        for (unsigned l = 0; l < h; ++l) s[l] = s[l] + s[l + h];
    return s[0];
}
// The same tree as obar_lanes_kernel evaluates it: lanes visited in k order (lane = bitrev8(k)), each value carried
// through the set low bits of k (left operand = the stored lower-k subtree), placed at the first clear bit; k = 255
// carries through all 8 levels and is the root.
inline float counter_tree(const std::array<float, kLanes>& lanes) {
    std::array<float, 8> stack{};
    float root = 0.0F;
    for (unsigned k = 0; k < kLanes; ++k) {
        float c = lanes[bitrev8(k)];
        bool placed = false;
        for (unsigned level = 0; level < 8 && !placed; ++level) {
            if ((k >> level) & 1U) {
                c = stack[level] + c;
            } else {
                stack[level] = c;
                placed = true;
            }
        }
        if (!placed) root = c;
    }
    return root;
}

// Lane chain of the o-bar: fmaf(p_t, c_t, a) over t = lane, lane + 256, ... ascending from +0.
inline std::array<float, kLanes> obar_lanes(std::span<const float> p, std::span<const float> c) {
    std::array<float, kLanes> a{};
    for (std::size_t t = 0; t < p.size(); ++t) a[t % kLanes] = __builtin_fmaf(p[t], c[t], a[t % kLanes]);
    return a;
}
// Lane chain of the softmax sum: a = a + w_t over t = lane, lane + 256, ... from +0.
inline std::array<float, kLanes> sum_lanes(std::span<const float> w) {
    std::array<float, kLanes> a{};
    for (std::size_t t = 0; t < w.size(); ++t) a[t % kLanes] = a[t % kLanes] + w[t];
    return a;
}

}  // namespace strix::attn_core::selected_exact::rule
