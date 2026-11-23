#pragma once
// Host rule of the eager selected score and value forms (device forms: eager_selected.hpp). Lane-by-lane emulation:
// one product per lane and the LDS halving tree over Threads lanes, scaled by 1 / sqrtf(head_dim); the ascending
// fmaf value chain from +0; the sigmoid-gated rounded epilogue. `exp` and `round` are the caller's functions, so the
// rule fixes the order of operations, not the transcendental.
// Home: shared (attn_core): no model names, no SDK.
#include <cmath>
#include <cstdint>
#include <vector>

#include "attn_core/eager_softmax_rule.hpp"

namespace strix::attn_core::eager_selected {

// Score of one (head, slot): q and k hold head_dim values each (head_dim <= Threads).
template <unsigned Threads>
inline float score_rule(const float* q, const float* k, std::uint32_t head_dim) {
    static_assert(Threads >= 1U && (Threads & (Threads - 1U)) == 0U, "eager_selected: Threads must be a power of two");
    std::vector<float> lanes(Threads, 0.0F);
    for (std::uint32_t f = 0; f < head_dim && f < Threads; ++f) lanes[f] = q[f] * k[f];
    const float root = eager_softmax::halving_tree(lanes, [](float a, float b) { return a + b; });
    return root / std::sqrt(static_cast<float>(head_dim));
}

// Value chain of one (head, feature): probabilities p[slot] and that feature's values v[slot], slot < count.
inline float value_rule(const float* p, const float* v, std::uint32_t count) {
    float sum = 0.0F;
    for (std::uint32_t slot = 0; slot < count; ++slot) sum = std::fma(p[slot], v[slot], sum);
    return sum;
}

template <class Exp, class Round>
inline float gated_output_rule(float sum, float gate_logit, Exp exp_fn, Round round_fn) {
    const float attn = round_fn(sum);
    const float gate = round_fn(1.0F / (1.0F + exp_fn(-gate_logit)));
    return round_fn(attn * gate);
}

}  // namespace strix::attn_core::eager_selected
