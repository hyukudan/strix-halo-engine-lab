#pragma once
// Host rule of the eager selected softmax (device form: eager_softmax.hpp). A lane-by-lane emulation of the block:
// strided lane partials, the LDS halving trees (fmaxf, then +=) over Threads lanes, the two failure checks and the
// rounded probabilities. `exp` and `round` are the caller's functions, so the rule fixes the order of operations,
// not the transcendental: with the device expf and rounding supplied it reproduces the device bits.
// Home: shared (attn_core): no model names, no SDK.
#include <cmath>
#include <cstdint>
#include <vector>

namespace strix::attn_core::eager_softmax {

// Lane 0's value after the LDS halving tree over lanes.size() (a power of two) lanes.
template <class Combine>
inline float halving_tree(std::vector<float> lanes, Combine combine) {
    for (std::size_t stride = lanes.size() / 2U; stride > 0U; stride >>= 1U)
        for (std::size_t lane = 0; lane < stride; ++lane) lanes[lane] = combine(lanes[lane], lanes[lane + stride]);
    return lanes.empty() ? 0.0F : lanes[0];
}

struct RowResult {
    bool failed = false;          // a check fired (row left unwritten)
    float maximum = 0.0F, denominator = 0.0F;
    std::vector<float> probabilities;  // rounded, size count when !failed
};

template <unsigned Threads, class Exp, class Round>
inline RowResult softmax_row_rule(const float* scores, std::uint32_t count, bool checks, Exp exp_fn, Round round_fn) {
    static_assert(Threads >= 1U && (Threads & (Threads - 1U)) == 0U, "eager_softmax: Threads must be a power of two");
    RowResult r;
    std::vector<float> lanes(Threads, -INFINITY);
    for (std::uint32_t lane = 0; lane < Threads; ++lane)
        for (std::uint32_t i = lane; i < count; i += Threads) lanes[lane] = std::fmax(lanes[lane], scores[i]);
    r.maximum = halving_tree(lanes, [](float a, float b) { return std::fmax(a, b); });
    if (checks && !std::isfinite(r.maximum)) {
        r.failed = true;
        return r;
    }
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        float sum = 0.0F;
        for (std::uint32_t i = lane; i < count; i += Threads) sum += exp_fn(scores[i] - r.maximum);
        lanes[lane] = sum;
    }
    r.denominator = halving_tree(lanes, [](float a, float b) { return a + b; });
    if (checks && (!std::isfinite(r.denominator) || r.denominator <= 0.0F)) {
        r.failed = true;
        return r;
    }
    r.probabilities.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) r.probabilities[i] = round_fn(exp_fn(scores[i] - r.maximum) / r.denominator);
    return r;
}

}  // namespace strix::attn_core::eager_softmax
