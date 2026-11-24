#pragma once
// F3 route projection, HIP device form R1 "row": one wave per (expert, row) computes the router logit by the chain of
// include/strix/moe/route_project.hpp (RouteLaneDot<Acc>), then writes Score::apply(logit). Rows are independent, so a
// row's logit is the same chain whatever the window size. The waves of one expert are adjacent (its weight row is read
// once from DRAM, then from the caches); each lane keeps the next Ahead products' operands in flight.
// Geometry: hidden % (32 * Ahead) == 0 and hidden >= 32 * Ahead. Grid: ceil(experts * rows / Waves) blocks of 32 * Waves.
// Weights: FP32, or a 16-bit BF16 copy widened exactly (u16 << 16).
#include "strix/moe/route_project.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::moe_core {

template <class W>
__device__ __forceinline__ float route_project_weight(const W* w, unsigned i) {
    if constexpr (sizeof(W) == 2) return __uint_as_float(static_cast<unsigned>(w[i]) << 16U);
    else return w[i];
}

template <class Acc, class Score, unsigned Ahead, unsigned Waves, class W>
__device__ __forceinline__ void route_project_row_body(const float* __restrict__ x, const W* __restrict__ weight,
                                                       float* __restrict__ scores, unsigned rows, unsigned hidden,
                                                       unsigned experts) {
    using Dot = moe::RouteLaneDot<Acc, 32U>;
    constexpr unsigned U = Ahead, step = 32U * U;
    const unsigned lane = threadIdx.x % 32U, wave = blockIdx.x * Waves + threadIdx.x / 32U;
    if (wave >= experts * rows) return;
    const unsigned expert = wave / rows, r = wave % rows;
    const W* w = weight + static_cast<unsigned long long>(expert) * hidden + lane;
    const float* xr = x + static_cast<unsigned long long>(r) * hidden + lane;
    float wv[U], xv[U];
#pragma unroll
    for (unsigned k = 0; k < U; ++k) { wv[k] = route_project_weight(w, 32U * k); xv[k] = xr[32U * k]; }
    float acc = 0.0F;
    for (unsigned c0 = 0; c0 < hidden; c0 += step) {
        const unsigned c1 = min(c0 + step, hidden - step);  // the last chunk loads itself again (unused)
        float wn[U], xn[U];
#pragma unroll
        for (unsigned k = 0; k < U; ++k) { wn[k] = route_project_weight(w, c1 + 32U * k); xn[k] = xr[c1 + 32U * k]; }
#pragma unroll
        for (unsigned k = 0; k < U; ++k) acc = Dot::step(acc, wv[k], xv[k]);
#pragma unroll
        for (unsigned k = 0; k < U; ++k) { wv[k] = wn[k]; xv[k] = xn[k]; }
    }
    acc = Dot::reduce(acc, [](float v, int d) { return __shfl_xor(v, d, 32); });
    if (lane == 0U) scores[static_cast<unsigned long long>(r) * experts + expert] = Score::apply(acc);
}

}  // namespace strix::moe_core
