#pragma once
// minimax-m3 post-exchange step (fused, designed for the TP2 decode): after the FFN/attention exchange of the two
// ranks' partial rows, one kernel per row does h += part0 + part1 (fixed rank order, so both ranks hold the same
// bits), the Gemma RMS norm of the next sublayer, and optionally the MoE router: FP32 logits over the 128 experts,
// sigmoid, + score-correction bias for the choice, top-4 (choice desc, expert id asc), weights = sigmoid / (sum +
// 1e-20) x 2.0. Rows are independent (batch invariance: a row's bits never depend on the other rows of the call).
// Host reference = the declared order of the device kernel (tests/hip_minimax_moe_tests.hip checks it bit for bit).
#include <cstdint>
#include <span>

#include "strix/minimax_msa.hpp"

namespace strix::minimax::moe {

inline constexpr unsigned kHidden = 6144, kExperts = 128, kTopE = 4;
inline constexpr unsigned kNormThreads = 256, kNormPerThread = kHidden / kNormThreads;  // 24
inline constexpr unsigned kRouterLanes = 32, kRouterPerLane = kHidden / kRouterLanes;   // 192
inline constexpr float kNormEps = 1e-6F, kRouteScale = 2.0F, kSumEps = 1e-20F;

// Declared order:
//  residual: h_i = h_i + (part0_i + part1_i).
//  norm: thread t (0..255) chains acc = fmaf(h_j, h_j, acc) over j = t, t + 256, ... (24 terms); the 256 partials are
//        summed by the tree p[t] += p[t + s] for s = 128, 64, ..., 1; r = 1 / sqrt(p[0] / 6144 + 1e-6);
//        y_i = (h_i * r) * (1 + w_i).
//  router logit e: lane l (0..31) chains acc = fmaf(y_j, W[e][j], acc) over j = l, l + 32, ... (192 terms); butterfly
//        v += v[l ^ d] for d = 16..1. s_e = 1 / (1 + exp_f(-logit_e)); c_e = s_e + bias_e.
//  top-4: by (c desc, id asc), in that order; sum = ((s_0 + s_1) + s_2) + s_3 over the chosen order;
//        w_k = (s_k / (sum + 1e-20)) * 2.
struct RouteRow {
    std::uint32_t ids[kTopE];
    float weights[kTopE];
};
void post_exchange_row(float* h, const float* part0, const float* part1, const float* norm_w, float* y);
RouteRow route_row(const float* y, const float* router_w /* [128][6144] */, const float* bias);

}  // namespace strix::minimax::moe
