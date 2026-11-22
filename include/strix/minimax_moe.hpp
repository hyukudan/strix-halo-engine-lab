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
#include "strix/model/bind.hpp"
#include "strix/model_data/minimax_m3_1_flash.hpp"
#include "strix/moe/route_policy.hpp"

namespace strix::minimax::moe {

inline constexpr unsigned kHidden = 6144, kExperts = 128, kTopE = 4;
inline constexpr unsigned kNormThreads = 256, kNormPerThread = kHidden / kNormThreads;  // 24
inline constexpr unsigned kRouterLanes = 32, kRouterPerLane = kHidden / kRouterLanes;   // 192
inline constexpr float kNormEps = 1e-6F, kRouteScale = 2.0F, kSumEps = 1e-20F;

// The router, bound from models/minimax-m3.1-flash.toml [moe.route]: sigmoid + selection bias, serial float keys,
// selection order, renormalised, x 2.0 (host route_row; the device route_topk serial scan).
namespace route_data = ::strix::model_data::minimax_m3_1_flash::moe;
inline constexpr ::strix::moe::RoutePolicy kRoutePolicy = ::strix::moe::route_policy_named(
    route_data::route::score, route_data::route::selection_bias, route_data::route::key, route_data::route::order,
    ::strix::model::narrow<std::uint32_t>(route_data::experts), ::strix::model::narrow<std::uint32_t>(route_data::top_k),
    route_data::route::renormalize, route_data::route::renorm_single,
    ::strix::model::nearest_float(route_data::route::renorm_eps), ::strix::model::nearest_float(route_data::route::scale));
static_assert(::strix::moe::validate(kRoutePolicy) == nullptr);
static_assert(kRoutePolicy.score == ::strix::moe::RouteScore::sigmoid && kRoutePolicy.selection_bias &&
                  kRoutePolicy.key == ::strix::moe::RouteKey::float_serial &&
                  kRoutePolicy.order == ::strix::moe::RouteOrder::selection && kRoutePolicy.experts == kExperts &&
                  kRoutePolicy.top_k == kTopE && kRoutePolicy.n_group == 1U && kRoutePolicy.topk_group == 1U &&
                  kRoutePolicy.renormalize && kRoutePolicy.renorm_single && kRoutePolicy.renorm_eps == kSumEps &&
                  kRoutePolicy.scale == kRouteScale,
              "kRoutePolicy == the former shared preset minimax_m3 {sigmoid, bias, float_serial, selection, 128, 4, 1, "
              "1, true, true, 1e-20F, 2.0F}");

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
