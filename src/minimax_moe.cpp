#include "strix/minimax_moe.hpp"
#include "strix/moe/route_policy.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace strix::minimax::moe {

void post_exchange_row(float* h, const float* part0, const float* part1, const float* norm_w, float* y) {
    for (unsigned i = 0; i < kHidden; ++i) h[i] = h[i] + (part0[i] + part1[i]);
    float p[kNormThreads];
    for (unsigned t = 0; t < kNormThreads; ++t) {
        float acc = 0.0F;
        for (unsigned j = t; j < kHidden; j += kNormThreads) acc = std::fmaf(h[j], h[j], acc);
        p[t] = acc;
    }
    for (unsigned s = kNormThreads / 2; s >= 1; s >>= 1U)
        for (unsigned t = 0; t < s; ++t) p[t] = p[t] + p[t + s];
    const float r = 1.0F / std::sqrt(p[0] / static_cast<float>(kHidden) + kNormEps);
    for (unsigned i = 0; i < kHidden; ++i) y[i] = (h[i] * r) * (1.0F + norm_w[i]);
}

RouteRow route_row(const float* y, const float* router_w, const float* bias) {
    float s[kExperts];
    for (unsigned e = 0; e < kExperts; ++e) {
        const float* w = router_w + static_cast<std::size_t>(e) * kHidden;
        float lanes[kRouterLanes];
        for (unsigned l = 0; l < kRouterLanes; ++l) {
            float acc = 0.0F;
            for (unsigned j = l; j < kHidden; j += kRouterLanes) acc = std::fmaf(y[j], w[j], acc);
            lanes[l] = acc;
        }
        msa::butterfly(lanes);
        s[e] = 1.0F / (1.0F + msa::exp_f(-lanes[0]));
    }
    // Selection and weights: the shared router on kRoutePolicy (= the device route_topk serial scan; the former
    // partial_sort gave the same picks for finite choices and was not a strict weak order for NaN).
    RouteRow out{};
    strix::moe::route_select_ref(kRoutePolicy, s, std::span<const float>(bias, kExperts), out.ids, out.weights);
    return out;
}

}  // namespace strix::minimax::moe
