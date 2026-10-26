#include "strix/minimax_moe.hpp"

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
    float s[kExperts], c[kExperts];
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
        c[e] = s[e] + bias[e];
    }
    std::uint32_t order[kExperts];
    std::iota(order, order + kExperts, 0U);
    std::partial_sort(order, order + kTopE, order + kExperts,
                      [&](std::uint32_t a, std::uint32_t b) { return c[a] != c[b] ? c[a] > c[b] : a < b; });
    RouteRow out{};
    float sum = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k) sum = sum + s[order[k]];
    for (unsigned k = 0; k < kTopE; ++k) {
        out.ids[k] = order[k];
        out.weights[k] = (s[order[k]] / (sum + kSumEps)) * kRouteScale;
    }
    return out;
}

}  // namespace strix::minimax::moe
