#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/moe/activation_policy.hpp"
#include "strix/moe/route_policy.hpp"
#include "strix/moe_plan.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace strix::inkling2 {

inline constexpr moe::RoutePolicy kMiniRoute{
    moe::RouteScore::sigmoid, true, moe::RouteKey::float_serial,
    moe::RouteOrder::selection, kExperts, kExpertsPerToken, 1, 1, true, true, 1e-20F, kRouteScale};
static_assert(moe::validate(kMiniRoute) == nullptr);

template <class Math>
inline constexpr moe::SoftcappedSwiglu<Math> kMiniExpertActivation{kMlpGateSoftcap, kMlpUpSoftcap};

namespace detail {

struct MiniMoeMath {
#if defined(__HIPCC__) || defined(__CUDACC__)
    __host__ __device__
#endif
    static float exp(float value) { return ::expf(value); }

#if defined(__HIPCC__) || defined(__CUDACC__)
    __host__ __device__
#endif
    static float tanh(float value) { return ::tanhf(value); }
};

}  // namespace detail

// Select from token-major FP32 router logits with one layer-wide correction
// bias. Return normalized weights and IDs in selection order for dispatch.
[[nodiscard]] inline MoeRoutingSelection select_mini_routes(std::span<const float> logits,
                                                            std::span<const float> selection_bias) {
    const auto finite = [](float value) { return std::isfinite(value); };
    const auto rows = logits.size() / kExperts;
    if (rows == 0 || rows > kMaxContext || logits.size() % kExperts != 0 || selection_bias.size() != kExperts ||
        !std::ranges::all_of(logits, finite) || !std::ranges::all_of(selection_bias, finite)) {
        throw std::invalid_argument("Inkling 2 Mini MoE: invalid FP32 router logits or bias");
    }
    const auto assignments = rows * kExpertsPerToken;
    MoeRoutingSelection routes;
    routes.experts.resize(assignments);
    routes.weights.resize(assignments);
    moe::route_rows_ref<detail::MiniMoeMath>(kMiniRoute, logits, selection_bias, routes.experts, routes.weights);
    return routes;
}

}  // namespace strix::inkling2
