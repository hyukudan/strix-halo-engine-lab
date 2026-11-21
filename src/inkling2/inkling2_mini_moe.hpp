#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/moe/activation_policy.hpp"
#include "strix/moe/route_policy.hpp"

namespace strix::inkling2 {

inline constexpr moe::RoutePolicy kMiniRoute{
    moe::RouteScore::sigmoid, true, moe::RouteKey::float_serial,
    moe::RouteOrder::selection, kExperts, kExpertsPerToken, 1, 1, true, true, 1e-20F, kRouteScale};
static_assert(moe::validate(kMiniRoute) == nullptr);

template <class Math>
inline constexpr moe::SoftcappedSwiglu<Math> kMiniExpertActivation{kMlpGateSoftcap, kMlpUpSoftcap};

}  // namespace strix::inkling2
