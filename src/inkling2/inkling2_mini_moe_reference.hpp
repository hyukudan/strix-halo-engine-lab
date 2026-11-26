#pragma once

#include "inkling2_mini_moe.hpp"
#include "moe_core/reference.hpp"
#include "strix/moe/combine.hpp"

#include <span>
#include <utility>

namespace strix::inkling2 {

template <class ProjectExpert, class ObserveRoutes = std::nullptr_t>
void mini_moe_reference(std::span<const float> hidden, std::span<const float> router, std::span<const float> selection_bias,
                        ProjectExpert&& project, std::span<float> output, ObserveRoutes observe_routes = nullptr,
                        std::span<const float> projected_router = {}) {
    if (hidden.size() != kHiddenWidth) {
        throw std::invalid_argument("Inkling 2 Mini MoE: hidden row width differs from the checkpoint");
    }
    moe::routed_reference<detail::MiniMoeMath>(kMiniRoute, kExpertWidth, hidden, router, selection_bias,
                                             std::forward<ProjectExpert>(project),
                                             kMiniExpertActivation<detail::MiniMoeMath>,
                                             moe::combine_round_bf16, output, observe_routes, projected_router);
}

}  // namespace strix::inkling2
