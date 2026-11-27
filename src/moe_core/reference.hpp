#pragma once

#include "core/linear_reference.hpp"
#include "strix/moe/combine.hpp"
#include "strix/moe/route_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace strix::moe {

enum class ExpertProjection { gate_up, down };

struct NativeGluMath {
    static float exp(float value) { return std::exp(value); }
    static float tanh(float value) { return std::tanh(value); }
};

// The caller owns the format at each GEMM boundary. 'active' may alias gate.
template <class Activation, class Convert>
void activate_glu_reference(std::span<const float> gate, std::span<const float> up, std::span<float> active,
                            const Activation& activation, const Convert& convert) {
    if (gate.size() != up.size() || gate.size() != active.size())
        throw std::invalid_argument("GLU reference: inconsistent projected rows");
    for (std::size_t column = 0; column < active.size(); ++column)
        active[column] = convert(activation(convert(gate[column]), convert(up[column])));
}

// Grouped reference for one row, with independently selected routes. Combine
// in ascending expert ID, independently of the weight-normalization order.
template <class ProjectExpert, class Activation, class Convert>
void combine_routed_reference(std::size_t intermediate, std::span<const float> input,
                              std::span<const std::uint32_t> ids, std::span<const float> weights,
                              ProjectExpert&& project, Activation&& activation, Convert&& convert,
                              std::span<float> output) {
    if (input.empty() || output.size() != input.size() || ids.empty() || ids.size() != weights.size() ||
        intermediate == 0 || intermediate > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::invalid_argument("routed reference: invalid dimensions");
    }
    std::vector<std::size_t> slots(ids.size());
    std::iota(slots.begin(), slots.end(), std::size_t{0});
    std::ranges::sort(slots, {}, [&](std::size_t slot) { return ids[slot]; });
    std::vector<float> gate_up(2 * intermediate);
    std::vector<float> active(intermediate);
    std::vector<float> down(output.size());
    std::vector<float> combined(output.size());
    for (const auto slot : slots) {
        project(ids[slot], ExpertProjection::gate_up, input, std::span<float>(gate_up));
        activate_glu_reference(std::span(gate_up).first(intermediate), std::span(gate_up).subspan(intermediate),
                               active, activation, convert);
        project(ids[slot], ExpertProjection::down, std::span<const float>(active), std::span<float>(down));
        for (std::size_t column = 0; column < output.size(); ++column)
            combined[column] = add_weighted_expert_f32(combined[column], weights[slot], convert(down[column]));
    }
    std::ranges::transform(combined, output.begin(), convert);
}

// One row of routed MoE. The conversion policy owns rounding at the hidden,
// matvec, activation and final output boundaries.
template <class Math, class ProjectExpert, class Activation, class Convert, class ObserveRoutes = std::nullptr_t>
void routed_reference(const RoutePolicy& route, std::size_t intermediate, std::span<const float> hidden,
                      std::span<const float> router, std::span<const float> selection_bias,
                      ProjectExpert&& project, Activation activation, Convert convert, std::span<float> output,
                      ObserveRoutes observe_routes = nullptr, std::span<const float> projected_router = {}) {
    if (const char* why = validate(route)) throw std::invalid_argument(why);
    if (hidden.empty() || output.size() != hidden.size() ||
        (projected_router.empty() && (router.size() % hidden.size() != 0 || router.size() / hidden.size() != route.experts)) ||
        (route.selection_bias && selection_bias.size() != route.experts) ||
        intermediate == 0 || intermediate > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::invalid_argument("routed reference: invalid dimensions");
    }
    std::vector<float> input(hidden.size());
    std::ranges::transform(hidden, input.begin(), convert);
    std::vector<float> logits(route.experts), scores(route.experts), weights(route.top_k);
    std::vector<std::uint32_t> ids(route.top_k);
    if (projected_router.empty()) core::linear_reference(router, input, logits);
    else {
        if (projected_router.size() != logits.size()) throw std::invalid_argument("invalid projected router width");
        std::ranges::copy(projected_router, logits.begin());
    }
    route_scores_ref<Math>(route.score, logits, scores);
    route_select_ref(route, scores, selection_bias, ids, weights);
    if constexpr (!std::is_same_v<ObserveRoutes, std::nullptr_t>) observe_routes(ids, weights);
    combine_routed_reference(intermediate, input, ids, weights, project, activation, convert, output);
}

}  // namespace strix::moe
