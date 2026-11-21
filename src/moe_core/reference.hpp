#pragma once

#include "core/linear_reference.hpp"
#include "strix/moe/route_policy.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::moe {

enum class ExpertProjection { gate_up, down };

// One row of routed MoE with a fused gate/up matvec, then a down matvec.
// Projection outputs are FP32 accumulators. The conversion policy owns
// rounding at the hidden, matvec, activation and final output boundaries.
// Routing and weighted accumulation stay FP32. The combine order is by
// expert ID, independent of the route's weight-normalization order.
template <class Math, class ProjectExpert, class Activation, class Convert>
void routed_reference(const RoutePolicy& route, std::size_t intermediate, std::span<const float> hidden,
                      std::span<const float> router, std::span<const float> selection_bias,
                      ProjectExpert&& project, Activation activation, Convert convert, std::span<float> output) {
    if (const char* why = validate(route)) throw std::invalid_argument(why);
    if (hidden.empty() || output.size() != hidden.size() ||
        router.size() % hidden.size() != 0 || router.size() / hidden.size() != route.experts ||
        (route.selection_bias && selection_bias.size() != route.experts) ||
        intermediate == 0 || intermediate > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::invalid_argument("routed reference: invalid dimensions");
    }

    std::vector<float> input(hidden.size());
    std::ranges::transform(hidden, input.begin(), convert);
    std::vector<float> logits(route.experts);
    std::vector<float> scores(route.experts);
    std::vector<std::uint32_t> ids(route.top_k);
    std::vector<float> weights(route.top_k);
    core::linear_reference(router, input, logits);
    route_scores_ref<Math>(route.score, logits, scores);
    route_select_ref(route, scores, selection_bias, ids, weights);

    std::vector<std::uint32_t> slots(route.top_k);
    std::iota(slots.begin(), slots.end(), 0U);
    std::ranges::sort(slots, {}, [&](std::uint32_t slot) { return ids[slot]; });
    std::vector<float> gate_up(2 * intermediate);
    std::vector<float> active(intermediate);
    std::vector<float> down(hidden.size());
    std::vector<float> combined(hidden.size());
    for (std::uint32_t slot : slots) {
        project(ids[slot], ExpertProjection::gate_up, std::span<const float>(input), std::span<float>(gate_up));
        for (std::size_t column = 0; column < intermediate; ++column) {
            active[column] = convert(activation(convert(gate_up[column]), convert(gate_up[intermediate + column])));
        }
        project(ids[slot], ExpertProjection::down, std::span<const float>(active), std::span<float>(down));
        for (std::size_t column = 0; column < hidden.size(); ++column) {
            // Explicit multiply before accumulation. Build this reference with contraction disabled.
            const float contribution = weights[slot] * convert(down[column]);
            combined[column] += contribution;
        }
    }
    std::ranges::transform(combined, output.begin(), convert);
}

}  // namespace strix::moe
