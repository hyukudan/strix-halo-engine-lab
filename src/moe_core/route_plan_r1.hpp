#pragma once

#include "moe_core/microscaled_r1_geometry.hpp"
#include "strix/moe/route_policy.hpp"
#include "strix/weight/formats/microscaled.hpp"

#include <array>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_R1_PLAN_HD __host__ __device__
#else
#define STRIX_R1_PLAN_HD
#endif

namespace strix::moe_core::route_plan_r1 {

using weight::microscaled::MatrixFormat;

struct LayerPlan {
    MatrixFormat format;
    unsigned hidden;
    unsigned intermediate;
    unsigned columns_gate_up;
    unsigned columns_down;
};

// Column limits certified for the resolved layer format on the target device.
// A caller without such a certificate uses the single-column form.
struct ColumnLimits {
    unsigned gate_up = 1U;
    unsigned down = 1U;
};

constexpr unsigned columns_to_use(unsigned limit, unsigned width, unsigned workers) {
    if (limit >= 4U && width >= 4U * workers) return 4U;
    if (limit >= 2U && width >= 2U * workers) return 2U;
    return 1U;
}

constexpr bool valid_column_limit(unsigned columns) {
    return columns == 1U || columns == 2U || columns == 4U;
}

[[nodiscard]] constexpr std::optional<LayerPlan> make_layer_plan(
    MatrixFormat format, unsigned hidden, unsigned intermediate, ColumnLimits limits = {}) {
    unsigned block;
    switch (format) {
    case MatrixFormat::nvfp4: block = weight::microscaled::kNvfp4BlockColumns; break;
    case MatrixFormat::mxfp8: block = weight::microscaled::kMxfp8BlockColumns; break;
    default: return std::nullopt;
    }
    constexpr auto max_width = std::numeric_limits<unsigned>::max() / 2U;
    if (hidden == 0U || hidden > max_width || hidden % block != 0U ||
        intermediate == 0U || intermediate > max_width || intermediate % block != 0U ||
        !valid_column_limit(limits.gate_up) || !valid_column_limit(limits.down))
        return std::nullopt;
    return LayerPlan{format, hidden, intermediate,
                     columns_to_use(limits.gate_up, intermediate, microscaled_r1::kWavesPerBlock),
                     columns_to_use(limits.down, hidden, 1U)};
}

[[nodiscard]] constexpr bool valid(const LayerPlan& plan) {
    const auto canonical = make_layer_plan(plan.format, plan.hidden, plan.intermediate,
                                           {plan.columns_gate_up, plan.columns_down});
    return canonical && canonical->columns_gate_up == plan.columns_gate_up &&
           canonical->columns_down == plan.columns_down;
}

template <class Policy>
struct Selection : moe::RouteSerialScorePlanes<Policy> {
    static_assert(Policy::value.score == moe::RouteScore::sigmoid);
    static_assert(Policy::value.top_k <= 32U);
};

// Each picked expert has exactly one assignment in R1. Its rank among the
// picked IDs is its compact group index. Route IDs and weights stay in
// selection order so the original normalization and activation layout survive.
template <unsigned TopK>
STRIX_R1_PLAN_HD inline void compact_groups(const std::uint32_t* ids, std::uint32_t* group_by_slot) {
    for (unsigned slot = 0; slot < TopK; ++slot) {
        std::uint32_t group = 0U;
        for (unsigned other = 0; other < TopK; ++other) group += ids[other] < ids[slot];
        group_by_slot[slot] = group;
    }
}

template <unsigned TopK>
struct Route {
    std::array<std::uint32_t, TopK> ids;
    std::array<float, TopK> weights;
    std::array<std::uint32_t, TopK> group_by_slot;
};

// Host reference for projection, selection and compact singleton groups.
template <class Policy, class Math, class Dot = moe::RouteStridedFmaXor32>
Route<Selection<Policy>::top_k> reference(const LayerPlan& plan, std::span<const float> input,
                                         std::span<const float> weights, std::span<const float> bias) {
    using Select = Selection<Policy>;
    if (!valid(plan) || input.size() != plan.hidden || bias.size() != Select::experts ||
        weights.size() / plan.hidden != Select::experts || weights.size() % plan.hidden != 0U)
        throw std::invalid_argument("R1 route plan: invalid projection geometry");

    std::array<float, Select::experts> scores, choices;
    for (std::size_t expert = 0; expert < Select::experts; ++expert) {
        const auto pair = moe::route_project_sigmoid_ref<Dot, Math>(
            input, weights.subspan(expert * plan.hidden, plan.hidden), bias[expert]);
        scores[expert] = pair.unbiased;
        choices[expert] = pair.biased;
    }
    Route<Select::top_k> route;
    Select::template select_row<1U>({scores.data(), choices.data()}, 0U, route.ids.data(), route.weights.data());
    compact_groups<Select::top_k>(route.ids.data(), route.group_by_slot.data());
    return route;
}

}  // namespace strix::moe_core::route_plan_r1

#undef STRIX_R1_PLAN_HD
