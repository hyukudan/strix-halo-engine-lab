#pragma once

#include "inkling2_mini_moe.hpp"
#include "inkling2_mini_schedule.hpp"
#include "moe_core/expert_major_plan.hpp"
#include "moe_core/grouped_microscaled_rows.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::inkling2 {

static_assert(2U * kExpertWidth == kHiddenWidth);

struct MiniMoeWorkspace {
    // Gather and activation share the BF16 arena. Gate/up and down share
    // the FP32 arena. Each reuse follows its final reader on the stream.
    std::span<std::uint16_t> bf16;
    std::span<float> fp32;
};

struct MiniMoeExpert {
    weight::microscaled::PackedMatrix gate_up;
    weight::microscaled::PackedMatrix down;
};

struct MiniMoeDispatch {
    unsigned physical_layer{};
    unsigned rows{};
    moe_core::ExpertMajorPlan routes;
    std::vector<moe_core::grouped_microscaled::TileRows> gate_up;
    std::vector<moe_core::grouped_microscaled::TileRows> down;
};

// Per arena, in elements. The caller chooses a bounded chunk for prefill.
inline std::size_t mini_moe_workspace_elements(unsigned rows) {
    if (rows == 0 || rows > kMaxContext) {
        throw std::invalid_argument("Inkling 2 Mini MoE: invalid row count");
    }
    return static_cast<std::size_t>(rows) * kExpertsPerToken * kHiddenWidth;
}

// IDs and weights are token-major and in selection order, after normalizing
// by kMiniRoute. Weight and workspace addresses refer to device memory.
template <class Backend>
inline MiniMoeDispatch prepare_mini_moe(
    unsigned physical_layer, unsigned rows, std::span<const std::uint32_t> experts,
    std::span<const float> weights, std::span<const MiniMoeExpert> matrices, MiniMoeWorkspace workspace) {
    namespace grouped = moe_core::grouped_microscaled;
    const auto elements = mini_moe_workspace_elements(rows);
    if (physical_layer >= kPhysicalLayers || matrices.size() != kExperts ||
        workspace.bf16.size() < elements || workspace.fp32.size() < elements) {
        throw std::invalid_argument("Inkling 2 Mini MoE: invalid layer, weights or workspace");
    }
    auto routes = moe_core::build_expert_major_plan(
        {rows, kExpertsPerToken, kExperts, 0, experts}, weights, Backend::rows_per_tile);
    if (routes.tiles.size() > grouped::kMaxTiles) {
        throw std::invalid_argument("Inkling 2 Mini MoE: chunk exceeds the backend launch grid");
    }
    MiniMoeDispatch dispatch{physical_layer, rows, std::move(routes), {}, {}};
    dispatch.gate_up.reserve(dispatch.routes.tiles.size());
    dispatch.down.reserve(dispatch.routes.tiles.size());
    for (const auto& tile : dispatch.routes.tiles) {
        const auto& matrix = matrices[tile.expert];
        const auto wide_offset = static_cast<std::size_t>(tile.first_assignment) * kHiddenWidth;
        const auto narrow_offset = static_cast<std::size_t>(tile.first_assignment) * kExpertWidth;
        auto* result = workspace.fp32.data() + wide_offset;
        const grouped::TileRows gate_up{
            matrix.gate_up, workspace.bf16.data() + wide_offset, result, kHiddenWidth, kHiddenWidth, tile.rows};
        const grouped::TileRows down{
            matrix.down, workspace.bf16.data() + narrow_offset, result, kExpertWidth, kHiddenWidth, tile.rows};
        if (!Backend::fits_tile(physical_layer, MiniExpertProjection::gate_up, gate_up) ||
            !Backend::fits_tile(physical_layer, MiniExpertProjection::down, down)) {
            throw std::invalid_argument("Inkling 2 Mini MoE: incompatible expert descriptor");
        }
        dispatch.gate_up.push_back(gate_up);
        dispatch.down.push_back(down);
    }
    return dispatch;
}

struct MiniMoeDevicePlan {
    unsigned physical_layer;
    unsigned rows;
    // All spans are device arrays uploaded from one prepared dispatch.
    std::span<const std::uint32_t> assignment_rows;
    std::span<const std::uint32_t> combine_rows;
    std::span<const float> assignment_weights;
    std::span<const moe_core::grouped_microscaled::TileRows> gate_up;
    std::span<const moe_core::grouped_microscaled::TileRows> down;
};

// Upload the plan before launching on this stream. Keep the plan, weights,
// and disjoint input/output/workspace buffers alive through stream completion.
// Use the same workspace addresses supplied when preparing the descriptors.
template <class Backend>
inline typename Backend::Status enqueue_mini_moe(
    const MiniMoeDevicePlan& plan, std::span<const float> input, std::span<float> output,
    MiniMoeWorkspace workspace, typename Backend::Stream stream) {
    namespace grouped = moe_core::grouped_microscaled;
    const std::size_t row_elements = static_cast<std::size_t>(plan.rows) * kHiddenWidth;
    if (plan.rows == 0 || plan.rows > kMaxContext) return Backend::invalid_value;
    const unsigned assignments = plan.rows * kExpertsPerToken;
    const std::size_t elements = static_cast<std::size_t>(assignments) * kHiddenWidth;
    if (plan.physical_layer >= kPhysicalLayers ||
        input.size() < row_elements || output.size() < row_elements ||
        workspace.bf16.size() < elements || workspace.fp32.size() < elements ||
        plan.assignment_rows.size() != assignments || plan.combine_rows.size() != assignments ||
        plan.assignment_weights.size() != assignments || plan.gate_up.size() != plan.down.size() ||
        plan.gate_up.empty() || plan.gate_up.size() > grouped::kMaxTiles) return Backend::invalid_value;

    const auto count = static_cast<unsigned>(plan.gate_up.size());
    if (!Backend::fits({plan.gate_up.data(), count, kHiddenWidth, kHiddenWidth}) ||
        !Backend::fits({plan.down.data(), count, kExpertWidth, kHiddenWidth})) return Backend::invalid_value;

    auto status = Backend::gather({input.data(), kHiddenWidth, plan.assignment_rows.data(), assignments,
                               kHiddenWidth, workspace.bf16.data(), kHiddenWidth}, stream);
    if (status != Backend::success) return status;
    status = Backend::project(plan.physical_layer, MiniExpertProjection::gate_up,
                              plan.gate_up.data(), count, stream);
    if (status != Backend::success) return status;
    status = Backend::activate(
        {workspace.fp32.data(), workspace.bf16.data(), assignments, kExpertWidth, kHiddenWidth, kExpertWidth},
        kMlpGateSoftcap, kMlpUpSoftcap, stream);
    if (status != Backend::success) return status;
    status = Backend::project(plan.physical_layer, MiniExpertProjection::down,
                              plan.down.data(), count, stream);
    if (status != Backend::success) return status;
    return Backend::combine({workspace.fp32.data(), kHiddenWidth, plan.combine_rows.data(), plan.assignment_weights.data(),
                             plan.rows, kExpertsPerToken, kHiddenWidth, output.data(), kHiddenWidth}, stream);
}

}  // namespace strix::inkling2
