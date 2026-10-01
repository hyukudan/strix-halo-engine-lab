#include "strix/moe_plan.hpp"

#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("MoE route byte calculation overflow");
    }
    return left * right;
}

}  // namespace


MoeExpertStoragePlan build_moe_expert_storage_plan(
    const MoeExpertStorageRequest& request) {
    if (request.hidden_size == 0 ||
        request.intermediate_size == 0 ||
        request.expert_count == 0) {
        throw std::invalid_argument(
            "MoE storage dimensions must be non-zero");
    }

    const auto gate_geometry =
        gguf::tensor_type_geometry(request.gate_up_ggml_type);
    const auto down_geometry =
        gguf::tensor_type_geometry(request.down_ggml_type);
    if (!gate_geometry.has_value() || !down_geometry.has_value()) {
        throw std::invalid_argument(
            "MoE storage type has no pinned GGML geometry");
    }

    MoeExpertStoragePlan plan;
    plan.gate_up_compatible =
        request.hidden_size % gate_geometry->block_elements == 0;
    plan.down_compatible =
        request.intermediate_size % down_geometry->block_elements == 0;

    if (plan.gate_up_compatible) {
        const auto blocks_per_output =
            request.hidden_size / gate_geometry->block_elements;
        const auto gate_outputs =
            checked_mul(request.intermediate_size, 2ULL);
        plan.gate_up_bytes_per_expert =
            checked_mul(
                checked_mul(blocks_per_output,
                            gate_geometry->block_bytes),
                gate_outputs);
    }

    if (plan.down_compatible) {
        const auto blocks_per_output =
            request.intermediate_size / down_geometry->block_elements;
        plan.down_bytes_per_expert =
            checked_mul(
                checked_mul(blocks_per_output,
                            down_geometry->block_bytes),
                request.hidden_size);
    }

    if (plan.gate_up_compatible && plan.down_compatible) {
        if (plan.down_bytes_per_expert >
            std::numeric_limits<std::uint64_t>::max() -
                plan.gate_up_bytes_per_expert) {
            throw std::overflow_error("MoE expert storage addition overflow");
        }
        plan.total_bytes_per_expert =
            plan.gate_up_bytes_per_expert +
            plan.down_bytes_per_expert;
        plan.total_bytes_all_experts =
            checked_mul(plan.total_bytes_per_expert,
                        request.expert_count);
    }

    return plan;
}

MoeRoutePlan build_moe_route_plan(const MoeRouteRequest& request) {
    if (request.rows == 0 || request.experts_per_row == 0 ||
        request.expert_count == 0 ||
        request.experts_per_row > request.expert_count) {
        throw std::invalid_argument("invalid MoE route geometry");
    }

    const auto expected_assignments =
        checked_mul(request.rows, request.experts_per_row);
    if (request.selected_experts.size() != expected_assignments) {
        throw std::invalid_argument(
            "selected expert count does not match rows * experts_per_row");
    }

    MoeRoutePlan plan;
    plan.rows = request.rows;
    plan.experts_per_row = request.experts_per_row;
    plan.expert_count = request.expert_count;
    plan.assignments = expected_assignments;
    plan.packed_weight_bytes_per_expert =
        request.packed_weight_bytes_per_expert;
    plan.expert_assignment_counts.assign(request.expert_count, 0);
    plan.touched_expert_bitmap.assign(
        (static_cast<std::size_t>(request.expert_count) + 63U) / 64U,
        0);

    for (std::uint32_t row = 0; row < request.rows; ++row) {
        const auto begin =
            static_cast<std::size_t>(row) * request.experts_per_row;
        for (std::uint32_t slot = 0; slot < request.experts_per_row; ++slot) {
            const auto expert = request.selected_experts[begin + slot];
            if (expert >= request.expert_count) {
                throw std::invalid_argument("selected expert is out of range");
            }
            for (std::uint32_t previous = 0; previous < slot; ++previous) {
                if (request.selected_experts[begin + previous] == expert) {
                    throw std::invalid_argument(
                        "one MoE row selects the same expert more than once");
                }
            }
            ++plan.expert_assignment_counts[expert];
        }
    }

    plan.minimum_rows_per_touched_expert =
        std::numeric_limits<std::uint32_t>::max();
    std::uint64_t touched_assignments = 0;
    for (std::uint32_t expert = 0; expert < request.expert_count; ++expert) {
        const auto count = plan.expert_assignment_counts[expert];
        if (count == 0) {
            continue;
        }
        plan.touched_experts.push_back(expert);
        plan.touched_expert_bitmap[expert / 64U] |=
            std::uint64_t{1} << (expert % 64U);
        plan.maximum_rows_per_expert =
            std::max(plan.maximum_rows_per_expert, count);
        plan.minimum_rows_per_touched_expert =
            std::min(plan.minimum_rows_per_touched_expert, count);
        touched_assignments += count;
    }
    plan.distinct_experts =
        static_cast<std::uint32_t>(plan.touched_experts.size());
    if (plan.distinct_experts == 0) {
        throw std::logic_error("valid MoE route unexpectedly touches no experts");
    }
    plan.mean_rows_per_touched_expert =
        static_cast<double>(touched_assignments) /
        static_cast<double>(plan.distinct_experts);

    plan.expert_offsets.resize(
        static_cast<std::size_t>(request.expert_count) + 1U, 0);
    for (std::uint32_t expert = 0; expert < request.expert_count; ++expert) {
        plan.expert_offsets[static_cast<std::size_t>(expert) + 1U] =
            plan.expert_offsets[expert] +
            plan.expert_assignment_counts[expert];
    }

    plan.assignment_rows.resize(static_cast<std::size_t>(plan.assignments));
    plan.assignment_slots.resize(static_cast<std::size_t>(plan.assignments));
    auto grouping_cursors = plan.expert_offsets;
    for (std::uint32_t row = 0; row < request.rows; ++row) {
        const auto begin =
            static_cast<std::size_t>(row) * request.experts_per_row;
        for (std::uint32_t slot = 0; slot < request.experts_per_row; ++slot) {
            const auto expert = request.selected_experts[begin + slot];
            auto& cursor = grouping_cursors[expert];
            const auto output = static_cast<std::size_t>(cursor++);
            plan.assignment_rows[output] = row;
            plan.assignment_slots[output] = slot;
        }
    }

    if (request.rows > 1) {
        std::uint64_t overlap_sum = 0;
        for (std::uint32_t row = 1; row < request.rows; ++row) {
            const auto previous_begin =
                static_cast<std::size_t>(row - 1U) * request.experts_per_row;
            const auto current_begin =
                static_cast<std::size_t>(row) * request.experts_per_row;
            std::uint32_t overlap = 0;
            for (std::uint32_t left = 0; left < request.experts_per_row; ++left) {
                const auto expert =
                    request.selected_experts[previous_begin + left];
                for (std::uint32_t right = 0; right < request.experts_per_row; ++right) {
                    if (expert ==
                        request.selected_experts[current_begin + right]) {
                        ++overlap;
                        break;
                    }
                }
            }
            overlap_sum += overlap;
        }
        plan.mean_adjacent_row_overlap =
            static_cast<double>(overlap_sum) /
            static_cast<double>(
                static_cast<std::uint64_t>(request.rows - 1U) *
                request.experts_per_row);
    }

    plan.direct_assignment_weight_bytes =
        checked_mul(plan.assignments, request.packed_weight_bytes_per_expert);
    plan.ideal_grouped_weight_bytes =
        checked_mul(plan.distinct_experts,
                    request.packed_weight_bytes_per_expert);
    plan.ideal_grouped_weight_bytes_saved =
        plan.direct_assignment_weight_bytes >= plan.ideal_grouped_weight_bytes
            ? plan.direct_assignment_weight_bytes -
                  plan.ideal_grouped_weight_bytes
            : 0;

    return plan;
}

}  // namespace strix


namespace strix {

MoeGroupingPlan build_moe_grouping_plan(
    const MoeRouteRequest& request) {
    // Route planning owns the canonical fused grouping metadata so callers
    // cannot observe a cost plan and grouping plan built from different
    // validation or ordering rules.
    auto route = build_moe_route_plan(request);

    MoeGroupingPlan plan;
    plan.expert_count = request.expert_count;
    plan.assignments = route.assignments;
    plan.expert_offsets = std::move(route.expert_offsets);
    plan.assignment_rows = std::move(route.assignment_rows);
    plan.assignment_slots = std::move(route.assignment_slots);
    return plan;
}

}  // namespace strix


namespace strix {
namespace {

std::uint32_t largest_supported_chunk(std::uint64_t remaining) {
    if (remaining >= 8) return 8;
    if (remaining >= 4) return 4;
    if (remaining >= 2) return 2;
    if (remaining >= 1) return 1;
    return 0;
}

}  // namespace

MoeExpertExecutionPlan build_moe_expert_execution_plan(
    const MoeRouteRequest& request) {
    const auto route = build_moe_route_plan(request);
    const auto grouping = build_moe_grouping_plan(request);

    MoeExpertExecutionPlan plan;
    plan.assignments = route.assignments;
    plan.direct_weight_traversals = route.assignments;
    plan.direct_weight_bytes = checked_mul(
        route.assignments, request.packed_weight_bytes_per_expert);

    for (std::uint32_t expert = 0; expert < request.expert_count; ++expert) {
        auto offset = grouping.expert_offsets[expert];
        auto remaining =
            grouping.expert_offsets[static_cast<std::size_t>(expert) + 1U] -
            offset;

        while (remaining != 0) {
            const auto rows = largest_supported_chunk(remaining);
            if (rows == 0) {
                throw std::logic_error("MoE expert batch decomposition failed");
            }
            plan.batches.push_back(
                MoeExpertBatch{expert, offset, rows});
            offset += rows;
            remaining -= rows;
        }
    }

    plan.expert_batches =
        static_cast<std::uint64_t>(plan.batches.size());
    plan.grouped_weight_traversals = plan.expert_batches;
    plan.grouped_weight_bytes = checked_mul(
        plan.grouped_weight_traversals,
        request.packed_weight_bytes_per_expert);
    plan.grouped_weight_bytes_saved =
        plan.direct_weight_bytes >= plan.grouped_weight_bytes
            ? plan.direct_weight_bytes - plan.grouped_weight_bytes
            : 0;

    return plan;
}

}  // namespace strix
