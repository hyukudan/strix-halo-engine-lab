#include "strix/moe_plan.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

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
