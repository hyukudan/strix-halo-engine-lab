#include "strix/moe_plan.hpp"

#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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

MoeRoutingSelection select_moe_experts_deterministic(
    std::span<const float> logits,
    std::uint32_t top_k,
    bool normalize_selected) {
    if (logits.empty() || top_k == 0 || top_k > logits.size()) {
        throw std::invalid_argument("invalid MoE top-k geometry");
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (const auto logit : logits) {
        if (!std::isfinite(logit)) {
            throw std::invalid_argument("MoE router logits must be finite");
        }
        maximum = std::max(maximum, logit);
    }

    std::vector<float> probabilities(logits.size(), 0.0F);
    float denominator = 0.0F;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const auto probability = std::exp(logits[index] - maximum);
        probabilities[index] = probability;
        denominator += probability;
    }
    if (!(denominator > 0.0F) || !std::isfinite(denominator)) {
        throw std::runtime_error("MoE softmax normalization is invalid");
    }
    for (auto& probability : probabilities) {
        probability /= denominator;
    }

    std::vector<std::uint32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(
        order.begin(),
        order.end(),
        [&probabilities](std::uint32_t left, std::uint32_t right) {
            if (probabilities[left] != probabilities[right]) {
                return probabilities[left] > probabilities[right];
            }
            return left < right;
        });
    order.resize(top_k);

    MoeRoutingSelection selection;
    selection.experts = order;
    selection.weights.reserve(top_k);
    float selected_sum = 0.0F;
    for (const auto expert : selection.experts) {
        const auto weight = probabilities[expert];
        selection.weights.push_back(weight);
        selected_sum += weight;
    }

    if (normalize_selected) {
        if (!(selected_sum > 0.0F) || !std::isfinite(selected_sum)) {
            throw std::runtime_error("MoE selected-weight normalization is invalid");
        }
        for (auto& weight : selection.weights) {
            weight /= selected_sum;
        }
    }

    return selection;
}

MoeHierarchicalRoutingPlan build_moe_hierarchical_routing_plan(
    std::uint32_t expert_count,
    std::uint32_t top_k,
    std::uint32_t group_width) {
    if (expert_count == 0 || top_k == 0 ||
        top_k > expert_count || group_width == 0) {
        throw std::invalid_argument(
            "invalid hierarchical MoE routing geometry");
    }

    MoeHierarchicalRoutingPlan plan;
    plan.expert_count = expert_count;
    plan.top_k = top_k;
    plan.group_width = group_width;
    plan.groups =
        static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(expert_count) +
             group_width - 1ULL) /
            group_width);
    plan.local_candidate_scans =
        checked_mul(expert_count, top_k);
    plan.group_reduction_candidates =
        checked_mul(plan.groups, top_k);
    return plan;
}

MoeRoutingSelection select_moe_experts_hierarchical(
    std::span<const float> logits,
    std::uint32_t top_k,
    std::uint32_t group_width,
    bool normalize_selected) {
    if (logits.empty() ||
        logits.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "hierarchical MoE router expert count is invalid");
    }

    const auto plan =
        build_moe_hierarchical_routing_plan(
            static_cast<std::uint32_t>(
                logits.size()),
            top_k,
            group_width);

    float maximum =
        -std::numeric_limits<float>::infinity();
    for (const auto logit : logits) {
        if (!std::isfinite(logit)) {
            throw std::invalid_argument(
                "MoE router logits must be finite");
        }
        maximum =
            std::max(
                maximum,
                logit);
    }

    std::vector<float> probabilities(
        logits.size(),
        0.0F);
    float denominator = 0.0F;
    for (std::size_t index = 0;
         index < logits.size();
         ++index) {
        const auto probability =
            std::exp(
                logits[index] -
                maximum);
        probabilities[index] =
            probability;
        denominator +=
            probability;
    }
    if (!(denominator > 0.0F) ||
        !std::isfinite(denominator)) {
        throw std::runtime_error(
            "MoE softmax normalization is invalid");
    }
    for (auto& probability :
         probabilities) {
        probability /=
            denominator;
    }

    std::vector<std::uint8_t> selected(
        logits.size(),
        0U);
    MoeRoutingSelection selection;
    selection.experts.reserve(top_k);
    selection.weights.reserve(top_k);

    for (std::uint32_t slot = 0;
         slot < top_k;
         ++slot) {
        float winner_value =
            -std::numeric_limits<float>::infinity();
        std::uint32_t winner_index =
            plan.expert_count;

        for (std::uint32_t group = 0;
             group < plan.groups;
             ++group) {
            const auto begin =
                static_cast<std::uint64_t>(
                    group) *
                plan.group_width;
            const auto end =
                std::min<std::uint64_t>(
                    begin +
                        plan.group_width,
                    plan.expert_count);

            float group_value =
                -std::numeric_limits<float>::infinity();
            std::uint32_t group_index =
                plan.expert_count;
            for (std::uint64_t raw_index =
                     begin;
                 raw_index < end;
                 ++raw_index) {
                const auto index =
                    static_cast<std::uint32_t>(
                        raw_index);
                if (selected[index] != 0U) {
                    continue;
                }

                const auto value =
                    probabilities[index];
                if (value > group_value ||
                    (value == group_value &&
                     index < group_index)) {
                    group_value = value;
                    group_index = index;
                }
            }

            if (group_index ==
                plan.expert_count) {
                continue;
            }
            if (group_value > winner_value ||
                (group_value == winner_value &&
                 group_index < winner_index)) {
                winner_value =
                    group_value;
                winner_index =
                    group_index;
            }
        }

        if (winner_index ==
            plan.expert_count) {
            throw std::logic_error(
                "hierarchical MoE selection exhausted candidates");
        }
        selected[winner_index] = 1U;
        selection.experts.push_back(
            winner_index);
        selection.weights.push_back(
            winner_value);
    }

    if (normalize_selected) {
        float selected_sum = 0.0F;
        for (const auto weight :
             selection.weights) {
            selected_sum +=
                weight;
        }
        if (!(selected_sum > 0.0F) ||
            !std::isfinite(selected_sum)) {
            throw std::runtime_error(
                "MoE selected-weight normalization is invalid");
        }
        for (auto& weight :
             selection.weights) {
            weight /=
                selected_sum;
        }
    }

    return selection;
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
