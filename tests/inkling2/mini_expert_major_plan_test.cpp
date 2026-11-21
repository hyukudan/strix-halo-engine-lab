#include "inkling2/inkling2_mini_config.hpp"
#include "moe_core/expert_major_plan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

int main() {
    using strix::MoeRouteRequest;
    using strix::moe_core::build_expert_major_plan;

    constexpr std::array<std::uint32_t, 12> ids{5, 2, 0, 2, 5, 1, 5, 1, 0, 5, 2, 1};
    constexpr std::array<float, 12> weights{0.25F, -0.5F, 1.25F, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const MoeRouteRequest small{4, 3, 7, std::numeric_limits<std::uint64_t>::max(), ids};
    const auto plan = build_expert_major_plan(small, weights, 2);
    constexpr std::array<std::uint64_t, 8> expert_offsets{0, 2, 5, 8, 8, 8, 12, 12};
    constexpr std::array<std::uint32_t, 12> assignment_rows{0, 2, 1, 2, 3, 0, 1, 3, 0, 1, 2, 3};
    constexpr std::array<std::uint32_t, 12> assignment_slots{2, 2, 2, 1, 2, 1, 0, 1, 0, 1, 0, 0};
    constexpr std::array<float, 12> assignment_weights{1.25F, 7, 4, 6, 10, -0.5F, 2, 9, 0.25F, 3, 5, 8};
    constexpr std::array<std::uint32_t, 12> combine_rows{0, 5, 8, 2, 6, 9, 1, 3, 10, 4, 7, 11};
    if (plan.grouping.expert_count != 7 || plan.grouping.assignments != 12 ||
        !std::ranges::equal(plan.grouping.expert_offsets, expert_offsets) ||
        !std::ranges::equal(plan.grouping.assignment_rows, assignment_rows) ||
        !std::ranges::equal(plan.grouping.assignment_slots, assignment_slots) ||
        !std::ranges::equal(plan.assignment_weights, assignment_weights) ||
        !std::ranges::equal(plan.combine_rows, combine_rows)) return 1;

    constexpr std::array<std::array<std::uint32_t, 3>, 7> tiles{{
        {0, 0, 2}, {1, 2, 2}, {1, 4, 1}, {2, 5, 2}, {2, 7, 1}, {5, 8, 2}, {5, 10, 2},
    }};
    if (plan.tiles.size() != tiles.size()) return 2;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const auto& tile = plan.tiles[i];
        if (std::array{tile.expert, tile.first_assignment, tile.rows} != tiles[i]) return 3;
    }

    using namespace strix::inkling2;
    constexpr std::uint32_t row_tile = 128;
    constexpr std::uint32_t rows = row_tile + 3;
    std::array<std::uint32_t, rows * kExpertsPerToken> mini_ids{};
    std::array<float, rows * kExpertsPerToken> mini_weights{};
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t slot = 0; slot < kExpertsPerToken; ++slot) {
            const auto i = row * kExpertsPerToken + slot;
            mini_ids[i] = slot == 4 ? 127U : (row * 29U + slot * 17U) % 127U;
            mini_weights[i] = static_cast<float>(i) * 0.03125F;
        }
    }
    const MoeRouteRequest mini{rows, kExpertsPerToken, kExperts, 0, mini_ids};
    const auto grouped = build_expert_major_plan(mini, mini_weights, row_tile);
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::uint32_t previous_expert = 0;
        for (std::uint32_t rank = 0; rank < kExpertsPerToken; ++rank) {
            const auto group_row = grouped.combine_rows[row * kExpertsPerToken + rank];
            if (grouped.grouping.assignment_rows[group_row] != row) return 4;
            const auto original = row * kExpertsPerToken + grouped.grouping.assignment_slots[group_row];
            const auto expert = mini_ids[original];
            if ((rank != 0 && expert <= previous_expert) || mini_weights[original] != grouped.assignment_weights[group_row] ||
                group_row < grouped.grouping.expert_offsets[expert] ||
                group_row >= grouped.grouping.expert_offsets[expert + 1U]) return 5;
            previous_expert = expert;
        }
    }
    if (grouped.tiles.size() < 2) return 6;
    const auto& first = grouped.tiles[grouped.tiles.size() - 2U];
    const auto& tail = grouped.tiles.back();
    if (first.expert != 127 || first.rows != row_tile || tail.expert != 127 || tail.rows != 3 ||
        first.first_assignment + first.rows != tail.first_assignment) return 7;

    const auto rejects = [](const MoeRouteRequest& request, std::span<const float> route_weights, std::uint32_t tile) {
        try {
            (void)build_expert_major_plan(request, route_weights, tile);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    auto invalid_weights = weights;
    invalid_weights[3] = std::numeric_limits<float>::quiet_NaN();
    auto invalid_ids = ids;
    invalid_ids[1] = invalid_ids[0];
    const MoeRouteRequest duplicate{4, 3, 7, 0, invalid_ids};
    if (!rejects(small, weights, 0) || !rejects(small, invalid_weights, 2) ||
        !rejects(small, std::span(weights).first(11), 2) || !rejects(duplicate, weights, 2)) return 8;
    invalid_ids[1] = 7;
    return rejects(duplicate, weights, 2) ? 0 : 9;
}
