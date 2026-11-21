#pragma once

#include "core/dense_row_plan.hpp"
#include "strix/weight/resident_slices.hpp"

#include <array>
#include <variant>

namespace strix::core {

using ResidentRowWeight = std::variant<const std::uint16_t*, weight::ResidentMatrixView>;

struct ResidentRowProjection {
    ResidentRowWeight weights;
    unsigned outputs;
    float* output;
};

// Keep BF16 projections grouped when possible. Each compact projection binds
// its own format and row geometry, with no reinterpretation as BF16.
template <std::size_t Limit, class Status, class Bf16, class Compact>
Status dispatch_resident_rows(std::span<const ResidentRowProjection> projections, Status success, Status invalid,
                              const Bf16& bf16, const Compact& compact) {
    if (projections.empty() || projections.size() > Limit) return invalid;
    std::array<Bf16RowProjection, Limit> run{};
    std::size_t count = 0;
    for (const auto& projection : projections) {
        if (const auto* bits = std::get_if<const std::uint16_t*>(&projection.weights)) {
            run[count++] = {*bits, projection.outputs, projection.output};
            continue;
        }
        if (count != 0) {
            if (auto status = bf16(std::span(run).first(count)); status != success) return status;
            count = 0;
        }
        const auto& matrix = std::get<weight::ResidentMatrixView>(projection.weights);
        if (auto status = compact(matrix, projection.outputs, projection.output); status != success) return status;
    }
    return count != 0 ? bf16(std::span(run).first(count)) : success;
}

}  // namespace strix::core
