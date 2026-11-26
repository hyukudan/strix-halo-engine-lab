#pragma once

#include "strix/weight/registry.hpp"

#include <string>
#include <vector>

namespace strix::serve { class Json; }

namespace strix::weight {

struct PackRouteCount {
    std::uint64_t layer{}, expert{}, count{};
};

// The same route-count input is used for planning and warming packed banks.
[[nodiscard]] std::vector<PackRouteCount> pack_route_counts(const serve::Json& routes);

struct PackPlanEntry {
    std::string tensor;
    FormatId storage;  // Empty for a tensor kept in its source representation.
    std::uint64_t bytes{};
};

struct PackPlan {
    std::vector<PackPlanEntry> tensors;
    std::uint64_t bytes{}, budget{}, reserve{}, hot_experts{};
    [[nodiscard]] bool fits() const { return bytes <= budget - reserve; }
};

[[nodiscard]] FormatId strixpack_format(std::string_view name);
[[nodiscard]] std::string_view strixpack_label(FormatId format);
[[nodiscard]] std::uint64_t strixpack_matrix_bytes(FormatId format, std::uint64_t rows, std::uint64_t columns);

// Expert priority and projection geometry come from the description. A hot
// expert is promoted across all its projections, bounded by the budget.
[[nodiscard]] PackPlan plan_strix_pack(const serve::Json& description, FormatId cold,
                                      std::uint64_t hot_limit, std::uint64_t budget, std::uint64_t reserve);

}  // namespace strix::weight
