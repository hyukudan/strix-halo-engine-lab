#pragma once

#include "strix/weight/registry.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::serve { class Json; }
namespace strix::weight::calib { class MomentSet; }

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
    std::string coder;
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

// ---- writers of the planner inputs (the readers are above) ----------------------------------------------

// One routed layer of a measured capture: its index and the name prefix of the moment inputs it owns.
struct MeasuredPackLayer {
    unsigned layer{};
    std::string tensor_prefix;
};

// Publishes a measured pack description into a NEW directory: `model_description` (a "strix-pack-model/v1"
// description) with the "routes" array [layer, expert, count] set to `routes`, as measured-pack-description.json,
// and one "layer-N.moments" partition of `moments` per entry of `layers` (inputs whose name starts with the prefix).
void write_measured_pack_description(const std::filesystem::path& directory,
                                     const std::filesystem::path& model_description,
                                     std::span<const PackRouteCount> routes, const calib::MomentSet& moments,
                                     std::span<const MeasuredPackLayer> layers);

struct PackSliceMatrix {
    std::string tensor;
    std::uint64_t rows{}, columns{};
};

// Writes the "tensor<TAB>format<TAB>bytes" plan of `matrices` stored as `storage` ("sp2", "sp3", ...). The path
// must not exist.
void write_pack_slice_plan(const std::filesystem::path& path, std::string_view storage,
                           std::span<const PackSliceMatrix> matrices);

}  // namespace strix::weight
