#pragma once

#include "strix/model_manifest.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace strix {

struct ModelTensorSelector {
    std::optional<TensorRole> role;
    std::optional<std::uint32_t> layer;
    std::optional<PlacementKind> placement;
    std::string_view exact_name;
    std::string_view name_contains;
};

struct ModelTensorIndex {
    std::uint64_t manifest_layout_identity = 0;
    std::vector<std::uint32_t> name_order;
};

[[nodiscard]] ModelTensorIndex build_model_tensor_index(
    const ModelManifest& manifest);

[[nodiscard]] const ManifestTensor* find_manifest_tensor(
    const ModelManifest& manifest,
    const ModelTensorIndex& index,
    std::string_view exact_name) noexcept;

[[nodiscard]] std::vector<std::uint32_t> select_manifest_tensors(
    const ModelManifest& manifest,
    const ModelTensorIndex& index,
    const ModelTensorSelector& selector);

[[nodiscard]] std::uint32_t select_unique_manifest_tensor(
    const ModelManifest& manifest,
    const ModelTensorIndex& index,
    const ModelTensorSelector& selector);

}  // namespace strix
