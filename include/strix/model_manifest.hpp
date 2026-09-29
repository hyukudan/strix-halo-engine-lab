#pragma once

#include "strix/gguf_reader.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace strix {

enum class TensorRole {
    unknown,
    embedding,
    output_head,
    normalization,
    attention_projection,
    dense_projection,
    moe_router,
    moe_expert,
    recurrent_state,
    sparse_index,
    ngram_table,
    ple_parameter,
    mtp,
};

enum class ResidencyPolicy {
    resident,
    file_backed_candidate,
    bounded_cache_candidate,
};

enum class PlacementKind {
    local,
    remote,
};

struct TensorPlacement {
    PlacementKind kind = PlacementKind::local;
    std::uint32_t node = 0;
};

struct ManifestTensor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t ggml_type = 0;
    std::uint64_t byte_size = 0;
    std::uint64_t absolute_offset = 0;
    TensorRole role = TensorRole::unknown;
    ResidencyPolicy residency = ResidencyPolicy::resident;
    TensorPlacement placement{};
    std::optional<std::uint32_t> layer;
};

struct ModelManifest {
    std::string architecture;
    std::uint64_t layout_identity = 0;
    std::uint64_t total_tensor_bytes = 0;
    std::uint64_t file_backed_candidate_bytes = 0;
    std::uint64_t bounded_cache_candidate_bytes = 0;
    std::uint64_t unknown_tensor_count = 0;
    std::vector<ManifestTensor> tensors;
};

[[nodiscard]] TensorRole classify_tensor_role(std::string_view name) noexcept;
[[nodiscard]] ResidencyPolicy default_residency(TensorRole role) noexcept;
[[nodiscard]] std::string_view to_string(TensorRole role) noexcept;
[[nodiscard]] std::string_view to_string(ResidencyPolicy policy) noexcept;
[[nodiscard]] std::string_view to_string(PlacementKind kind) noexcept;
[[nodiscard]] std::uint64_t model_manifest_layout_identity(
    const ModelManifest& manifest) noexcept;
[[nodiscard]] ModelManifest build_model_manifest(const gguf::FileInfo& info);

}  // namespace strix
