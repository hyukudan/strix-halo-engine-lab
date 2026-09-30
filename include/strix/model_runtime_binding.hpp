#pragma once

#include "strix/model_manifest.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace strix {

struct ModelContentIdentity {
    std::array<std::uint8_t, 32> bytes{};

    [[nodiscard]] bool valid() const noexcept;

    friend constexpr bool operator==(
        const ModelContentIdentity&,
        const ModelContentIdentity&) = default;
};

enum class RuntimeTensorStorage {
    resident,
    file_mapped,
    registered_file_mapped,
    repacked_cache,
    remote,
};

struct RuntimeTensorBinding {
    std::uint32_t tensor_index = 0;
    RuntimeTensorStorage storage =
        RuntimeTensorStorage::resident;
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t allocation_generation = 0;
    std::uint32_t owner_node = 0;
};

struct ModelRuntimeBinding {
    std::uint64_t manifest_layout_identity = 0;

    // Content identity is supplied by the loader after hashing all
    // execution-relevant model files. This is deliberately separate from the
    // manifest layout identity.
    ModelContentIdentity model_content_identity;

    std::vector<RuntimeTensorBinding> tensors;
};

[[nodiscard]] bool runtime_tensor_gpu_visible(
    RuntimeTensorStorage storage) noexcept;

[[nodiscard]] bool runtime_tensor_file_backed(
    RuntimeTensorStorage storage) noexcept;

void validate_model_runtime_binding(
    const ModelManifest& manifest,
    const ModelRuntimeBinding& binding);

[[nodiscard]] const RuntimeTensorBinding*
find_runtime_tensor_binding(
    const ModelRuntimeBinding& binding,
    std::uint32_t tensor_index) noexcept;

[[nodiscard]] const RuntimeTensorBinding*
find_runtime_tensor_binding(
    const ModelManifest& manifest,
    const ModelRuntimeBinding& binding,
    std::string_view tensor_name);

}  // namespace strix
