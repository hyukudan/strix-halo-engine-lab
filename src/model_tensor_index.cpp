#include "strix/model_tensor_index.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strix {
namespace {

void validate_index(
    const ModelManifest& manifest,
    const ModelTensorIndex& index) {
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(manifest) ||
        index.manifest_layout_identity !=
            manifest.layout_identity) {
        throw std::invalid_argument(
            "model tensor index manifest identity mismatch");
    }
    if (index.name_order.size() !=
        manifest.tensors.size()) {
        throw std::invalid_argument(
            "model tensor index size mismatch");
    }
}

bool matches_selector(
    const ManifestTensor& tensor,
    const ModelTensorSelector& selector) {
    if (selector.role.has_value() &&
        tensor.role != *selector.role) {
        return false;
    }
    if (selector.layer.has_value() &&
        tensor.layer != selector.layer) {
        return false;
    }
    if (selector.placement.has_value() &&
        tensor.placement.kind !=
            *selector.placement) {
        return false;
    }
    if (!selector.exact_name.empty() &&
        tensor.name != selector.exact_name) {
        return false;
    }
    if (!selector.name_contains.empty() &&
        tensor.name.find(
            selector.name_contains) ==
            std::string::npos) {
        return false;
    }
    if (!selector.name_contains_any.empty()) {
        const auto found =
            std::any_of(
                selector.name_contains_any.begin(),
                selector.name_contains_any.end(),
                [&](std::string_view needle) {
                    return !needle.empty() &&
                           tensor.name.find(needle) !=
                               std::string::npos;
                });
        if (!found) {
            return false;
        }
    }
    return true;
}

}  // namespace

ModelTensorIndex build_model_tensor_index(
    const ModelManifest& manifest) {
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(
                manifest)) {
        throw std::invalid_argument(
            "model tensor index requires a valid manifest layout identity");
    }

    ModelTensorIndex index;
    index.manifest_layout_identity =
        manifest.layout_identity;
    index.name_order.resize(
        manifest.tensors.size());

    for (std::size_t tensor = 0;
         tensor < manifest.tensors.size();
         ++tensor) {
        index.name_order[tensor] =
            static_cast<std::uint32_t>(tensor);
    }

    std::sort(
        index.name_order.begin(),
        index.name_order.end(),
        [&](std::uint32_t left,
            std::uint32_t right) {
            const auto& left_name =
                manifest.tensors[left].name;
            const auto& right_name =
                manifest.tensors[right].name;
            if (left_name != right_name) {
                return left_name < right_name;
            }
            return left < right;
        });

    for (std::size_t position = 1;
         position < index.name_order.size();
         ++position) {
        const auto previous =
            index.name_order[position - 1];
        const auto current =
            index.name_order[position];
        if (manifest.tensors[previous].name ==
            manifest.tensors[current].name) {
            throw std::invalid_argument(
                "model manifest contains duplicate tensor names");
        }
    }

    return index;
}

const ManifestTensor* find_manifest_tensor(
    const ModelManifest& manifest,
    const ModelTensorIndex& index,
    std::string_view exact_name) noexcept {
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(manifest) ||
        index.manifest_layout_identity !=
            manifest.layout_identity ||
        index.name_order.size() !=
            manifest.tensors.size()) {
        return nullptr;
    }

    const auto found =
        std::lower_bound(
            index.name_order.begin(),
            index.name_order.end(),
            exact_name,
            [&](std::uint32_t tensor,
                std::string_view name) {
                return manifest.tensors[tensor].name <
                       name;
            });

    if (found == index.name_order.end() ||
        manifest.tensors[*found].name !=
            exact_name) {
        return nullptr;
    }
    return &manifest.tensors[*found];
}

std::vector<std::uint32_t> select_manifest_tensors(
    const ModelManifest& manifest,
    const ModelTensorIndex& index,
    const ModelTensorSelector& selector) {
    validate_index(manifest, index);

    std::vector<std::uint32_t> selected;
    for (std::uint32_t tensor = 0;
         tensor < manifest.tensors.size();
         ++tensor) {
        if (matches_selector(
                manifest.tensors[tensor],
                selector)) {
            selected.push_back(tensor);
        }
    }
    return selected;
}

std::uint32_t select_unique_manifest_tensor(
    const ModelManifest& manifest,
    const ModelTensorIndex& index,
    const ModelTensorSelector& selector) {
    const auto selected =
        select_manifest_tensors(
            manifest,
            index,
            selector);
    if (selected.empty()) {
        throw std::invalid_argument(
            "model tensor selector matched no tensors");
    }
    if (selected.size() != 1) {
        throw std::invalid_argument(
            "model tensor selector is ambiguous");
    }
    return selected.front();
}

}  // namespace strix
