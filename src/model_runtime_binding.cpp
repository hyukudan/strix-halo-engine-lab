#include "strix/model_runtime_binding.hpp"

#include <limits>
#include <stdexcept>

namespace strix {

bool ModelContentIdentity::valid() const noexcept {
    for (const auto byte : bytes) {
        if (byte != 0) {
            return true;
        }
    }
    return false;
}

void validate_model_runtime_binding(
    const ModelManifest& manifest,
    const ModelRuntimeBinding& binding) {
    if (manifest.layout_identity == 0 ||
        binding.manifest_layout_identity !=
            manifest.layout_identity) {
        throw std::invalid_argument(
            "model runtime binding manifest identity mismatch");
    }
    if (!binding.model_content_identity.valid()) {
        throw std::invalid_argument(
            "model runtime binding requires non-zero content identity");
    }
    if (binding.tensors.size() !=
        manifest.tensors.size()) {
        throw std::invalid_argument(
            "model runtime binding tensor count mismatch");
    }

    for (std::size_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        const auto& tensor =
            manifest.tensors[index];
        const auto& runtime =
            binding.tensors[index];

        if (runtime.tensor_index != index) {
            throw std::invalid_argument(
                "model runtime tensor indices must be dense and ordered");
        }
        if (runtime.bytes < tensor.byte_size) {
            throw std::invalid_argument(
                "model runtime tensor binding is smaller than manifest tensor");
        }

        const bool expected_remote =
            tensor.placement.kind ==
            PlacementKind::remote;
        const bool runtime_remote =
            runtime.storage ==
            RuntimeTensorStorage::remote;

        if (expected_remote != runtime_remote) {
            throw std::invalid_argument(
                "model runtime storage does not match tensor placement");
        }
        if (runtime.owner_node !=
            tensor.placement.node) {
            throw std::invalid_argument(
                "model runtime tensor owner does not match placement");
        }

        if (runtime_remote) {
            if (runtime.address != 0) {
                throw std::invalid_argument(
                    "remote tensor must not expose a local address");
            }
            // A remote tensor is identified by placement/transport state rather
            // than a local allocation generation.
            if (runtime.allocation_generation != 0) {
                throw std::invalid_argument(
                    "remote tensor must not expose local allocation generation");
            }
            continue;
        }

        if (runtime.address == 0) {
            throw std::invalid_argument(
                "local tensor runtime binding has null address");
        }
        if (runtime.allocation_generation == 0) {
            throw std::invalid_argument(
                "local tensor allocation generation must be non-zero");
        }

        const auto maximum =
            std::numeric_limits<std::uintptr_t>::max();
        if (tensor.byte_size >
            static_cast<std::uint64_t>(
                maximum - runtime.address)) {
            throw std::overflow_error(
                "model runtime tensor address range overflows uintptr_t");
        }
    }
}

const RuntimeTensorBinding*
find_runtime_tensor_binding(
    const ModelRuntimeBinding& binding,
    std::uint32_t tensor_index) noexcept {
    if (tensor_index >= binding.tensors.size()) {
        return nullptr;
    }
    const auto& tensor =
        binding.tensors[tensor_index];
    return tensor.tensor_index == tensor_index
        ? &tensor
        : nullptr;
}

const RuntimeTensorBinding*
find_runtime_tensor_binding(
    const ModelManifest& manifest,
    const ModelRuntimeBinding& binding,
    std::string_view tensor_name) {
    validate_model_runtime_binding(
        manifest, binding);

    for (std::size_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        if (manifest.tensors[index].name ==
            tensor_name) {
            return &binding.tensors[index];
        }
    }
    return nullptr;
}

}  // namespace strix
