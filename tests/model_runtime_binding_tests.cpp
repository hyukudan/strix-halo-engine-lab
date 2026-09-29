#include "strix/model_runtime_binding.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::ModelManifest fixture_manifest() {
    strix::ModelManifest manifest;
    manifest.architecture = "fixture";

    strix::ManifestTensor embedding;
    embedding.name = "token_embd.weight";
    embedding.dimensions = {64, 128};
    embedding.ggml_type = 1;
    embedding.byte_size = 16'384;
    embedding.absolute_offset = 4096;
    embedding.role = strix::TensorRole::embedding;
    embedding.residency =
        strix::ResidencyPolicy::resident;
    embedding.placement.kind =
        strix::PlacementKind::local;
    embedding.placement.node = 0;
    manifest.tensors.push_back(embedding);

    strix::ManifestTensor remote;
    remote.name = "blk.0.remote.weight";
    remote.dimensions = {64, 64};
    remote.ggml_type = 1;
    remote.byte_size = 8192;
    remote.absolute_offset = 20'480;
    remote.role =
        strix::TensorRole::dense_projection;
    remote.residency =
        strix::ResidencyPolicy::resident;
    remote.placement.kind =
        strix::PlacementKind::remote;
    remote.placement.node = 1;
    manifest.tensors.push_back(remote);

    manifest.total_tensor_bytes =
        embedding.byte_size +
        remote.byte_size;
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);
    return manifest;
}

}  // namespace

int main() {
    const auto manifest =
        fixture_manifest();

    expect(manifest.layout_identity != 0,
           "fixture manifest has non-zero layout identity");

    strix::ModelRuntimeBinding binding;
    binding.manifest_layout_identity =
        manifest.layout_identity;
    binding.model_content_identity =
        0x12345678ULL;

    strix::RuntimeTensorBinding local;
    local.tensor_index = 0;
    local.storage =
        strix::RuntimeTensorStorage::resident;
    local.address =
        static_cast<std::uintptr_t>(
            0x60000000ULL);
    local.bytes =
        manifest.tensors[0].byte_size;
    local.allocation_generation = 3;
    local.owner_node = 0;
    binding.tensors.push_back(local);

    strix::RuntimeTensorBinding remote;
    remote.tensor_index = 1;
    remote.storage =
        strix::RuntimeTensorStorage::remote;
    remote.bytes =
        manifest.tensors[1].byte_size;
    remote.owner_node = 1;
    binding.tensors.push_back(remote);

    strix::validate_model_runtime_binding(
        manifest, binding);

    const auto* embedding =
        strix::find_runtime_tensor_binding(
            manifest,
            binding,
            "token_embd.weight");
    expect(embedding != nullptr &&
               embedding->address ==
                   local.address &&
               embedding->allocation_generation == 3,
           "local tensor resolves by manifest name");

    const auto* second =
        strix::find_runtime_tensor_binding(
            binding, 1);
    expect(second != nullptr &&
               second->storage ==
                   strix::RuntimeTensorStorage::remote &&
               second->address == 0,
           "remote tensor resolves without fake local address");

    auto wrong_layout = binding;
    ++wrong_layout.manifest_layout_identity;
    bool layout_rejected = false;
    try {
        strix::validate_model_runtime_binding(
            manifest, wrong_layout);
    } catch (const std::invalid_argument&) {
        layout_rejected = true;
    }
    expect(layout_rejected,
           "runtime binding rejects manifest identity mismatch");

    auto zero_content = binding;
    zero_content.model_content_identity = 0;
    bool content_rejected = false;
    try {
        strix::validate_model_runtime_binding(
            manifest, zero_content);
    } catch (const std::invalid_argument&) {
        content_rejected = true;
    }
    expect(content_rejected,
           "runtime binding requires explicit model content identity");

    auto undersized = binding;
    --undersized.tensors[0].bytes;
    bool size_rejected = false;
    try {
        strix::validate_model_runtime_binding(
            manifest, undersized);
    } catch (const std::invalid_argument&) {
        size_rejected = true;
    }
    expect(size_rejected,
           "runtime binding rejects undersized local tensor");

    auto remote_address = binding;
    remote_address.tensors[1].address =
        static_cast<std::uintptr_t>(
            0x70000000ULL);
    bool remote_address_rejected = false;
    try {
        strix::validate_model_runtime_binding(
            manifest, remote_address);
    } catch (const std::invalid_argument&) {
        remote_address_rejected = true;
    }
    expect(remote_address_rejected,
           "remote tensor cannot masquerade as locally addressable");

    auto overflow = binding;
    overflow.tensors[0].address =
        std::numeric_limits<std::uintptr_t>::max() -
        static_cast<std::uintptr_t>(
            manifest.tensors[0].byte_size / 2U);
    bool overflow_rejected = false;
    try {
        strix::validate_model_runtime_binding(
            manifest, overflow);
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    expect(overflow_rejected,
           "runtime binding rejects address-range overflow");

    if (failures == 0) {
        std::cout << "model runtime binding tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
