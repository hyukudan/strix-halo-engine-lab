#include "strix/mtp_draft_binding.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct Fixture {
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding runtime;

    Fixture() {
        auto add =
            [&](const char* name,
                strix::TensorRole role,
                std::uint64_t bytes) {
                strix::ManifestTensor tensor;
                tensor.name = name;
                tensor.dimensions = {64, 64};
                tensor.ggml_type = 1;
                tensor.byte_size = bytes;
                tensor.absolute_offset =
                    4096 +
                    manifest.total_tensor_bytes;
                tensor.role = role;
                tensor.residency =
                    strix::ResidencyPolicy::resident;
                tensor.placement.kind =
                    strix::PlacementKind::local;
                tensor.placement.node = 0;
                manifest.total_tensor_bytes +=
                    bytes;
                manifest.tensors.push_back(
                    tensor);
            };

        manifest.architecture = "fixture";
        add(
            "token_embd.weight",
            strix::TensorRole::embedding,
            8192);
        add(
            "mtp.0.proj.weight",
            strix::TensorRole::mtp,
            4096);
        add(
            "mtp.0.norm.weight",
            strix::TensorRole::mtp,
            1024);
        manifest.layout_identity =
            strix::model_manifest_layout_identity(
                manifest);

        runtime.manifest_layout_identity =
            manifest.layout_identity;
        runtime.model_content_identity.bytes[0] =
            0x42;

        for (std::uint32_t index = 0;
             index < manifest.tensors.size();
             ++index) {
            strix::RuntimeTensorBinding binding;
            binding.tensor_index = index;
            binding.storage =
                strix::RuntimeTensorStorage::resident;
            binding.address =
                static_cast<std::uintptr_t>(
                    0x10000000ULL +
                    0x10000ULL * index);
            binding.bytes =
                manifest.tensors[index].byte_size;
            binding.allocation_generation =
                static_cast<std::uint64_t>(
                    index + 1U);
            runtime.tensors.push_back(binding);
        }
    }
};

}  // namespace

int main() {
    {
        Fixture fixture;
        const auto draft =
            strix::build_mtp_draft_runtime_binding(
                fixture.manifest,
                fixture.runtime);

        expect(draft.tensor_indices ==
                   std::vector<std::uint32_t>({1, 2}),
               "MTP draft binding selects only MTP-role tensors");
        expect(draft.total_tensor_bytes == 5120,
               "MTP draft binding accounts exact selected tensor bytes");
        expect(draft.compatibility_identity != 0 &&
                   draft.physical_binding_identity != 0,
               "MTP draft binding exposes stable and physical identities");

        strix::validate_mtp_draft_runtime_binding(
            fixture.manifest,
            fixture.runtime,
            draft);

        auto relocated_runtime =
            fixture.runtime;
        relocated_runtime.tensors[1].address +=
            0x100000ULL;
        ++relocated_runtime.tensors[1]
              .allocation_generation;
        const auto relocated =
            strix::build_mtp_draft_runtime_binding(
                fixture.manifest,
                relocated_runtime);
        expect(relocated.compatibility_identity ==
                   draft.compatibility_identity &&
                   relocated.physical_binding_identity !=
                       draft.physical_binding_identity,
               "MTP compatibility survives relocation while physical identity changes");

        auto changed_content =
            fixture.runtime;
        ++changed_content.model_content_identity.bytes[0];
        const auto changed =
            strix::build_mtp_draft_runtime_binding(
                fixture.manifest,
                changed_content);
        expect(changed.compatibility_identity !=
                   draft.compatibility_identity,
               "MTP compatibility binds model content identity");
    }

    {
        Fixture fixture;
        for (auto& tensor :
             fixture.manifest.tensors) {
            tensor.role =
                strix::TensorRole::dense_projection;
        }
        fixture.manifest.layout_identity =
            strix::model_manifest_layout_identity(
                fixture.manifest);
        fixture.runtime.manifest_layout_identity =
            fixture.manifest.layout_identity;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::build_mtp_draft_runtime_binding(
                    fixture.manifest,
                    fixture.runtime));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected,
               "MTP draft binding rejects checkpoint without MTP tensors");
    }

    {
        Fixture fixture;
        auto draft =
            strix::build_mtp_draft_runtime_binding(
                fixture.manifest,
                fixture.runtime);
        ++draft.total_tensor_bytes;

        bool rejected = false;
        try {
            strix::validate_mtp_draft_runtime_binding(
                fixture.manifest,
                fixture.runtime,
                draft);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected,
               "MTP draft validation rejects stale derived binding");
    }

    if (failures == 0) {
        std::cout << "MTP draft binding tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
