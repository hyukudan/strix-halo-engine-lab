#include "strix/mtp_draft_tensor_audit.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

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
    strix::MtpDraftRuntimeBinding binding;

    Fixture() {
        auto add =
            [&](std::string name) {
                strix::ManifestTensor tensor;
                tensor.name = std::move(name);
                tensor.dimensions = {64, 64};
                tensor.ggml_type = 1;
                tensor.byte_size = 4096;
                tensor.absolute_offset =
                    4096 +
                    manifest.total_tensor_bytes;
                tensor.role =
                    strix::TensorRole::mtp;
                manifest.total_tensor_bytes +=
                    tensor.byte_size;
                binding.tensor_indices.push_back(
                    static_cast<std::uint32_t>(
                        manifest.tensors.size()));
                manifest.tensors.push_back(
                    std::move(tensor));
            };

        manifest.architecture = "fixture";
        add("mtp.fc_embedding.weight");
        add("mtp.fc_hidden.weight");
        add("mtp.hyper_connection_mixer.down.weight");
        add("mtp.layers.0.self_attn.q_proj.weight");
        add("mtp.layers.0.mlp.experts.0.gate_proj.weight");
        add("mtp.unresolved.extra.weight");
        manifest.layout_identity =
            strix::model_manifest_layout_identity(
                manifest);

        binding.manifest_layout_identity =
            manifest.layout_identity;
        binding.model_content_identity.bytes[0] =
            0x42;
        binding.compatibility_identity =
            0x1234ULL;
        binding.physical_binding_identity =
            0x5678ULL;
        binding.total_tensor_bytes =
            manifest.total_tensor_bytes;
    }
};

}  // namespace

int main() {
    {
        Fixture fixture;
        const auto manifest_audit =
            strix::audit_mtp_draft_tensor_families(
                fixture.manifest);
        expect(manifest_audit.proposal_identity == 0 &&
                   manifest_audit.structurally_complete(),
               "manifest-only MTP audit works before runtime binding");

        const auto audit =
            strix::audit_mtp_draft_tensor_families(
                fixture.manifest,
                fixture.binding);

        expect(audit.structurally_complete(),
               "MTP audit recognizes all required structural tensor families");
        expect(audit.unclassified_mtp_tensors ==
                   std::vector<std::uint32_t>({5}),
               "MTP audit preserves unresolved tensors instead of guessing their role");
        expect(audit.families.size() == 5,
               "MTP audit emits every required structural family");
    }

    {
        Fixture fixture;
        fixture.manifest.tensors[4].name =
            "mtp.unresolved.moe_removed";
        fixture.manifest.layout_identity =
            strix::model_manifest_layout_identity(
                fixture.manifest);
        fixture.binding.manifest_layout_identity =
            fixture.manifest.layout_identity;

        const auto audit =
            strix::audit_mtp_draft_tensor_families(
                fixture.manifest,
                fixture.binding);
        expect(!audit.structurally_complete() &&
                   audit.missing_required_families ==
                       std::vector<strix::MtpTensorFamily>({
                           strix::MtpTensorFamily::moe}),
               "MTP audit reports missing structural family without enabling execution");
    }

    if (failures == 0) {
        std::cout << "MTP draft tensor audit tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
